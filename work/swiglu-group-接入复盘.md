# `SwigluGroup` 融合算子接入复盘

> 本文以 `torchtitan-npu` 当前本地 `master` 为准。
>
> - 本地合并提交：`cd9821e`（合入 PR #468）
> - PR 最终提交：`5a6b658`
> - 对外 converter 名：`npu_swiglu_group`
> - 实现文件：`torchtitan_npu/converters/kernels/swiglu_group.py`
> - 最后更新：2026-08-03

## 1. 最终结论

当前接入没有把 `SwigluGroup` 塞进通用的 `npu_gmm` converter，也没有再维护一个
`npu_gmm_swiglu` 复合 converter，而是把 MoE 的“结构转换”和“激活实现选择”拆成三层：

| Converter | 负责内容 | 默认激活 | 硬件范围 |
| --- | --- | --- | --- |
| `npu_moe_dispatch` | 替换 MoE dispatch 路径，并把共享专家原地适配为 `NpuSharedExperts` | 原生 PyTorch 小算子分解 | 通用 NPU 路径 |
| `npu_gmm` | 把路由专家转换为 `NpuGroupedExperts`，完成两次 GMM 和 `w13` 参数布局 | `torch_npu.npu_swiglu` 及必要的 clamp/score 乘法 | 通用 NPU 路径 |
| `npu_swiglu_group` | 仅把路由专家、共享专家已预留的 activation callable 切到 `cann_ops_nn.swiglu_group` | `cann_ops_nn.swiglu_group` | 仅 A5 |

推荐配置顺序为：

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_swiglu_group"),
```

这套拆分解决了最关键的兼容性问题：

- A3 继续只使用 `npu_moe_dispatch + npu_gmm`，不会导入或调用 A5-only 算子；
- A5 显式追加 `npu_swiglu_group`，只替换两次 GMM 之间以及共享专家中的 activation；
- `npu_gmm` 的参数布局、state dict 和 compile 路径不因 A5 融合能力发生分叉；
- `npu_swiglu_group` 不创建新权重，也不注册 state-dict updater。

当前仓库不再注册 `npu_gmm_swiglu`，测试会明确断言旧名称不存在。

## 2. 背景：融合的到底是哪一段

SwiGLU FFN 的核心计算为：

```text
gate   = w1(x)
up     = w3(x)
hidden = silu(gate) * up
output = w2(hidden)
```

当 gate 和 up 沿最后一维打包为 `[gate | up]` 后，`SwigluGroup` 可以融合：

1. gate/up clamp；
2. `silu(gate) * up`；
3. 路由专家的 routed-score 逐 token 乘法。

它不负责矩阵乘本身，也不替代 GMM。因此路由专家的最终关系是：

```text
GMM-1（x @ w13）
  -> SwigluGroup（clamp + SwiGLU + routed score）
  -> GMM-2（hidden @ w2）
```

共享专家仍然保留独立的 `w1/w2/w3`：

```text
w1(x) ---- gate --\
                  +-> cat([gate, up]) -> SwigluGroup -> w2
w3(x) ------ up --/
```

这里保留 `w1/w2/w3`，可以避免改变 checkpoint key、参数对象、FQN、TP plan 以及其他按
模块名匹配的 converter。

## 3. 当前架构

### 3.1 `npu_moe_dispatch`：提前建立共享专家激活插槽

`torchtitan_npu/converters/kernels/moe_dispatch.py` 在转换 MoE 模块时，会同步处理
`shared_experts`：

```python
shared_experts = getattr(module, "shared_experts", None)
if shared_experts is None or isinstance(shared_experts, NpuSharedExperts):
    return
if not isinstance(shared_experts, FeedForward):
    raise ValueError("NPU MoE shared_experts must be a FeedForward module.")
NpuSharedExperts.convert(shared_experts)
```

`NpuSharedExperts` 定义在 `torchtitan_npu/models/common/moe.py`。转换采用原地
`__class__` 替换，保留：

- `w1/w2/w3` 子模块和参数对象；
- state dict key 与模块 FQN；
- forward/backward hooks；
- training/eval 状态；
- 模块 identity 和自定义属性。

它的 forward 只抽象出 activation callable：

```python
packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
activation_fn = getattr(
    self,
    "_expert_activation_fn",
    native_shared_expert_activation,
)
hidden = activation_fn(packed, getattr(self, "swiglu_limit", None))
return self.w2(hidden)
```

在没有启用 `npu_swiglu_group` 时，默认使用
`native_shared_expert_activation()`，即 clamp 后执行
`F.silu(gate) * up`。因此 `npu_moe_dispatch` 本身不依赖 A5 算子。

### 3.2 `npu_gmm`：为路由专家提供 activation 注入点

`torchtitan_npu/converters/kernels/gmm.py` 仍负责：

- `GroupedExperts -> NpuGroupedExperts`；
- 将 `w1/w3` 合并为路由专家使用的 `w13`；
- 两次 `torch._grouped_mm`；
- GMM state-dict 更新；
- TP、动态 token 和 activation-only compile 相关逻辑。

两次 GMM 中间统一调用传入的 `activation_fn`：

```python
h = torch._grouped_mm(
    x.bfloat16(),
    w13.bfloat16().transpose(-2, -1),
    offs=offsets,
)
h = activation_fn(h, swiglu_limit, routed_scores)
out = torch._grouped_mm(
    h,
    w2.bfloat16().transpose(-2, -1),
    offs=offsets,
).type_as(x)
```

默认 `_expert_activation()` 继续执行原有路径：

```text
可选 clamp -> torch_npu.npu_swiglu -> 可选 routed-score 乘法
```

`NpuGroupedExperts.set_expert_activation()` 是交给融合 converter 的公开交接点：

```python
def set_expert_activation(self, activation_fn):
    self._expert_activation_fn = activation_fn
    self._expert_activation_compile_key = None
```

清空 compile key 很重要：如果 activation 已被编译过，切换实现后不能继续复用旧图。

### 3.3 `npu_swiglu_group`：只选择 A5 融合激活

`NpuSwigluGroupConverter.convert()` 的处理顺序为：

1. 校验设备必须是 A5；
2. 校验模型中已有 `NpuGroupedExperts`，即 `npu_gmm` 已执行；
3. 扫描名为 `shared_experts` 的模块；如果存在但还不是 `NpuSharedExperts`，说明
   `npu_moe_dispatch` 未先执行，直接报错；
4. 导入 `cann_ops_nn.ops`，校验前向和反向 dispatcher 均存在；
5. 将路由专家 activation 切换为 `swiglu_group_activation`；
6. 将共享专家 activation 切换为 `swiglu_group_shared_activation`。

简化后的结构如下：

```text
npu_moe_dispatch
  -> shared_experts: FeedForward -> NpuSharedExperts
  -> shared activation = native_shared_expert_activation

npu_gmm
  -> GroupedExperts -> NpuGroupedExperts
  -> routed activation = _expert_activation

npu_swiglu_group（A5-only）
  -> 校验前置 converter 和 CANN dispatcher
  -> routed activation = swiglu_group_activation
  -> shared activation = swiglu_group_shared_activation
```

`npu_swiglu_group` 不做以下工作：

- 不再次转换 `GroupedExperts`；
- 不创建或重排 `w13/w2`；
- 不负责把普通 `FeedForward` 转为 `NpuSharedExperts`；
- 不修改 DeepSeek-V4 模型 forward；
- 不注册 state-dict updater；
- 不对非 A5 设备静默回退。

## 4. 算子调用约定

### 4.1 Python 入口

当前调用直接落到公开 dispatcher：

```python
return torch.ops.cann_ops_nn.swiglu_group.default(
    h.contiguous(),
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

参数映射如下：

| 参数 | 当前传法 | 原因 |
| --- | --- | --- |
| `x` | `h.contiguous()` | 保证 `[gate | up]` 输入连续 |
| `weight` | routed score 转 `float32` 且 contiguous；共享专家传 `None` | 对齐算子接口，并融合路由权重乘法 |
| `group_index` | `None` | token 分组已经由 dispatch 和 GMM offsets 完成 |
| `clamp_limit` | 有 `swiglu_limit` 时传其浮点值，否则传 `-1.0` | 保持有/无 clamp 语义 |

路由专家调用：

```python
swiglu_group_activation(h, swiglu_limit, routed_scores)
```

共享专家通过窄签名包装：

```python
swiglu_group_shared_activation(h, swiglu_limit)
```

共享专家没有 routed score，因此 wrapper 最终让 `weight=None`。

### 4.2 `group_index=None` 不会丢失专家分组

进入 activation 前已经完成：

1. router 选择专家；
2. token dispatch/reorder；
3. 根据 `num_tokens_per_expert` 构造 GMM offsets；
4. GMM-1 只输出当前实际参与计算的 routed rows。

所以 activation 只需逐行处理现有输入，不需要再表达一次专家边界。

### 4.3 Autograd 归属

当前代码不再定义本地 `_NpuSwigluGroup(torch.autograd.Function)`，也不在 Python
backward 中手动调用反向算子。

`swiglu_group_activation()` 直接调用公开的
`torch.ops.cann_ops_nn.swiglu_group.default`，Autograd 集成由 `cann_ops_nn` 包提供。
converter 只在启用时预检查两个 dispatcher：

```text
cann_ops_nn::swiglu_group
cann_ops_nn::swiglu_group_backward
```

这样做避免在 `torchtitan-npu` 内重复维护 backward context、`y_origin` 重算、梯度返回
顺序等算子内部语义。仓库只负责正确选择和接线公开算子。

### 4.4 空 Tensor 策略

当前实现不再为 `h.numel() == 0` 提供 Python 小算子回退，也不会返回人工构造的空
Tensor。所有输入都直接交给 `cann_ops_nn.swiglu_group`。

这与原 `torch_npu.npu_swiglu` 路径的约束保持一致：当前不支持空 Tensor。单元测试
验证的是“由算子自身拒绝空输入”，而不是“converter 对空输入自动回退”。

因此，不能再把以下内容当作当前能力：

- EP 空 token rank 上的分解式 fallback；
- 空 Tensor 的 eager/compile/backward 等价处理；
- 空输入时绕过 CANN forward/backward。

如果后续训练场景确实会产生空 routed rows，应在算子合同、dispatch 约束或更高层流程中
统一解决，不能在本 converter 内悄悄切换数学实现。

## 5. A5 与 A3 的配置边界

### 5.1 A3/通用路径

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
```

结果为：

- 路由专家：`NpuGroupedExperts + _expert_activation`；
- 共享专家：`NpuSharedExperts + native_shared_expert_activation`；
- 不导入 `cann_ops_nn.ops`；
- 不调用 `SwigluGroup`；
- 保留已验证的 `npu_gmm` 路径。

### 5.2 A5 融合路径

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_swiglu_group"),
```

结果为：

- 路由专家的两次 GMM 之间调用 `SwigluGroup`；
- 共享专家的 `w1/w3` 输出打包后调用 `SwigluGroup`；
- converter 启用时显式检查 A5 和 CANN dispatcher；
- 条件不满足时立即报错，不静默退回原生 activation。

### 5.3 为什么不放进默认 converter 列表

DeepSeek-V3/V3.2/V4 和 Qwen3 MoE 的默认 converter 列表仍包含 `npu_gmm`，但不默认包含
`npu_swiglu_group`。这是有意设计：A5-only 能力必须由 A5 配置显式选择，否则 A3
默认配置会在 converter 阶段失败。

## 6. 编译与 activation checkpointing

### 6.1 activation-only compile

`gmm.py` 的 `compile_expert_activation()` 不关心选择的是 native 还是 fused callable。
它读取当前 `NpuGroupedExperts._expert_activation_fn`，再按以下 key 缓存：

```python
compile_key = (backend, dynamic_tokens)
```

当 `npu_swiglu_group` 调用 `set_expert_activation()` 时，旧 compile key 会被清空，后续
parallelize 阶段会对新的 activation 重新建图。

EP 下 token 数可能动态变化。`dynamic_tokens=True` 时，wrapper 会标记 `h` 以及可选
`routed_scores` 的第 0 维为动态。

### 6.2 selective activation checkpointing

DeepSeek-V4 仍通过是否启用 `npu_gmm` 判断是否保存 grouped-MM 输出。这个判断不需要改成
`npu_swiglu_group`，因为：

- grouped MM 由 `npu_gmm` 引入；
- `npu_swiglu_group` 只改变中间 activation；
- 启用融合时仍然必须包含 `npu_gmm`。

## 7. 最终调用链

### 7.1 A3/通用路由专家

```text
NpuGroupedExperts.forward
  -> npu_grouped_experts_forward
     -> _run_experts_grouped_mm
        -> offsets = cumsum(num_tokens_per_expert)
        -> torch._grouped_mm(x, w13)
        -> _expert_activation
           -> 可选 clamp
           -> torch_npu.npu_swiglu
           -> 可选 routed-score 乘法
        -> torch._grouped_mm(hidden, w2)
```

### 7.2 A5 路由专家

```text
NpuGroupedExperts.forward
  -> npu_grouped_experts_forward
     -> _run_experts_grouped_mm
        -> offsets = cumsum(num_tokens_per_expert)
        -> torch._grouped_mm(x, w13)
        -> swiglu_group_activation
           -> h.contiguous()
           -> routed_scores.float().contiguous()
           -> cann_ops_nn::swiglu_group
        -> torch._grouped_mm(hidden, w2)
```

### 7.3 A3/通用共享专家

```text
NpuSharedExperts.forward
  -> gate = w1(x)
  -> up = w3(x)
  -> packed = cat(gate, up)
  -> native_shared_expert_activation
     -> 可选 clamp
     -> silu(gate) * up
  -> w2(hidden)
```

### 7.4 A5 共享专家

```text
NpuSharedExperts.forward
  -> gate = w1(x)
  -> up = w3(x)
  -> packed = cat(gate, up)
  -> swiglu_group_shared_activation
     -> cann_ops_nn::swiglu_group(weight=None, group_index=None)
  -> w2(hidden)
```

反向由 `cann_ops_nn` 对公开前向 op 注册的 Autograd 路径进入
`cann_ops_nn::swiglu_group_backward`，仓库内没有额外的 Python backward 桥。

## 8. 接入过程中的关键修正

### 8.1 从复合 converter 改为 activation-only converter

早期方案把名称和职责绑定为 `npu_gmm_swiglu`，容易让人误以为它负责完整 GMM
转换。最终名称改为 `npu_swiglu_group`，清楚表达“选择 SwigluGroup 激活实现”。

### 8.2 保护 A3 默认路径

不能让通用 `npu_gmm` 自动探测并进入 A5 算子，否则 A3 会受到硬件和 `cann_ops_nn`
依赖影响。最终采用显式的第三个 converter：

```text
A3/通用：npu_moe_dispatch + npu_gmm
A5：     npu_moe_dispatch + npu_gmm + npu_swiglu_group
```

### 8.3 共享专家结构归基础 converter 管理

早期由 SwigluGroup converter 自己寻找普通 `FeedForward` 并原地转换共享专家。最终把
这一步下沉到 `npu_moe_dispatch`：

- `npu_moe_dispatch` 建立通用 `NpuSharedExperts` 结构并安装 native activation；
- `npu_swiglu_group` 只做 activation selection；
- A3/A5 共享完全相同的模块结构，差异只在 callable；
- 融合 converter 不再承担通用模型结构适配。

### 8.4 直接复用算子包 Autograd

早期本地实现过 `torch.autograd.Function`，并显式管理 backward dispatcher。最终确认
`cann_ops_nn` 已为公开前向 op 提供 Autograd 后，仓库改为直接调用公开 op。

这减少了以下重复逻辑：

- forward context 保存；
- routed-score 梯度处理；
- `y_origin` 重算；
- backward 参数映射；
- 本地桥与算子包版本不一致的风险。

### 8.5 删除空 Tensor fallback

早期为了 EP 空 token 场景加入过分解式 fallback。最终实现遵守公开算子合同，不再在
converter 内为一部分输入切换实现。空输入由算子校验并报错。

### 8.6 删除独立验证脚本

最终提交移除了 standalone SwigluGroup validation script。验证集中在仓库的单元测试和
正式训练/性能流程，避免脚本与真实 converter 接线逐渐漂移。

## 9. 代码改动范围

### 9.1 主要实现文件

#### `torchtitan_npu/converters/kernels/swiglu_group.py`

- A5 设备检查；
- `cann_ops_nn.ops` 延迟导入；
- `swiglu_group` / `swiglu_group_backward` dispatcher 校验；
- 路由/共享专家 activation wrapper；
- `npu_gmm` 和共享专家前置转换校验；
- `npu_swiglu_group` 注册。

#### `torchtitan_npu/converters/kernels/gmm.py`

- 路由专家 activation callable 注入；
- `set_expert_activation()` 清理 compile cache key；
- 两次 GMM 中间统一调用所选 activation；
- activation-only compile 支持 native/fused callable。

#### `torchtitan_npu/models/common/moe.py`

- 通用 `NpuSharedExperts`；
- native shared-expert activation；
- 原地转换和 activation setter；
- 保留共享专家 `w1/w2/w3` 与 FQN。

#### `torchtitan_npu/converters/kernels/moe_dispatch.py`

- 在 MoE dispatch 转换时同步安装 `NpuSharedExperts`；
- 保证没有 A5 融合时共享专家仍有完整 native 路径。

### 9.2 测试文件

`tests/unit_tests/converters/test_swiglu_group.py` 当前覆盖：

- 公开 op 的参数、optional 默认值和 contiguous/dtype 转换；
- Autograd 委托给已注册的公开 op，而不调用本地 backward 桥；
- 空 Tensor 校验交给 CANN 算子；
- 前向/反向 dispatcher 缺失时的错误；
- 路由和共享专家 activation 切换；
- 共享专家对象、参数、属性、状态和 hooks 保持；
- A3/未知设备拒绝启用；
- `npu_gmm`、`npu_moe_dispatch` 的顺序约束；
- 算子加载失败时不做部分注入。

`tests/unit_tests/converters/test_moe_dispatch.py` 覆盖
`npu_moe_dispatch` 安装 native `NpuSharedExperts`。

`tests/unit_tests/converters/test_registry.py` 断言：

- `npu_swiglu_group` 已注册；
- `state_dict_updater is None`；
- 旧名称 `npu_gmm_swiglu` 未注册。

`tests/unit_tests/models/test_deepseek_v4_gmm_compile.py` 覆盖 activation 位于两次 GMM
之间，以及 compile/动态 token/selective AC 的接线。

> 本次只根据当前分支代码更新文档，没有重新执行上述测试。这里列出的是当前测试代码的
> 覆盖范围，不代表本次文档更新产生了新的实机验证结果。

## 10. 验收清单

### 10.1 配置与结构

- A5 配置顺序为 `npu_moe_dispatch -> npu_gmm -> npu_swiglu_group`；
- A3 配置不包含 `npu_swiglu_group`；
- 路由专家是 `NpuGroupedExperts`；
- 共享专家是 `NpuSharedExperts`；
- dense FFN 未被误转换；
- shared `w1/w2/w3` 和 state dict key 保持不变；
- `npu_swiglu_group` 没有 state-dict updater。

### 10.2 Framework/dispatcher

A5 非空前向应出现：

```text
cann_ops_nn::swiglu_group
```

训练反向应由算子注册的 Autograd 出现：

```text
cann_ops_nn::swiglu_group_backward
```

如果仅看到 Python wrapper 名称，还需要展开图或 profiling timeline，确认最终落到了真实
dispatcher/kernel。

### 10.3 数值与梯度

至少需要对齐：

- routed/shared forward；
- routed input、`w13`、`w2` 和 routed score 梯度；
- shared input、`w1/w2/w3` 梯度；
- 有 clamp 和无 clamp；
- eager 与 compile；
- 完整训练的 loss/grad_norm finite 状态和基线一致性。

fake dispatcher 单元测试只能验证 Python 接线，不能替代 A5 真机 kernel 数值验证。

### 10.4 性能

融合前路由专家中间区域为：

```text
clamp + cat/view + npu_swiglu + routed-score cast/mul
```

融合后为：

```text
SwigluGroup
```

性能验收应同时看 kernel 数量、融合区间耗时、稳态 step time、TPS/MFU 和 device active
time。共享专家仍保留 `w1/w3` 后的 `cat`，它不在本次融合范围内。

## 11. 旧文档内容与当前实现对照

| 旧描述 | 当前实现 |
| --- | --- |
| converter 为 `npu_gmm_swiglu` | converter 为 `npu_swiglu_group` |
| 文件为 `gmm_swiglu.py` | 文件为 `swiglu_group.py` |
| 只依赖 `npu_gmm` | 标准配置先执行 `npu_moe_dispatch` 和 `npu_gmm`；有共享专家时两者都是前置条件 |
| 融合 converter 自己转换共享专家 | `npu_moe_dispatch` 负责转换为 `NpuSharedExperts` |
| 本地 `_NpuSwigluGroup` 连接前反向 | 直接依赖 `cann_ops_nn` 为公开 op 注册的 Autograd |
| 反向名为 `swiglu_group_quant_backward` | 当前预检查 `swiglu_group_backward` |
| 空 Tensor 使用 Python fallback | 当前不支持空 Tensor，直接交由算子校验 |
| 存在 standalone 验证脚本 | 最终提交已删除 |

## 12. 关键提交

| Commit | 作用 |
| --- | --- |
| `3c89cd3` | 扩展到共享专家融合 |
| `2fbadeb` | 将 SwigluGroup 从 GMM 结构转换中解耦 |
| `12693c8` | 恢复 DeepSeek-V4 模型基线，避免把可选算子写入模型定义 |
| `e3627d5` | 按 review 继续收敛 converter 结构 |
| `ca0167c` | 补齐训练反向算子接线阶段的实现 |
| `bbda2aa` | 最终将 converter 命名收敛为 `npu_swiglu_group` |
| `3ea4795` | 删除空 Tensor fallback |
| `184c192` | 将共享专家通用适配从 SwigluGroup converter 中解耦 |
| `5a6b658` | 删除 standalone 验证脚本，形成 PR 最终代码 |

## 参考资料

- [CANN ops-nn：SwigluGroup](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)
- [CANN ops-nn：SwigluGroupBackward](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_backward/swiglu_group_backward.md)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)
- 仓库文档：`docs/feature_guides/npu_fused_ops.md`

> 说明：外部 `master` 链接可能随上游演进。长期归档时，应替换为与目标 CANN 环境匹配的
> tag 或 commit 链接。
