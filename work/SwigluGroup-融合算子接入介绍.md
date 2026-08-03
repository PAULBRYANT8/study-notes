# SwigluGroup 融合算子接入介绍

> 对应 PR：[cann/torchtitan-npu#468](https://gitcode.com/cann/torchtitan-npu/pull/468)<br>
> PR 最终提交：`5a6b658`<br>
> 本地 `master` 合并提交：`cd9821e`<br>
> 最后更新：2026-08-03

## 一句话概括

当前实现保留通用的 `npu_moe_dispatch` 和 `npu_gmm` 路径，在 A5 上显式追加
`npu_swiglu_group`，把路由专家两次 GMM 之间以及共享专家中的 clamp、SwiGLU 和可选
routed-score 乘法切换为 CANN `SwigluGroup`；A3 不配置该 converter，继续使用原生激活。

## 1. 优化了哪一段

### 路由专家

MoE 路由专家的 FFN 可以简化为：

```text
routed tokens
    -> GMM-1（w13，输出 [gate | up]）
    -> clamp + SwiGLU + routed-score multiply
    -> GMM-2（w2）
```

`SwigluGroup` 只替换两次 GMM 中间的激活区间：

```text
融合前：GMM-1 -> clamp -> npu_swiglu -> score multiply -> GMM-2
融合后：GMM-1 -> SwigluGroup                           -> GMM-2
```

### 共享专家

共享专家保留普通 `w1/w2/w3` 结构，只替换线性层之间的 activation：

```text
w1(x) ---- gate --\
                  +-> cat([gate, up]) -> SwigluGroup -> w2
w3(x) ------ up --/
```

接入没有改变专家权重、token dispatch、TP 布局或 checkpoint key。路由专家仍由
`npu_gmm` 负责 GMM 和 `w13`；共享专家仍保留独立的 `w1/w2/w3`。

## 2. 如何配置

### A3/通用原生路径

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
```

### A5 SwigluGroup 融合路径

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_swiglu_group"),
```

三个 converter 的顺序有明确含义：

1. `npu_moe_dispatch` 建立 NPU MoE dispatch，并把共享专家适配为可切换 activation 的
   `NpuSharedExperts`；
2. `npu_gmm` 把路由专家转换为 `NpuGroupedExperts`；
3. `npu_swiglu_group` 校验 A5 和 CANN 算子后，只替换前两步预留的 activation callable。

| 平台与配置 | 路由专家激活 | 共享专家激活 | 行为 |
| --- | --- | --- | --- |
| A3/A2 + 前两个 converter | 原生 clamp + `npu_swiglu` + score multiply | 原生 `silu(gate) * up` | 正常使用 |
| A5 + 前两个 converter | 原生路径 | 原生路径 | 可用于兼容或对比 |
| A5 + 三个 converter | `SwigluGroup` | `SwigluGroup` | 启用融合 |
| 非 A5 + `npu_swiglu_group` | 不执行 | 不执行 | converter 阶段直接报错 |

`npu_swiglu_group` 不是 `npu_gmm` 的替代项，而是依赖 `npu_gmm` 的 activation-only
扩展。当前仓库不再注册旧名称 `npu_gmm_swiglu`。

## 3. 三个 converter 如何协作

### 3.1 `npu_moe_dispatch`

除了 MoE token dispatch，它还会把 `shared_experts` 原地转换为
`NpuSharedExperts`，并安装默认的 native activation：

```text
FeedForward shared_experts
  -> NpuSharedExperts
  -> native_shared_expert_activation
```

原地转换保留模块 identity、`w1/w2/w3` 参数、FQN、hooks 和 training/eval 状态。

### 3.2 `npu_gmm`

`npu_gmm` 继续负责完整的路由专家结构转换：

- `GroupedExperts -> NpuGroupedExperts`；
- `w1/w3 -> w13`；
- GMM-1 和 GMM-2；
- TP 与 state-dict updater；
- activation-only compile。

两次 GMM 中间统一调用当前模块保存的 activation：

```python
h = torch._grouped_mm(x, w13, offs=offsets)
h = activation_fn(h, swiglu_limit, routed_scores)
out = torch._grouped_mm(h, w2, offs=offsets)
```

默认 activation 仍是原生 `_expert_activation`。

### 3.3 `npu_swiglu_group`

`NpuSwigluGroupConverter` 的职责很窄：

```text
检查 A5
  -> 检查 NpuGroupedExperts 已存在
  -> 检查 shared_experts 已由 npu_moe_dispatch 适配
  -> 导入 cann_ops_nn.ops
  -> 检查 swiglu_group / swiglu_group_backward
  -> 切换 routed/shared activation callable
```

它不重新转换 `GroupedExperts`，不创建参数，也不注册 state-dict updater。

## 4. 前向参数如何对应

当前直接调用公开算子：

```python
torch.ops.cann_ops_nn.swiglu_group.default(
    h.contiguous(),
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

| 参数 | 当前传值 |
| --- | --- |
| `x` | `[gate | up]`，进入算子前转为 contiguous |
| `weight` | 路由专家传连续 FP32 `routed_scores`；共享专家传 `None` |
| `group_index` | `None` |
| `clamp_limit` | 模型的 `swiglu_limit`；无 clamp 时传 `-1.0` |

`group_index=None` 不表示丢失专家分组。token 在第一次 GMM 前已经完成路由、重排和
offsets 计算，中间 activation 只需逐行处理已有 routed rows。

## 5. 训练反向如何工作

当前实现不再在仓库中维护 `_NpuSwigluGroup(torch.autograd.Function)`，也不手动调用
backward dispatcher。

Python 代码直接调用：

```text
torch.ops.cann_ops_nn.swiglu_group.default
```

该公开 op 的 Autograd 集成由 `cann_ops_nn` 包提供。converter 启用时会提前确认训练所需
dispatcher 均存在：

```text
cann_ops_nn::swiglu_group
cann_ops_nn::swiglu_group_backward
```

因此 `torchtitan-npu` 只负责选择正确的公开 op，不再重复实现 backward context、
`y_origin` 重算或梯度参数映射。

## 6. 共享专家为什么也能复用同一算子

`NpuSharedExperts.forward()` 先保留原来的三个 Linear：

```python
packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
hidden = activation_fn(packed, swiglu_limit)
return self.w2(hidden)
```

区别只在 `activation_fn`：

```text
未启用融合：native_shared_expert_activation
启用 A5 融合：swiglu_group_shared_activation
```

共享专家没有 routed score，所以最终调用 `SwigluGroup` 时 `weight=None`。

把共享专家基础适配放在 `npu_moe_dispatch`，而不是放在 A5-only converter 中，有三个
好处：

- A3 和 A5 使用同一种模块结构；
- 硬件差异只体现在 activation callable；
- 融合 converter 不承担通用模型结构转换。

## 7. A3 为什么不会受影响

A3 配置不包含 `npu_swiglu_group`，所以：

- 不执行 A5 设备检查；
- 不导入 `cann_ops_nn.ops`；
- 不检查或调用 `SwigluGroup`；
- 路由专家继续使用 `npu_gmm` 的原生 activation；
- 共享专家继续使用 `native_shared_expert_activation`。

如果 A3 误配 `npu_swiglu_group`，转换阶段会明确报错：

```text
npu_swiglu_group requires an A5 device.
```

不会静默回退，也不会修改已经安装的 native activation。

## 8. 前置条件和依赖缺失时的行为

### 未先运行 `npu_gmm`

```text
npu_swiglu_group requires npu_gmm to run first.
```

### 有共享专家，但未先运行 `npu_moe_dispatch`

```text
npu_swiglu_group requires npu_moe_dispatch to run first for shared experts.
```

### CANN 算子包不完整

converter 会导入 `cann_ops_nn.ops` 并检查：

```text
swiglu_group
swiglu_group_backward
```

任一入口缺失都会在注入 fused activation 前报错，因此不会产生“部分路由专家已切换、
部分共享专家未切换”的半转换状态。

## 9. 空 Tensor 边界

当前 `swiglu_group_activation()` 不包含 `h.numel() == 0` 的 Python fallback，所有输入都
直接传给公开算子。

与原 `torch_npu.npu_swiglu` 路径一致，当前不支持空 Tensor。测试验证空输入校验由
CANN 算子负责，而不是由 converter 自动切换到分解式实现。

这意味着当前能力不包括：

- EP 空 token rank 的 Python 小算子回退；
- 空输入的 eager/compile/backward 等价保证；
- 空输入时绕开 CANN op。

## 10. Compile 与 activation checkpointing

`npu_gmm` 的 `compile_expert_activation()` 会编译当前选中的 activation callable。
`NpuGroupedExperts.set_expert_activation()` 在切换为融合实现时会清除旧 compile key，保证
后续针对 `SwigluGroup` 重新建图。

动态 token 场景继续由原有 wrapper 标记 `h` 和可选 `routed_scores` 的第 0 维。

selective activation checkpointing 仍只判断是否启用 `npu_gmm`。这是正确的，因为
grouped MM 由 `npu_gmm` 引入，`npu_swiglu_group` 只改变中间 activation，并不代表另一套
GMM 实现。

## 11. 代码范围

主要实现文件：

- `torchtitan_npu/converters/kernels/swiglu_group.py`：A5-only converter、算子参数映射和 activation 选择；
- `torchtitan_npu/converters/kernels/gmm.py`：通用 GMM、路由专家 activation 注入点和 activation compile；
- `torchtitan_npu/models/common/moe.py`：`NpuSharedExperts` 与 native shared activation；
- `torchtitan_npu/converters/kernels/moe_dispatch.py`：安装通用共享专家适配；
- `docs/feature_guides/npu_fused_ops.md`：仓内用户说明。

主要测试文件：

- `tests/unit_tests/converters/test_swiglu_group.py`；
- `tests/unit_tests/converters/test_moe_dispatch.py`；
- `tests/unit_tests/converters/test_registry.py`；
- `tests/unit_tests/models/test_deepseek_v4_gmm_compile.py`。

本次没有：

- 修改 DeepSeek-V4 模型 forward；
- 为共享专家合并 `w1/w3` 参数；
- 改变 GMM state-dict 映射；
- 让 `npu_gmm` 根据硬件自动切换；
- 为 `npu_swiglu_group` 注册 state-dict updater；
- 在本仓实现本地 Autograd bridge；
- 为空 Tensor 提供 Python fallback。

## 12. 验证重点

当前单元测试代码覆盖：

- `npu_moe_dispatch` 安装 native `NpuSharedExperts`；
- `npu_gmm` 保留 native routed activation；
- A5 的 `npu_swiglu_group` 切换 routed/shared activation；
- A3/未知设备拒绝融合且不加载 A5 op；
- `npu_gmm` 和 `npu_moe_dispatch` 的顺序保护；
- 算子缺失时不做部分注入；
- routed score 转为连续 FP32；
- `group_index=None`；
- 无 clamp 使用 `-1.0`；
- Autograd 委托给公开 op；
- 共享专家对象、参数、状态和 hooks 保持；
- 空 Tensor 校验交给公开算子；
- registry 只注册 `npu_swiglu_group`，旧名称不存在；
- activation 位于两次 GMM 之间，compile/selective AC 接线不变。

目标 A5 环境还应验证：

- routed/shared 前向输出；
- routed input、`w13`、`w2`、routed score 梯度；
- shared input、`w1/w2/w3` 梯度；
- eager 与 compile；
- loss、grad norm 和 NaN/Inf；
- step time、TPS/MFU、显存和 profiler kernel。

> 本次只根据当前分支更新文档，没有重新运行测试或 A5 实机训练。

## 13. 常见问题

### 需要改训练配置吗？

如果希望在 A5 启用融合，需要在已有 `npu_moe_dispatch` 和 `npu_gmm` 后追加：

```python
get_model_converter_config("npu_swiglu_group")
```

A3 不追加。

### `npu_gmm` 和 `npu_swiglu_group` 可以同时配置吗？

不仅可以，而且 `npu_swiglu_group` 要求 `npu_gmm` 先运行。两者不是互斥 converter：

- `npu_gmm` 负责路由专家结构和 GMM；
- `npu_swiglu_group` 负责选择融合 activation。

### 为什么还需要 `npu_moe_dispatch`？

共享专家的通用 `NpuSharedExperts` 结构由 `npu_moe_dispatch` 安装。存在共享专家时，如果
没有先运行它，融合 converter 会报错。

### 为什么不继续使用 `npu_gmm_swiglu`？

这个名称会让人误解为另一套完整 GMM converter。最终职责已经收敛为 activation-only，
所以名称改为 `npu_swiglu_group`，旧名称也不再注册。

### 算子名有 Group，为什么不传 `group_index`？

专家分组已由 token dispatch、重排和 GMM offsets 完成；当前 activation 不需要再次分组。

### routed score 为什么转成 FP32？

公开算子的 `weight` 参数要求 FP32，因此调用前显式转换并整理为连续内存。

### 反向为什么没有 Python bridge？

因为 `cann_ops_nn` 已为公开前向 op 提供 Autograd 集成。本仓只预检查
`swiglu_group_backward` 存在，不重复维护本地 backward 实现。

### 空 token 会自动回退吗？

不会。当前实现不支持空 Tensor，也没有 Python fallback；空输入由公开算子校验并报错。

## 14. 可以直接这样介绍

> `SwigluGroup` 以独立的 activation-only converter 接入。通用的
> `npu_moe_dispatch` 负责 MoE dispatch 和共享专家适配，`npu_gmm` 负责路由专家的两次
> grouped matmul；A5 再显式追加 `npu_swiglu_group`，把路由专家和共享专家的中间激活
> 切到 CANN `SwigluGroup`。
>
> A3 不配置这个 A5-only converter，继续走原生 activation。融合接入不修改模型 forward、
> GMM 权重、共享专家 `w1/w2/w3` 或 checkpoint。前向直接调用公开
> `cann_ops_nn::swiglu_group`，反向使用算子包已经注册的
> `cann_ops_nn::swiglu_group_backward` Autograd 路径。

## 参考资料

- [CANN `SwigluGroup` 文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)
- [CANN `SwigluGroupBackward` 文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_backward/swiglu_group_backward.md)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)
- 仓内文档：`docs/feature_guides/npu_fused_ops.md`
- 详细复盘：`E:\study_notes\work\swiglu-group-接入复盘.md`
