# DeepSeek-V4 接入 `SwigluGroup` 开发复盘

> 对应 PR：[cann/torchtitan-npu#468](https://gitcode.com/cann/torchtitan-npu/pull/468)  
> 当前 PR 头提交：`d33a1e1`（前置整合提交：`bd64a61`）  
> 最后更新：2026-07-29

## 1. 最终结论

这次接入最终收敛为一个很小的改动：

- 配置名仍然是既有的 `npu_gmm`，用户不需要增加或替换 converter；
- 只替换路由专家两次 GMM 之间的激活区间；
- A5 自动使用 CANN `SwigluGroup`；
- A3、A2 等非 A5 平台继续使用原来的 `torch_npu.npu_swiglu` 分解路径；
- 不转换共享专家，不新增 `NpuSharedExperts`；
- 不新增 `npu_gmm_swiglu` converter；
- 不在模型代码中加入硬件判断；
- 只保留一个薄的算子适配与 Autograd 桥。

最终执行路径如下：

```text
所有平台均配置 npu_gmm
          │
          ├─ A5
          │    GMM-1
          │      → cann_ops_nn.swiglu_group
          │      → GMM-2
          │
          └─ A3 / A2 / 其他非 A5
               GMM-1
                 → clamp（模型配置了 swiglu_limit 时）
                 → torch_npu.npu_swiglu
                 → routed-score multiply
                 → GMM-2
```

这个结构同时满足两个目标：A5 能使用新融合算子，A3 的默认 `npu_gmm` 路径不受影响。

## 2. 接入范围为什么只应是一小段

路由专家的 `npu_gmm` 已经负责：

1. 把专家 `w1`、`w3` 合并为 `w13`；
2. 执行第一次 grouped matmul；
3. 执行中间激活；
4. 执行第二次 grouped matmul；
5. 维护既有 TP、编译和 state-dict 逻辑。

`SwigluGroup` 能替换的只是第 3 步。

融合前：

```text
h = GMM-1(x, w13)
gate, up = h.chunk(2, dim=-1)
gate = clamp_max(gate, swiglu_limit)
up = clamp(up, -swiglu_limit, swiglu_limit)
h = torch_npu.npu_swiglu(cat(gate, up))
h = h * routed_scores
out = GMM-2(h, w2)
```

A5 融合后：

```text
h = GMM-1(x, w13)
h = cann_ops_nn.swiglu_group(
    h,
    weight=routed_scores.float(),
    group_index=None,
    clamp_limit=swiglu_limit,
)
out = GMM-2(h, w2)
```

因此本次没有理由重做 GMM converter、共享专家模块、参数布局或 checkpoint 映射。把改动限制
在 activation callback 边界，能够最大程度复用已有实现。

## 3. 平台兼容策略

### 3.1 为什么不能无条件替换

`SwigluGroup` 当前只在 A5 产品上支持。若默认 `npu_gmm` 无条件加载或调用该算子，A3 会在
模型转换或执行时失败。

最终实现把硬件选择放在 `NpuGroupedExpertConverter.convert()`：

```python
activation_fn = _expert_activation
if get_npu_device_type() == "A5":
    from torchtitan_npu.converters.kernels.gmm_swiglu import (
        ensure_swiglu_group_ops,
        swiglu_group_activation,
    )

    ensure_swiglu_group_ops()
    activation_fn = swiglu_group_activation
```

随后同一个 `NpuGroupedExperts` 接收选定的 `activation_fn`。

### 3.2 兼容矩阵

| 平台 | 配置 | 中间激活 | 是否加载 `cann_ops_nn` |
| --- | --- | --- | --- |
| A5 | `npu_gmm` | `swiglu_group` | 是 |
| A3 | `npu_gmm` | 原生 `_expert_activation` | 否 |
| A2 | `npu_gmm` | 原生 `_expert_activation` | 否 |
| A5 但算子包不完整 | `npu_gmm` | 转换阶段给出明确错误 | 尝试加载并校验 |

非 A5 分支不会导入 `gmm_swiglu.py`，也不会调用 `ensure_swiglu_group_ops()`。这保证 A3 不需要
安装 A5 专属算子包，也不会因算子不存在而受影响。

### 3.3 为什么保留同一个 converter

独立的 `npu_gmm_swiglu` converter 会引入新的配置选择，并要求用户按硬件维护两套配置。
这与本次“只替换 `npu_gmm` 的一小段实现”的实际边界不匹配。

同一个 `npu_gmm` 内按能力选择 activation 有几个直接收益：

- 原有训练配置无需修改；
- A3 自然回退到已经验证的路径；
- GMM、TP、state dict 和 compile 逻辑只有一份；
- A5 专属代码仍被隔离在独立的小文件中；
- 不会出现两个 GMM converter 重复转换同一模块的问题。

## 4. `SwigluGroup` 接口映射

前向使用公开接口：

```python
torch.ops.cann_ops_nn.swiglu_group.default(
    x,
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

仓内参数映射为：

| 算子参数 | 当前取值 | 原因 |
| --- | --- | --- |
| `x` | 第一次 GMM 的连续输出 `h.contiguous()` | 输入布局为 `[gate | up]` |
| `weight` | `routed_scores.float().contiguous()`，没有 score 时为 `None` | 融合 routed-score 乘法，且公开接口要求 weight 为 FP32 |
| `group_index` | `None` | token 已在 GMM 前完成路由和重排 |
| `clamp_limit` | 模型的 `swiglu_limit`；没有时为 `-1.0` | 保持原 clamp/无 clamp 语义 |

`group_index=None` 不会取消专家分组。专家分组已经由 token 重排、每专家 token 数和 GMM
offsets 完成；中间激活只需处理第一次 GMM 产生的有效 rows。

官方前向文档：
[SwigluGroup](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)。

## 5. 为什么还需要一个薄 Autograd 桥

公开的 `swiglu_group` Python 扩展注册了设备前向，但没有把前向与梯度算子自动连接起来。
直接调用前向能够推理，不代表训练时 PyTorch 能自动找到反向。

当前实现使用局部 `torch.autograd.Function`：

```text
_NpuSwigluGroup.forward
  └─ cann_ops_nn::swiglu_group

_NpuSwigluGroup.backward
  └─ cann_ops_nn::swiglu_group_quant_backward
```

这里必须使用公开存在的 `swiglu_group_quant_backward`，而不是旧实现中假定存在的
`swiglu_group_backward`。它的参数与本路径需要的 `grad_output`、`x`、`weight`、
`y_origin`、`group_index` 和 `clamp_limit` 对齐。

官方反向文档：
[SwigluGroupQuantBackward](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_quant_backward/swiglu_group_quant_backward.md)。

### 5.1 routed-score 梯度

前向语义可写成：

```text
y_origin = swiglu(clamped_x)
y = y_origin * weight
```

反向计算 `grad_weight` 需要 `y_origin`。不能通过 `y / weight` 恢复，因为 routed score
可能为 0。

因此有 weight 时，桥接层会额外执行一次不带 weight 的前向：

```python
y_origin = swiglu_group(
    x,
    weight=None,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

再把 `y_origin` 传给 `swiglu_group_quant_backward`。没有 routed score 时，不需要这次重算。

### 5.2 clamp 参数保持一致

无 clamp 时，当前前向与反向都传 `-1.0`。旧方案中关于“前向 `-1.0`、反向 `0.0`”的
转换来自对另一个接口的假设，不适用于当前公开的
`swiglu_group_quant_backward`，已经删除。

### 5.3 影响范围

Autograd 注册是局部的，只覆盖 torchtitan-npu 通过 `_NpuSwigluGroup.apply()` 发起的调用，
不会对进程中其他 `cann_ops_nn::swiglu_group` 调用做全局覆盖。桥接使用
`once_differentiable`，支持训练所需的一阶反向，不承诺二阶梯度。

## 6. 最终代码结构

### 6.1 `gmm.py`

`torchtitan_npu/converters/kernels/gmm.py` 继续承载通用能力：

- `npu_gmm` converter 注册；
- `NpuGroupedExperts`；
- 两次 grouped matmul；
- A3/A2 原生 `_expert_activation`；
- A5/non-A5 activation 选择；
- activation-only `torch.compile`；
- GMM state-dict 转换。

通用执行函数通过参数接收激活：

```python
def _run_experts_grouped_mm(..., activation_fn=_expert_activation):
    ...
    h = activation_fn(h, swiglu_limit, routed_scores)
    ...
```

### 6.2 `gmm_swiglu.py`

`torchtitan_npu/converters/kernels/gmm_swiglu.py` 只保留三项职责：

1. 延迟加载并校验前向、反向 dispatcher；
2. 局部 Autograd bridge；
3. 将 `h`、`routed_scores`、`swiglu_limit` 映射到算子参数。

它不注册 converter，不遍历模型，不转换共享专家，也不维护全局 op cache。

### 6.3 编译路径

`compile_expert_activation()` 编译 converter 已选定的 activation。一次模型转换只会选择一个
实现，因此 compile key 继续使用：

```text
(backend, dynamic_tokens)
```

不需要为了已经移除的“双 converter”设计引入 `activation_key`。

## 7. 明确不在本次范围内的内容

最终版本不包含：

- 共享专家的 `w1/w3` 拼接与激活替换；
- `NpuSharedExperts`；
- 独立的 `npu_gmm_swiglu` 配置；
- 模型 forward 中的 A5 判断；
- `swiglu_group_quant` 量化前向；
- 二阶梯度；
- CANN 版本字符串或 Python package metadata 判断；
- 前后向 clamp sentinel 的人为转换；
- 基于字符串的 activation cache 分组。

尤其是共享专家并不位于 `npu_gmm` 的两次 grouped matmul 路径内。为了“复用融合算子”而
改造共享专家，会扩大模型、TP、checkpoint 和测试范围，偏离本次需求。

## 8. 测试与验证

新增/更新的测试重点覆盖：

- A5 的 `npu_gmm` 选择 `swiglu_group_activation`；
- A3 的 `npu_gmm` 保留 `_expert_activation`；
- A3 不加载 A5 算子；
- 前向参数、FP32 routed score、连续内存和 `group_index=None`；
- 无 clamp 时使用公开接口约定的 `-1.0`；
- 反向调用 `swiglu_group_quant_backward`；
- routed score 为 0 时仍通过 `y_origin` 正确计算梯度；
- 没有 routed score 时跳过额外前向；
- 算子缺失时给出明确错误；
- converter 注册仍只有 `npu_gmm`；
- activation compile 继续使用 converter 选定的函数。

本次提交前完成了：

- 变更 Python 文件的 `compileall`；
- `git diff --check`；
- 过时符号扫描，确认不存在 `npu_gmm_swiglu`、`NpuSharedExperts`、
  `swiglu_group_backward` 和 `activation_key`。

当前 Windows Python 环境缺少项目所需的 `torch`、`pytest`、`pre-commit` 依赖，因此本地
没有执行运行时测试；本次按要求也没有主动触发远端流水线。运行时数值、梯度和性能结论仍应
在目标 A5 环境验证。

## 9. 建议的 A5 验证顺序

### 9.1 接口检查

确认两个 dispatcher 都存在：

```python
torch.ops.cann_ops_nn.swiglu_group.default
torch.ops.cann_ops_nn.swiglu_group_quant_backward.default
```

### 9.2 最小数值测试

对相同输入比较：

```text
原生：clamp + npu_swiglu + score multiply
融合：swiglu_group
```

至少覆盖：

- 有/无 `swiglu_limit`；
- 有/无 routed score；
- routed score 含 0；
- BF16 激活与 FP32 score；
- 前向输出、`grad_x` 和 `grad_score`。

### 9.3 图与 profiling

确认 A5 图中两次 GMM 之间出现 `cann_ops_nn::swiglu_group`，并确认 A3 图中仍为原生路径。
性能分析时应分别观察 kernel launch、中间访存和可能由 `y_origin` 引入的反向重算，不能只
根据算子数量推导端到端加速比。

### 9.4 端到端训练

最后比较基线与融合版本的：

- loss；
- grad norm；
- 是否出现 NaN/Inf；
- step time；
- 峰值显存；
- 实际 profiler 节点。

## 10. 这次重构的主要经验

### 10.1 先确认真实接口，再设计封装

旧方案围绕不存在的 `swiglu_group_backward` 做了大量适配。查阅公开前向、反向文档和实际
dispatcher 后，正确入口是 `swiglu_group` 与 `swiglu_group_quant_backward`。

### 10.2 平台专属能力必须保留默认回退

A5-only 算子不能破坏 A3 默认配置。硬件判断应位于 converter 的能力选择边界，并确保非
A5 分支不导入、不校验、不调用 A5 依赖。

### 10.3 按真正的替换边界控制改动

本算子只替换两个 GMM 之间的一小段。最合适的抽象是 activation callback，而不是新建完整
converter、共享专家子类或另一套配置协议。

### 10.4 薄适配层比全局封装更安全

保留独立 `gmm_swiglu.py` 是有价值的，因为公开前向缺少自动 Autograd 连接；但这个文件只
需要承担算子边界职责，不应扩展成第二套模型转换框架。

## 11. 当前提交

| 提交 | 内容 |
| --- | --- |
| `bd64a61` | 将 PR #468 的原始功能整合到本地 `master` |
| `d33a1e1` | 精简为单一 `npu_gmm`、A5 条件融合、A3 原生回退和薄 Autograd 桥，并修复 CI 静态检查 |

PR #468 的远端源分支当前已更新到 `d33a1e1`。

## 参考资料

- [CANN `SwigluGroup` 前向文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)
- [CANN `SwigluGroupQuantBackward` 文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_quant_backward/swiglu_group_quant_backward.md)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)

> 上述 CANN 文档链接指向可变的 `master`。长期归档时，建议替换为与实际部署环境一致的
> tag 或 commit 链接。
