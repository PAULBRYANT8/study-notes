# SwigluGroup 融合算子接入介绍

> 对应 PR：[cann/torchtitan-npu#468](https://gitcode.com/cann/torchtitan-npu/pull/468)  
> 当前 PR 头提交：`d33a1e1`  
> 最后更新：2026-07-29

## 一句话概括

现有 `npu_gmm` 在 A5 上自动使用 CANN `SwigluGroup` 替换两次 GMM 之间的 clamp、
SwiGLU 和 routed-score 乘法；A3、A2 等平台仍走原来的 `npu_swiglu` 路径，配置无需改变。

## 1. 优化了哪一段

MoE 路由专家的 FFN 可以简化为：

```text
routed tokens
    → GMM-1（w1/w3，输出 [gate | up]）
    → clamp + SwiGLU + routed-score multiply
    → GMM-2（w2）
```

`SwigluGroup` 只替换中间激活区间：

```text
融合前：
GMM-1 → clamp → npu_swiglu → score multiply → GMM-2

A5 融合后：
GMM-1 → swiglu_group                        → GMM-2
```

它不融合前后的 GMM，也不改变专家权重、token dispatch、TP 或 checkpoint 结构。

## 2. 为什么仍然只配置 `npu_gmm`

这个融合是 `npu_gmm` 的平台实现差异，不是一套新的模型转换能力。最终没有新增
`npu_gmm_swiglu` converter。

```python
get_model_converter_config("npu_gmm")
```

同一配置在不同硬件上的行为为：

| 平台 | 激活实现 | `cann_ops_nn` 要求 |
| --- | --- | --- |
| A5 | `swiglu_group` | 需要前向和反向算子 |
| A3 / A2 / 其他非 A5 | 原生 clamp + `npu_swiglu` + score multiply | 不加载、不需要 |

这样既不要求用户维护两套配置，也不会让 A5-only 算子破坏 A3 的默认路径。

## 3. 代码如何选择实现

硬件选择发生在既有 `NpuGroupedExpertConverter` 中：

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

选定的函数被注入同一个 `NpuGroupedExperts`：

```text
npu_gmm
   │
   └─ NpuGroupedExperts
        ├─ GMM-1
        ├─ activation_fn
        │    ├─ A5：swiglu_group_activation
        │    └─ 非 A5：_expert_activation
        └─ GMM-2
```

非 A5 分支不会导入 `gmm_swiglu.py`，所以 A3 不会因缺少 A5 算子包而失败。

## 4. 前向参数如何对应

A5 调用公开算子：

```python
torch.ops.cann_ops_nn.swiglu_group.default(
    h.contiguous(),
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

参数含义如下：

| 参数 | 当前传值 |
| --- | --- |
| `x` | 第一次 GMM 输出的 `[gate | up]` |
| `weight` | 连续的 FP32 `routed_scores`；没有 score 时为 `None` |
| `group_index` | `None` |
| `clamp_limit` | 模型的 `swiglu_limit`；没有 clamp 时为 `-1.0` |

`group_index` 可以为 `None`，因为 token 在第一次 GMM 前已经完成路由、重排和专家 offsets
计算；中间激活只需逐行处理有效 routed token。

## 5. 为什么需要 Autograd bridge

公开扩展提供了前向 `swiglu_group` 和独立的梯度算子
`swiglu_group_quant_backward`，但前向没有自动关联到 PyTorch Autograd。为支持训练，
`gmm_swiglu.py` 保留了一个局部且很薄的 `torch.autograd.Function`：

```text
forward  → cann_ops_nn::swiglu_group
backward → cann_ops_nn::swiglu_group_quant_backward
```

这里没有使用旧方案假定的 `swiglu_group_backward`，因为公开接口中实际可用且参数匹配的是
`swiglu_group_quant_backward`。

当传入 routed score 时，反向计算 `grad_weight` 需要未乘权的激活 `y_origin`。由于 score
可能为 0，不能用加权输出除以 score 恢复，因此 backward 前会额外重算一次
`weight=None` 的前向。没有 routed score 时会跳过这次重算。

当前桥接支持训练所需的一阶梯度，不承诺二阶梯度。

## 6. 代码范围

主要文件只有：

- `torchtitan_npu/converters/kernels/gmm.py`：通用 GMM、平台选择和 activation 注入；
- `torchtitan_npu/converters/kernels/gmm_swiglu.py`：A5 算子参数映射与 Autograd bridge；
- `tests/unit_tests/converters/test_gmm_swiglu_group.py`：A5/A3 选择、前向和反向测试；
- `docs/feature_guides/npu_fused_ops.md`：仓内用户说明。

本次明确没有：

- 新增 `npu_gmm_swiglu` converter；
- 转换共享专家；
- 新增 `NpuSharedExperts`；
- 修改模型 forward；
- 改动 GMM 权重或 state-dict 映射；
- 使用不存在的 `swiglu_group_backward`；
- 为 native/fused 分支增加额外 `activation_key`。

## 7. A3 为什么不会受影响

A3 的执行顺序是：

1. `get_npu_device_type()` 返回非 A5；
2. `activation_fn` 保持 `_expert_activation`；
3. 不导入 A5 适配模块；
4. 不校验 `cann_ops_nn`；
5. 继续执行原来的 clamp、`torch_npu.npu_swiglu` 和 score 乘法。

因此 A3 仍然使用原来的 `npu_gmm` 配置和实现，不会调用 A5-only 算子。

## 8. A5 依赖缺失时的行为

A5 选择融合路径时会在模型转换阶段校验：

```text
cann_ops_nn::swiglu_group
cann_ops_nn::swiglu_group_quant_backward
```

任一接口缺失都会立即给出包含算子名的错误，避免训练到第一次 forward/backward 才发现
环境不完整。

## 9. 验证重点

单元测试覆盖：

- A5 选择融合 activation；
- A3 保留原生 activation 且不加载 A5 op；
- routed score 转为连续 FP32；
- `group_index=None`；
- 无 clamp 使用 `-1.0`；
- backward 调用公开的 `swiglu_group_quant_backward`；
- score 为 0 时仍通过 `y_origin` 计算梯度；
- 无 score 时不做额外重算；
- 算子缺失时报错。

目标 A5 环境还应验证：

- 融合与原生路径的前向、`grad_x`、`grad_score` 数值；
- `torch.compile` 图中实际出现 `cann_ops_nn::swiglu_group`；
- loss、grad norm 和 NaN/Inf；
- step time、峰值显存和 profiler kernel。

本次按要求没有主动触发远端流水线。

## 10. 常见问题

### 需要改训练配置吗？

不需要。A5 和 A3 都继续配置：

```python
get_model_converter_config("npu_gmm")
```

### 为什么不保留独立的 `npu_gmm_swiglu`？

因为新算子只替换现有 GMM 路径中的一小段。独立 converter 会重复 GMM 转换职责，增加配置
和测试复杂度，也不利于 A3 自动保留原路径。

### 为什么不转换共享专家？

共享专家不在这段 grouped matmul 执行路径内。改造共享专家会额外涉及普通 Linear、模块
结构、TP、checkpoint 和更大测试范围，不属于本次小范围替换。

### 算子名有 Group，为什么不传 `group_index`？

专家分组已由进入 GMM 前的 token 重排和 offsets 完成。`group_index` 是算子的可选能力，
当前调用不需要再次分组。

### 为什么 backward 名字里有 `quant`？

当前公开、已注册并且参数与前向训练梯度需求匹配的接口是
`swiglu_group_quant_backward`。接入应以实际公开 dispatcher 为准，而不是根据名字猜测一个
`swiglu_group_backward`。

### routed score 为什么必须是 FP32？

公开算子要求 `weight` 为 FP32，所以进入算子前会显式转换并整理为连续内存。

## 11. 可以直接这样介绍

> 我们没有新建一套 GMM converter，而是在原来的 `npu_gmm` 激活边界做了硬件选择。A5
> 使用 CANN `SwigluGroup`，把两个 GMM 之间的 clamp、SwiGLU 和 routed-score 乘法融合；
> A3 等平台继续走原来的 `npu_swiglu` 路径，配置完全不变。
>
> A5 适配文件只负责算子参数映射和局部 Autograd。前向调用 `swiglu_group`，反向调用公开
> 的 `swiglu_group_quant_backward`。没有改共享专家、模型结构、GMM 权重或 checkpoint，
> 所以改动范围与实际替换点保持一致。

## 参考资料

- [CANN `SwigluGroup` 前向文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)
- [CANN `SwigluGroupQuantBackward` 文档](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_quant_backward/swiglu_group_quant_backward.md)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)
- `DeepSeek-V4 接入 SwigluGroup 开发复盘`：`E:\study_notes\work\swiglu-group-接入复盘.md`
