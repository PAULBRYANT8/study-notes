# `SwigluGroup` 融合算子接入复盘

> 本文以 `torchtitan-npu` 当前实现为准。
>
> - 对外 Converter 名：`npu_swiglu_group`
> - 实现文件：`torchtitan_npu/converters/kernels/swiglu_group.py`
> - 关联 PR：[`cann/torchtitan-npu#468`](https://gitcode.com/cann/torchtitan-npu/pull/468)
> - 最后更新：2026-08-06

## 1. 当前实现概览

`npu_swiglu_group` 将专家 FFN 中的 clamp、SwiGLU 和可选的 routed-score 缩放交给
`cann_ops_nn.swiglu_group` 执行。Converter 根据当前 NPU 环境中的真实
forward/backward 探测结果选择融合实现或原生小算子实现。

当前支持三种专家计算路径：

| 专家类型 | 第一层投影 | 激活 | 第二层投影 |
| --- | --- | --- | --- |
| 原始 `GroupedExperts` | 普通 matmul | `swiglu_group` 或原生小算子 | 普通 matmul |
| 启用 upstream grouped-mm 的 `GroupedExperts` | `torch._grouped_mm` | `swiglu_group` 或原生小算子 | `torch._grouped_mm` |
| `NpuGroupedExperts` | NPU GMM | `swiglu_group` 或原有 activation | NPU GMM |

名为 `shared_experts` 的 `FeedForward` 也可以由该 Converter 直接适配，通过
`NpuSharedExperts` 在 `w1/w3` 与 `w2` 之间调用同一个 activation。

## 2. 融合区间

专家 FFN 的核心计算为：

```text
gate   = w1(x)
up     = w3(x)
hidden = silu(gate) * up
output = w2(hidden)
```

gate 和 up 沿最后一维打包为 `[gate | up]` 后，`SwigluGroup` 完成：

1. 可选的 gate/up clamp；
2. `silu(gate) * up`；
3. 可选的 routed-score 逐 token 缩放。

因此完整关系是：

```text
gate/up projection
  -> cat([gate, up])
  -> SwigluGroup(clamp + SwiGLU + optional routed score)
  -> output projection
```

矩阵乘可以是普通 matmul、upstream grouped-mm 或 NPU GMM，activation 的输入输出约定
保持一致。

## 3. 运行时能力探测

### 3.1 探测内容

`probe_swiglu_group_ops()` 首先导入 `cann_ops_nn.ops`，随后在当前 NPU 上实际执行两条
小规模训练路径：

- routed 路径：BF16 输入、float32 routed score、clamp、forward 和 backward；
- shared 路径：BF16 输入、无 routed score、forward 和 backward。

探测输入规模为 `[2, 128]`，并在 forward 和 backward 后分别执行
`torch.npu.synchronize()`，确保异步错误能够在 Converter 阶段暴露。

`torch.autograd.grad()` 同时检查 routed 输入、shared 输入和 routed score 的梯度。
PyTorch 报出的未注册 Autograd kernel warning 会被转为异常，避免把不可训练的调用误判为
可用。

### 3.2 探测失败后的行为

导入、forward、backward 或同步阶段的任意异常都会被捕获。Converter 输出 warning，并按
模型当前结构继续使用小算子路径：

- 已有 `NpuGroupedExperts` 保留当前 activation；
- 已有 `NpuSharedExperts` 保留当前 activation；
- 原始 `GroupedExperts` 转换为 `SwigluGroupExperts`，使用原生小算子 activation；
- 原始 shared `FeedForward` 保持原实现。

原始路由专家仍转换为 adapter，是为了让它能够接收
`(x, num_tokens_per_expert, routed_scores)`，兼容 `npu_moe_dispatch` 的专家调用约定。

模型中没有兼容的路由专家或 shared expert 时，Converter 直接返回，不执行 NPU 探测。

## 4. 算子调用约定

统一 activation 入口为：

```python
def swiglu_group_activation(h, swiglu_limit=None, routed_scores=None):
    weight = (
        None
        if routed_scores is None
        else routed_scores.to(dtype=torch.float32).contiguous()
    )
    clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
    return torch.ops.cann_ops_nn.swiglu_group.default(
        h.contiguous(),
        weight=weight,
        group_index=None,
        clamp_limit=clamp_limit,
    )
```

参数映射如下：

| 参数 | 当前传法 |
| --- | --- |
| `x` | 打包后的 `[gate | up]`，转为 contiguous |
| `weight` | routed score 转 float32、contiguous；shared expert 传 `None` |
| `group_index` | `None` |
| `clamp_limit` | 使用 `swiglu_limit`；未配置时传 `-1.0` |

token 的专家分组已由 dispatch 排序和 `num_tokens_per_expert`/offsets 表达，因此 activation
按现有行逐行计算，不再传入额外 group index。

Autograd 由 `cann_ops_nn` 对公开前向算子注册的实现负责，训练代码直接对
`torch.ops.cann_ops_nn.swiglu_group.default` 求导。

## 5. 原始路由专家适配

### 5.1 `SwigluGroupExperts`

对于精确类型为 `GroupedExperts` 的模块，Converter 使用原地 `__class__` 转换为
`SwigluGroupExperts`，保留原参数对象、模块 FQN、state-dict key、hooks 和训练状态。

adapter 增加可注入的 `_expert_activation_fn`，并把 forward 签名扩展为：

```python
forward(x, num_tokens_per_expert, routed_scores=None)
```

DTensor 权重沿用 upstream 专家实现的处理方式，在本地计算前通过 `to_local()` 取得
`w1/w2/w3`。

### 5.2 普通 matmul 路径

`use_grouped_mm=False` 时，根据 `num_tokens_per_expert` 拆分 token：

```text
每个专家执行 x @ w1.T 和 x @ w3.T
  -> 合并所有专家的 [gate | up]
  -> 调用所选 activation
  -> 按专家拆分 hidden
  -> 每个专家执行 hidden @ w2.T
```

routed score 与打包后的 token 顺序一致，直接传给 activation。

### 5.3 upstream grouped-mm 路径

`use_grouped_mm=True` 时，通过 `num_tokens_per_expert` 的累积和构造 int32 offsets：

```text
torch._grouped_mm(x, w1)
torch._grouped_mm(x, w3)
  -> cat([gate, up])
  -> 调用所选 activation
  -> torch._grouped_mm(hidden, w2)
```

输出最终通过 `.type_as(x)` 恢复输入 dtype。

## 6. NPU GMM 路径

模型已经包含 `NpuGroupedExperts` 时，探测成功后直接调用
`set_expert_activation(swiglu_group_activation)`，两次 GMM 的参数布局和 state-dict 处理继续
由 `npu_gmm` 负责。

两个 Converter 的先后顺序都能保留融合 activation：

- `npu_gmm -> npu_swiglu_group`：通过 `set_expert_activation()` 更新 callable；
- `npu_swiglu_group -> npu_gmm`：GMM Converter 从原始专家读取
  `_expert_activation_fn`，再传给 `NpuGroupedExperts`。

`NpuGroupedExperts.set_expert_activation()` 会清理 activation compile key，后续编译使用新的
callable 建图。

## 7. Shared expert 路径

Converter 扫描 leaf name 为 `shared_experts` 且类型为 `FeedForward` 的模块。探测成功后统一
调用：

```python
NpuSharedExperts.convert(
    shared_module,
    activation_fn=swiglu_group_activation,
)
```

`NpuSharedExperts` 的 forward 为：

```text
w1(x) ---- gate --\
                  +-> cat([gate, up]) -> activation -> w2
w3(x) ------ up --/
```

shared expert 调用 activation 时只传 `packed` 和 `swiglu_limit`，因此
`routed_scores=None`、算子 `weight=None`。

`npu_swiglu_group` 先执行时，后续 `npu_moe_dispatch` 会识别已经转换的
`NpuSharedExperts` 并保留其 activation；`npu_moe_dispatch` 先执行时，
`npu_swiglu_group` 会在现有 `NpuSharedExperts` 上更新 activation。

## 8. 配置方式

单独启用：

```python
get_model_converter_config("npu_swiglu_group"),
```

与 MoE dispatch、GMM 组合：

```python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_swiglu_group"),
```

组合配置中，Converter 的先后顺序不会丢失已经选择的 activation。

## 9. 参数、状态和编译

`npu_swiglu_group` 本身不创建专家权重，也没有 state-dict updater：

- raw routed expert 保留 `w1/w2/w3`；
- shared expert 保留 `w1/w2/w3`；
- `w13` 的创建与拆分仍属于 `npu_gmm`；
- activation-only compile 使用当前 `_expert_activation_fn`；
- dynamic-token compile 继续由 GMM 编译路径标记 activation 输入的第 0 维。

## 10. 当前测试覆盖

`tests/unit_tests/converters/test_swiglu_group.py` 覆盖：

- 公开算子的参数映射、dtype、contiguous 和默认值；
- 公开算子的 Autograd 委托；
- weighted/unweighted forward/backward 真实探测接线；
- 探测失败 warning 和小算子回退；
- 无兼容目标时跳过探测；
- GMM 后选择融合 activation；
- raw routed/shared expert 直接转换；
- 普通 matmul 和 upstream grouped-mm 两条 raw expert 路径；
- `swiglu -> gmm` 和 `swiglu -> moe_dispatch` 的 activation 保留；
- shared expert 参数梯度；
- 未启用融合时 GMM/shared expert 保持原生 activation；
- 无 GMM 且探测失败时 raw adapter 的 routed-score 小算子路径。

本地已执行：

- `python -m py_compile`：通过；
- `git diff --check`：通过。

当前本地 Python 环境缺少 `torch`，目标 pytest 未能执行。真实 NPU 环境仍需补充目标单测和
最小训练冒烟验证。

## 11. 验收关注点

### 11.1 功能与梯度

- routed/shared forward 数值；
- routed input、`w1/w3/w2` 或 `w13/w2`、routed score 梯度；
- shared input 和 `w1/w2/w3` 梯度；
- 有 clamp 与无 clamp；
- 普通 matmul、upstream grouped-mm、NPU GMM；
- eager 与 compile；
- 探测失败后的 warning 和训练连续性。

### 11.2 性能

融合前 activation 区间为：

```text
clamp + SwiGLU + routed-score cast/mul
```

融合后为：

```text
SwigluGroup
```

性能验证应同时观察 kernel 数量、融合区间耗时、稳态 step time、TPS/MFU 和 device active
time。shared expert 的 `w1/w3` 输出仍需在 Python 侧执行 `cat` 后再进入融合算子。

## 参考资料

- [CANN ops-nn：SwigluGroup](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)
- [CANN ops-nn：SwigluGroupBackward](https://gitcode.com/cann/ops-nn/blob/master/torch_extension/cann_ops_nn/ops/activation/swiglu_group_backward/swiglu_group_backward.md)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)
- 仓库文档：`docs/feature_guides/npu_fused_ops.md`
