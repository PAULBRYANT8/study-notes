# DeepSeek-V4 接入 `SwigluGroup` 开发复盘

> 本文基于 `torchtitan-npu` 的 `feat/swiglu-group-a5-fusion` 分支和
> [PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468) 整理。
>
> 当前对外配置名为 `npu_gmm_swiglu`，实现文件为
> `torchtitan_npu/converters/kernels/gmm_swiglu.py`。它是一个 A5-only 的激活替换
> converter，必须配置在 `npu_gmm` 之后。
>
> 最后更新：2026-07-30。

## 1. 背景与目标

### 1.1 SwiGLU 是什么

SwiGLU（Swish-Gated Linear Unit）是 Transformer FFN 中常用的门控激活。以
DeepSeek-V4 的前馈网络为例，输入 `x` 先经过两条升维线性层：

```text
gate = w1(x)
up   = w3(x)
```

然后计算：

```text
hidden = silu(gate) * up
output = w2(hidden)
```

如果把 `gate` 和 `up` 沿最后一维拼接：

```text
packed = [gate | up]
```

SwiGLU 会把最后一维均分为 `A` 和 `B`，计算：

```text
y = silu(A) * B
```

因此本次接入必须保证 `w1(x)` 在前、`w3(x)` 在后。顺序颠倒会改变
模型数学语义。

### 1.2 `SwigluGroup` 比普通 SwiGLU 多做什么

CANN `SwigluGroup` 除了执行 `silu(A) * B`，还能把 clamp 和可选 token 权重
乘法融合进同一个算子：

```text
x = [A | B]

if clamp_limit > 0:
    A = min(A, clamp_limit)
    B = min(max(B, -clamp_limit), clamp_limit)

y_origin = silu(A) * B
y = y_origin if weight is None else y_origin * weight
```

当前接入中的参数含义如下：

| 参数 | 作用 | 当前用法 |
| --- | --- | --- |
| `x` | 最后一维为 `[A \| B]`，输出最后一维减半 | 路由专家的 `w13` 输出，或共享专家 `w1/w3` 输出的拼接结果 |
| `weight` | 对每个 token 的激活结果乘权 | 路由专家传 `routed_scores.float()`；共享专家传 `None` |
| `group_index` | count 模式下的分组 token 数 | 当前两条路径都传 `None` |
| `clamp_limit` | 激活前截断阈值 | 有 `swiglu_limit` 时传正数；无 clamp 时传 `-1.0` |

官方 `SwigluGroup` 文档声明该算子支持 Ascend 950PR 和 Ascend 950DT，不支持
A3/A2。文档还要求 `weight` 为 FLOAT32，并要求输入最后一维可被 2 整除。

参见：[ops-nn `SwigluGroup` 文档](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group/README.md)。

### 1.3 路由专家与共享专家

DeepSeek-V4 MoE 包含两条专家路径。

#### 路由专家

Router 为每个 token 选择若干专家，并产生 routed score。token 按专家重排后进入
GMM 计算：

```text
token
  ├─ router 选择专家并产生 routed score
  ├─ dispatch / token reorder
  ├─ GMM-1：w13
  ├─ clamp + SwiGLU + routed-score scaling
  ├─ GMM-2：w2
  └─ combine
```

路由专家需要 `npu_gmm` 负责 token 分组后的两次 grouped matmul、`w13` 参数
布局和 checkpoint 映射。

#### 共享专家

共享专家不由 router 选择，所有 token 经过同一套 FFN：

```text
token
  └─ shared_experts
      ├─ w1
      ├─ w3
      ├─ clamp + SwiGLU
      └─ w2
```

共享专家本质上是普通 `FeedForward`，没有 routed score，也不需要 GMM 分组。它
可以复用同一个 `SwigluGroup` 激活入口，但继续保留独立的 `w1/w2/w3`。

### 1.4 接入前的计算路径

`npu_gmm` 已经把路由专家的 `w1/w3` 和 `w2` 替换为两次 grouped matmul。融合前，
两次 GMM 中间是分解激活：

```text
GMM-1：x @ w13
  → chunk gate/up
  → clamp gate/up（可选）
  → cat
  → torch_npu.npu_swiglu
  → routed_scores cast + mul（可选）
  → GMM-2：hidden @ w2
```

共享专家融合前的激活路径为：

```text
w1(x) → gate clamp → silu ─┐
w3(x) → up clamp        ─├→ mul → w2
                            ┘
```

这些 elementwise 操作会产生额外 kernel launch 和中间 Tensor 读写。

### 1.5 最终接入目标

最终实现的目标是：

1. 在 A5 上用 `SwigluGroup` 替换路由专家两次 GMM 之间的 clamp、SwiGLU 和
   routed-score 乘法。
2. 用同一激活入口替换共享专家的 clamp 和 SwiGLU，保留 `w1/w2/w3` 参数
   与 state dict key。
3. 使用局部 `torch.autograd.Function` 把 CANN 前向与
   `swiglu_group_quant_backward` 反向 dispatcher 连接起来。
4. 保留 activation-only compile 和 selective activation checkpointing 设计。
5. 对 EP 下合法的空 token 输入使用等价分解路径，避免 CANN 算子接收空
   Tensor。
6. 把 A5-only 能力与通用 `npu_gmm` 解耦，使 A3 继续只运行原生分解激活。

最终配置必须按以下顺序出现：

```python
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_gmm_swiglu"),
```

这是单向依赖：

- `npu_gmm` 可以单独使用；
- `npu_gmm_swiglu` 不能单独使用；
- 公共 `_DEFAULT_CONVERTERS` 不应无条件包含 `npu_gmm_swiglu`，否则 A3 配置会
  在 converter 阶段报错。

本次的非目标包括：

- 不让 `npu_gmm_swiglu` 再次执行 GMM 结构转换；
- 不给 `npu_gmm_swiglu` 注册独立 state-dict updater；
- 不把共享专家 `w1/w3` 合并为 `w13`；
- 不批量替换非专家 dense FFN；
- 不支持非 A5 平台强制运行 `SwigluGroup`；
- 不承诺二阶梯度。

## 2. 硬件与软件边界

### 2.1 A5 能力边界

`torchtitan_npu.tools.device` 当前的设备能力映射为：

```python
_NPU_DEVICE_TYPE_MAP = {
    "Ascend950DT": "A5",
    "Ascend950PR": "A5",
    "Ascend910_95": "A5",
    "Ascend950": "A5",
    "Ascend910_93": "A3",
    "Ascend910B": "A2",
}
```

`NpuSwigluGroupConverter.convert()` 在修改模型前先执行：

```python
if get_npu_device_type() != "A5":
    raise ValueError("npu_gmm_swiglu requires an A5 device.")
```

因此：

- `npu_gmm` 不会加载 `cann_ops_nn`，A3 可继续使用；
- `npu_gmm_swiglu` 只在显式选择且设备是 A5 时生效；
- 非 A5 误选时显式报错，不静默回退。

### 2.2 以 dispatcher 是否存在判断能力

当前代码不依赖 Python distribution metadata 判断 CANN 能力，而是在 converter
应用时执行：

```python
importlib.import_module("cann_ops_nn.ops")

for op_name in ("swiglu_group", "swiglu_group_quant_backward"):
    _ = getattr(torch.ops.cann_ops_nn, op_name).default
```

当前训练路径实际使用的 dispatcher 是：

```text
torch.ops.cann_ops_nn.swiglu_group.default
torch.ops.cann_ops_nn.swiglu_group_quant_backward.default
```

前向或反向入口任一缺失，`ensure_swiglu_group_ops()` 都会在 converter 阶段抛出包含
算子名的 `RuntimeError`。这比等到首个 backward 再失败更容易定位环境问题。

### 2.3 为什么保留局部 Autograd 桥

目标 CANN 包分别提供前向和反向 dispatcher，但前向 op 没有自动关联到当前代码
使用的反向入口。直接调用 `swiglu_group` 不会因为存在一个名字相似的反向 op
就自动建立梯度边。

因此在 converter 局部定义 `_NpuSwigluGroup(torch.autograd.Function)`：

```text
_NpuSwigluGroup.apply
  ├─ forward  → cann_ops_nn::swiglu_group
  └─ backward → cann_ops_nn::swiglu_group_quant_backward
```

这不是在 Python 中手写 SwiGLU 梯度公式。Python 层只负责保存 context、准备
`y_origin` 并调用 CANN 反向算子。相比 `torch.library.register_autograd`，局部
`autograd.Function` 不会修改外部算子的全局 Autograd 注册。

### 2.4 空 Tensor 是独立的边界条件

EP、小 micro-batch 或路由不均匀时，某个 rank 可能合法地收不到 token。此时专家
激活输入可能是 shape 为 `[0, 2H]` 的空 Tensor。

`SwigluGroup` 不能直接处理该输入，所以 `swiglu_group_activation()` 在调用 CANN 前检查：

```python
if h.numel() == 0:
    gate, up = h.chunk(2, dim=-1)
    if swiglu_limit is not None:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    output = torch.nn.functional.silu(gate) * up
    if routed_scores is not None:
        output = output * routed_scores.to(output.dtype)
    return output
```

这是对空输入的等价分解实现，不是对普通非空输入的静默回退。它保留
eager、`torch.compile` 和 backward 的空梯度语义，同时确保 CANN op 不会收到空 Tensor。

## 3. 最终架构与取舍

### 3.1 Converter 责任划分

最终不再让一个 A5-only converter 同时承担 GMM 与 SwiGLU 激活替换。两个
converter 的职责如下：

| Converter | 职责 | 硬件边界 | State-dict updater |
| --- | --- | --- | --- |
| `npu_gmm` | 把 `GroupedExperts` 转为 `NpuGroupedExperts`，建立 `w13`，执行两次 GMM，默认使用原生分解激活 | 通用 NPU 路径 | `GMMStateDictUpdater` |
| `npu_gmm_swiglu` | 将已转换专家的激活切换为 `SwigluGroup`，并原地转换共享专家 | 仅 A5 | 无 |

虽然对外名称仍是 `npu_gmm_swiglu`，但“GMM”表示它替换的是 GMM 专家中间的
SwiGLU 区间，不表示它会再次执行 GMM 转换。

### 3.2 转换顺序为什么是强约束

`NpuSwigluGroupConverter.convert()` 先查找模型中的 `gmm.NpuGroupedExperts`：

```python
experts = [
    module
    for module in model.modules()
    if isinstance(module, gmm.NpuGroupedExperts)
]
if not experts:
    raise ValueError("npu_gmm_swiglu requires npu_gmm to run first.")
```

这个检查发生在 `ensure_swiglu_group_ops()` 之前。因此顺序错误时：

1. 先报出清晰的配置错误；
2. 不会提前导入 A5 CANN 包；
3. 不会修改路由专家或共享专家。

正确过程为：

```text
npu_gmm
  └─ GroupedExperts → NpuGroupedExperts
       └─ 默认 activation = _expert_activation

npu_gmm_swiglu
  ├─ A5 校验
  ├─ 确认 NpuGroupedExperts 已存在
  ├─ 校验 CANN forward/backward dispatcher
  ├─ set_expert_activation(swiglu_group_activation)
  └─ *.shared_experts → NpuSharedExperts（原地）
```

### 3.3 路由专家的公开交接点

`gmm.py` 为独立激活 converter 提供了：

```python
def set_expert_activation(self, activation_fn):
    self._expert_activation_fn = activation_fn
    self._expert_activation_compile_key = None
```

这个方法做两件事：

- 替换两次 GMM 之间的 activation callable；
- 清空旧 compile key，避免之前缓存的 native activation 图被错误复用。

`npu_gmm_swiglu` 不重新包装 `NpuGroupedExperts`，也不改变其参数布局。

### 3.4 共享专家为什么保留 `w1/w2/w3`

共享专家采用“保留线性层，只替换激活”的方案：

```text
w1(x) ─┐
       ├─ cat → SwigluGroup → w2
w3(x) ─┘
```

没有把 `w1/w3` 合并为 `w13`，原因是：

1. checkpoint 和 HF state dict 使用现有 key；
2. TP plan 依赖 `moe.shared_experts.w1/w2/w3` FQN；
3. MXFP8 等 converter 可能按原 FQN 查找 Linear；
4. 本次目标是激活融合，不应扩大为共享专家参数重排。

为了保留模块 identity、参数、buffer、hook 和 training/eval 状态，转换通过
`parent.__class__ = NpuSharedExperts` 原地完成。

### 3.5 为什么不修改 `moe.py`

最终 PR 已将 `torchtitan_npu/models/deepseek_v4/moe.py` 的所有差异恢复到基线。

这个取舍符合插件仓原则：

- 算子替换应通过 converter 完成；
- 不把 A5/CANN 依赖写进模型 forward；
- 不为了一个可选融合路径改变默认模型语义。

## 4. 路由专家接入

### 4.1 两次 GMM 之间的融合点

`_run_experts_grouped_mm()` 接收统一签名的 `activation_fn`：

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

`npu_gmm` 单独使用时，`activation_fn` 是 `_expert_activation`。

`npu_gmm_swiglu` 执行后，`activation_fn` 被切换为
`swiglu_group_activation`。

### 4.2 非空输入的 A5 融合路径

`swiglu_group_activation()` 对非空输入的核心处理是：

```python
weight = (
    None
    if routed_scores is None
    else routed_scores.to(dtype=torch.float32).contiguous()
)
clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
return _NpuSwigluGroup.apply(h.contiguous(), weight, clamp_limit)
```

这里有四个关键语义。

#### `h` 转为 contiguous

算子入口前显式整理连续内存，避免对 stride 的隐式假设与实际输入不一致。

#### routed score 转为 FLOAT32

官方约束要求 `weight` 为 FLOAT32，所以不能直接把 BF16 score 传给算子。

#### `group_index=None`

前面的 dispatch 和 GMM 已经只物化实际 routed rows，`num_tokens_per_expert` 用于生成
GMM offsets。中间激活遍历所有现有 rows 即可，不需要再传一次分组边界。

#### 无 clamp 时使用 `-1.0`

Python 侧 `swiglu_limit=None` 映射到前向算子的 `clamp_limit=-1.0`。当前反向调用
也原样传递该值：代码中没有 `-1.0 → 0.0` 的 sentinel 转换。

### 4.3 空路由专家路径

当 `h.numel() == 0` 时，不调用 `_NpuSwigluGroup.apply`，而是直接执行空 Tensor 上的
等价 clamp/SwiGLU/score 乘法。

这条路径要求：

- 输出 shape 为 `[0, H]`；
- dtype 与输入一致；
- `h.grad` 和 `routed_scores.grad` 存在且为空/零张量；
- eager 和 `torch.compile` 行为一致；
- CANN forward/backward 都不被调用。

### 4.4 `npu_gmm` 原生路径

`npu_gmm` 仍执行原来的分解激活：

```python
if swiglu_limit is not None:
    gate, up = h.chunk(2, -1)
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)
    h = torch.cat([gate, up], dim=-1)

h = torch_npu.npu_swiglu(h, dim=-1)

if routed_scores is not None:
    h = h * routed_scores.to(h.dtype)
```

这条路径无论设备类型都不会自动切到 CANN `SwigluGroup`。

### 4.5 预期融合收益

融合前：

```text
chunk/views + clamp(gate) + clamp(up) + cat + npu_swiglu + score cast/mul
```

融合后的非空路径：

```text
SwigluGroup(clamp + silu + gate multiply + routed-score multiply)
```

预期收益来自减少 elementwise kernel launch 和中间 Tensor 写回。但端到端结果还受
GMM、通信、dispatch/combine、activation checkpointing 和 token shape 影响，不能仅根据
算子数量承诺固定加速比。

## 5. 共享专家接入

### 5.1 只匹配名为 `shared_experts` 的 `FeedForward`

共享专家不能通过 `isinstance(module, FeedForward)` 无差别替换，否则 dense transformer
block 的 FFN 也会被改变。当前匹配条件为：

```python
for name, module in list(model.named_modules()):
    if (
        isinstance(module, FeedForward)
        and name.rsplit(".", 1)[-1] == "shared_experts"
    ):
        NpuSharedExperts.convert(module)
```

它只要求路径最后一段精确等于 `shared_experts`，比写死完整 DeepSeek-V4 FQN 更
通用，又不会修改其他普通 FFN。

### 5.2 原地类转换保留了什么

```python
class NpuSharedExperts(FeedForward):
    @classmethod
    def convert(cls, parent):
        parent.__class__ = cls
        return cast("NpuSharedExperts", parent)
```

对象本身没有被新建或替换，因此保留：

- `w1/w2/w3` 子模块与参数对象；
- buffer；
- forward/backward hook；
- training/eval 状态；
- module identity；
- state dict key。

若 Python 不允许该 `__class__` 赋值，代码会转换为明确的 `RuntimeError`。

### 5.3 共享专家前向

```python
def forward(self, x):
    packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
    hidden = swiglu_group_activation(
        packed,
        swiglu_limit=getattr(self, "swiglu_limit", None),
    )
    return self.w2(hidden)
```

关键语义：

| 项目 | 取值 | 原因 |
| --- | --- | --- |
| packed 顺序 | `[w1(x) \| w3(x)]` | 对齐 `silu(w1(x)) * w3(x)` |
| `weight` | `None` | 共享专家没有 routed score |
| `group_index` | `None` | 所有现有行都参与计算 |
| `clamp_limit` | `swiglu_limit` 或 `-1.0` | 保持模型原有 clamp/无 clamp 语义 |

### 5.4 空共享专家

当共享专家输入 shape 为 `[0, D]` 时：

1. `w1/w3` 生成空输出；
2. `cat` 产生 `[0, 2H]`；
3. `swiglu_group_activation()` 走空 Tensor 分解路径；
4. `w2` 产生 `[0, D]`；
5. `x` 与 `w1/w2/w3` 的梯度均按 PyTorch 空张量语义保留。

单元测试同时覆盖 eager 和 `torch.compile`，并断言 CANN op 不会收到空输入。

### 5.5 共享专家的性能取舍

融合前：

```text
w1 + w3 + clamp(gate) + clamp(up) + silu + mul + w2
```

融合后：

```text
w1 + w3 + cat + SwigluGroup + w2
```

融合减少 clamp/silu/mul 的独立调度，但为了构造 `[gate | up]` 新增了 `cat`。
净性能收益需要使用真实 shape 的 timeline 和稳态 step time 验证。

## 6. 反向与 `torch.compile`

### 6.1 Autograd 前向保存什么

`_NpuSwigluGroup.forward()` 的 Python 输入只有：

```text
x, weight, clamp_limit
```

`group_index` 不是 Autograd Function 的输入，当前实现在前反向 CANN 调用中始终传
`None`。

前向保存：

```python
ctx.save_for_backward(x, weight)
ctx.clamp_limit = clamp_limit
```

然后调用：

```python
torch.ops.cann_ops_nn.swiglu_group.default(
    x,
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

### 6.2 路由专家为什么要重算 `y_origin`

路由专家前向是：

```text
y_origin = swiglu(clamped x)
y = y_origin * weight
```

计算 `grad_weight` 需要未加权的 `y_origin`。不能使用 `y / weight` 恢复，因为 routed
score 可能为零，会导致除零或数值放大。

当 `weight is not None` 时，反向先重算一次不带 weight 的前向：

```python
y_origin = torch.ops.cann_ops_nn.swiglu_group.default(
    x,
    weight=None,
    group_index=None,
    clamp_limit=ctx.clamp_limit,
)
```

再调用：

```python
grad_x, grad_weight = (
    torch.ops.cann_ops_nn.swiglu_group_quant_backward.default(
        grad_output.contiguous(),
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=None,
        clamp_limit=ctx.clamp_limit,
    )
)
```

所以路由专家 backward 附近通常会看到：

```text
1 × SwigluGroup                       # 重算 y_origin
1 × SwigluGroupQuantBackward/Grad    # 计算 grad_x / grad_weight
```

具体硬件 kernel 名可能与 Python dispatcher 名不完全一致，应结合调用位置判断。

### 6.3 共享专家为什么不需要重算

共享专家传 `weight=None`，不存在 routed-score 梯度，因此：

```text
y_origin = None
```

可直接调用 CANN 反向算子。预期只有一个融合反向 kernel，没有为
`grad_weight` 进行的额外前向重算。

### 6.4 当前 no-clamp 参数语义

当前代码在无 clamp 时使用：

```text
forward clamp_limit = -1.0
backward clamp_limit = -1.0
```

单元测试明确覆盖 `-1.0` 和正数 `0.5` 两种情况，断言反向收到的值与前向
一致。文档不应再描述已不存在的 `-1.0 → 0.0` 转换。

### 6.5 梯度返回与二阶梯度边界

Autograd Function 按三个前向输入的顺序返回：

```python
return (
    grad_x if ctx.needs_input_grad[0] else None,
    grad_weight if weight is not None and ctx.needs_input_grad[1] else None,
    None,  # clamp_limit
)
```

`backward` 使用 `@torch.autograd.function.once_differentiable`，当前只支持一阶梯度。

### 6.6 activation-only compile

`gmm.py` 保留只编译两次 GMM 之间激活桥的设计：

```python
compiled_activation = torch.compile(
    activation_fn,
    backend=backend,
    fullgraph=True,
    options={"custom_partitioner_fn": _NpuGmmAotDefaultPartitioner()},
)
```

`compile_expert_activation()` 查找所有 `NpuGroupedExperts`，使用：

```python
compile_key = (backend, dynamic_tokens)
```

判断当前激活是否已编译。`npu_gmm_swiglu` 切换激活时会把模块的 compile key
清空，所以后续 parallelize 阶段会为新激活重新建图。

EP 场景 token 数可动态变化。当 `dynamic_tokens=True` 时，编译 wrapper 会标记 `h` 和可选
`routed_scores` 的第 0 维为动态。

### 6.7 selective activation checkpointing 与 converter 判断

DeepSeek-V4 `parallelize.py` 只通过是否存在 `npu_gmm` 判断 GMM 路径：

```python
npu_gmm_enabled = has_npu_converter(
    model_converters.converters,
    "npu_gmm",
)
```

这是正确的，因为融合配置必须同时包含 `npu_gmm`，而 `npu_gmm_swiglu` 本身不代表
GMM 已转换。selective AC 继续保存 grouped-mm 输出，避免 backward 重算 GMM。

## 7. 接入过程中的问题与修正

### 7.1 A5 融合不应默认进入 `npu_gmm`

早期设计将 A5 分支放在通用 GMM converter 中，会使 A3 的默认 `npu_gmm` 也受
`cann_ops_nn` 和 A5 算子能力影响。

最终修正为两个可组合 converter：

```text
A3/通用：npu_gmm
A5 融合：npu_gmm + npu_gmm_swiglu
```

这个设计保护了已验证的 A3 路径，也使 A5-only 依赖只在显式配置时加载。

### 7.2 “有前向和反向算子”不等于“PyTorch 会自动求导”

前向和反向 dispatcher 是两个独立入口。若包本身没有为前向注册 Autograd 公式，
PyTorch 不会仅根据命名自动调用反向算子。

局部 `autograd.Function` 的作用是建立这条边，而不是在 Python 中重复 CANN 的
数学梯度实现。

### 7.3 routed score 为零时不能通过除法恢复 `y_origin`

使用 `y / weight` 会在 score 为零时产生错误。当前使用不带 weight 的 CANN 前向重算
`y_origin`，并用单元测试显式覆盖零 score。

### 7.4 空 token rank 是正常 EP 边界，不是非法输入

某 rank 在当前 micro-batch 收不到 token 是合法情况。如果无条件调用 CANN op，会因
空 Tensor 合约失败。

修正不是整个 converter 回退，而是只在 `h.numel() == 0` 时执行数学等价的 PyTorch
分解实现。

### 7.5 共享专家前向通不代表反向已接入

共享专家 `weight=None`，因此不需要 `grad_weight`，但仍需通过 CANN 反向算子生成
packed gate/up 的输入梯度，再继续传给 `w1/w3`。

单元测试显式确认：

- CANN backward 被调用；
- `x.grad` 存在；
- `w1/w2/w3.weight.grad` 都存在。

### 7.6 测试代码也需要遵守仓库 CodeCheck

接入期间曾处理过两类流水线问题：

- 测试直接访问受保护成员，触发 G.CLS.11；
- 测试代码块重复和 `type(x) is T` 写法，触发重复代码与 G.TYP.08。

最终测试使用公开 converter/激活入口、共享 helper 和 `isinstance()`，避免为测试屏蔽
真实规则告警。

### 7.7 名称与职责的最终决策

实现文件最终使用 `gmm_swiglu.py`，对外配置名使用 `npu_gmm_swiglu`。
这个命名强调它替换的是 GMM 专家中间激活。

但实现职责仍然是 activation-only：

- 必须在 `npu_gmm` 之后；
- 不再转换 `GroupedExperts`；
- 不持有 `GMMStateDictUpdater`；
- 只切换路由专家激活和共享专家 forward。

## 8. 如何判断接入成功

### 8.1 Converter 和模型结构

首先确认配置顺序：

```python
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_gmm_swiglu"),
```

并确认：

- 设备类型为 A5；
- routed `GroupedExperts` 已由 `npu_gmm` 转为 `NpuGroupedExperts`；
- `NpuGroupedExperts._expert_activation_fn` 已切换为 `swiglu_group_activation`；
- `*.shared_experts` 已原地转为 `NpuSharedExperts`；
- dense `feed_forward` 仍是原模块；
- state dict key 仍保持不变。

这只能证明 converter 已安装，不能单独证明硬件 kernel 已执行。

### 8.2 Framework/dispatcher 层

非空前向应看到：

```text
cann_ops_nn::swiglu_group
```

训练反向应看到当前包暴露的 quant-backward/grad 路径：

```text
cann_ops_nn::swiglu_group_quant_backward
```

若只看到 Python 范围 `swiglu_group_activation` 或 `_NpuSwigluGroup`，仍需继续展开确认其
内部存在真实 `torch.ops` 节点。

### 8.3 Ascend Hardware timeline

#### 路由专家前向

```text
GMM-1
  → SwigluGroup
  → GMM-2
```

同一区间不应再出现旧的 clamp/cat/npu_swiglu/routed-score mul 组合。

#### 共享专家前向

```text
w1 matmul / w3 matmul
  → cat
  → SwigluGroup
  → w2 matmul
```

#### 路由专家反向

```text
SwigluGroup                    # y_origin 重算
SwigluGroupQuantBackward/Grad  # grad_x / grad_weight
```

#### 共享专家反向

```text
SwigluGroupQuantBackward/Grad
```

activation checkpointing 也可能在 backward 阶段重放整段 forward。判断额外前向来源时需要
结合调用栈、shape 和前后算子，不能把所有 backward 期间的 `SwigluGroup` 都当作
`y_origin` 局部重算。

### 8.4 数值与梯度

至少需要对齐：

- routed/shared forward 输出；
- `x.grad`；
- routed `w13.grad` 和 `w2.grad`；
- `routed_scores.grad`；
- shared `w1/w2/w3` 梯度；
- 空 token 输入的 shape/dtype/gradient；
- 有 clamp 和无 clamp 场景；
- eager 与 compile；
- 完整训练的 loss/grad_norm 与 finite 状态。

fake dispatcher 单元测试只验证 Python 接线，不能代替 A5 真实 kernel 数值对齐。

### 8.5 性能

性能验收建议同时看：

1. 融合区间 kernel 数量；
2. 融合区间累计耗时；
3. 稳态单步时间；
4. TPS/MFU；
5. device active time；
6. 共享专家 `cat` 与路由 backward 重算是否引入新瓶颈。

第一次执行可能包含 Python 包导入、dispatcher 注册、图编译和 cache 建立，不应用作
稳态性能结论。

### 8.6 当前测试命令

目标 converter、Autograd、空输入与 compile 选择测试：

```bash
pytest -q \
  tests/unit_tests/converters/test_swiglu_group.py \
  tests/unit_tests/converters/test_registry.py \
  tests/unit_tests/models/test_deepseek_v4_gmm_compile.py
```

默认 GMM A3 冒烟测试：

```bash
pytest -q tests/smoke_tests/features/test_gmm.py
```

扩大回归范围：

```bash
pytest -q tests/unit_tests/converters -x
pytest -q tests/unit_tests/models/test_deepseek_v4*.py -x
```

### 8.7 截至本文的验证状态

- 最终命名和模块路径调整后，相关 converter/registry/compile 测试：`34 passed`。
- 架构解耦后扩大到全部 converter 和 DeepSeek-V4 单元测试：`95 passed`。
- A3 环境 GMM 冒烟测试：`2 passed, 1 skipped`；跳过项因环境缺少 `npuc`。
- 空 routed/shared 输入已覆盖 eager、compile 和 backward，并断言 CANN op 不被调用。
- 共享专家 backward 已覆盖 `w1/w2/w3` 梯度。
- 分支早期融合路径已在 A5 服务器验证可运行；最终 converter 解耦与命名调整后
  本轮记录没有重新执行 A5 实机训练。合入前应以最终 HEAD 再跑一次 A5
  forward/backward/compile 和稳态训练验证。
- 本地 Pyrefly 因系统 Python 缺少 PyTorch 无法形成有效结果；远端测试环境未安装
  Pyrefly。

## 9. 最终调用链对照

### 9.1 `npu_gmm` 原生路由专家

```text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
      └─ _run_experts_grouped_mm
          ├─ offsets = cumsum(num_tokens_per_expert)
          ├─ torch._grouped_mm(x, w13)
          ├─ _expert_activation
          │    ├─ clamp gate/up（可选）
          │    ├─ torch_npu.npu_swiglu
          │    └─ mul routed_scores（可选）
          └─ torch._grouped_mm(hidden, w2)
```

### 9.2 `npu_gmm + npu_gmm_swiglu` A5 路由专家

```text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
      └─ _run_experts_grouped_mm
          ├─ offsets = cumsum(num_tokens_per_expert)
          ├─ torch._grouped_mm(x, w13)
          ├─ swiglu_group_activation
          │    ├─ h.numel() == 0
          │    │    └─ 等价 PyTorch 分解路径
          │    └─ h.numel() > 0
          │         └─ _NpuSwigluGroup.apply
          │              └─ cann_ops_nn::swiglu_group
          └─ torch._grouped_mm(hidden, w2)
```

### 9.3 路由专家反向

```text
_NpuSwigluGroup.backward
  ├─ weight is not None
  │    └─ swiglu_group(weight=None) 重算 y_origin
  └─ swiglu_group_quant_backward
       ├─ grad_x
       └─ grad_weight → routed_scores.grad
```

### 9.4 A5 共享专家

```text
NpuSharedExperts.forward
  ├─ gate = w1(x)
  ├─ up = w3(x)
  ├─ packed = cat(gate, up)
  ├─ swiglu_group_activation
  │    ├─ 空输入 → 等价分解路径
  │    └─ 非空输入 → _NpuSwigluGroup.apply(weight=None)
  └─ output = w2(hidden)
```

### 9.5 共享专家反向

```text
_NpuSwigluGroup.backward
  ├─ weight is None
  ├─ y_origin = None，不重算前向
  └─ swiglu_group_quant_backward
       └─ grad_x → w1/w3 路径
```

## 10. 代码改动范围

### 10.1 主要实现文件

#### `torchtitan_npu/converters/kernels/gmm.py`

保留通用 GMM 能力：

- 两次 grouped MM、TP 处理和 `_expert_activation`；
- `NpuGroupedExperts`、`GMMStateDictUpdater` 与 `npu_gmm` 注册；
- `set_expert_activation()` 公开交接点；
- activation-only compile 框架；
- 默认 `npu_gmm` 永远选择 native activation。

#### `torchtitan_npu/converters/kernels/gmm_swiglu.py`

承担 A5-only 激活能力：

- `cann_ops_nn.ops` 延迟导入和前反向 dispatcher 校验；
- `_NpuSwigluGroup` 局部 Autograd Function；
- `swiglu_group_activation()` 与空 Tensor 回退；
- `NpuSharedExperts` 原地转换；
- A5 校验和 `npu_gmm` 前置顺序校验；
- `npu_gmm_swiglu` 注册，不注册 state-dict updater。

#### `torchtitan_npu/models/deepseek_v4/parallelize.py`

- 只以 `npu_gmm` 判断 GMM 是否已启用；
- selective AC 继续保存 grouped-mm 输出；
- model compile 时调用 `compile_expert_activation()`。

`torchtitan_npu/models/deepseek_v4/moe.py` 的 PR 改动已全部回退，不在最终改动范围内。

### 10.2 测试文件

`tests/unit_tests/converters/test_swiglu_group.py` 覆盖：

- CANN 前向输入格式和 optional 参数；
- routed score 转 FLOAT32/contiguous；
- no-clamp `-1.0`；
- routed/shared 空 Tensor 的 eager/compile/backward；
- Autograd forward/backward 连接；
- 零 routed score 下 `y_origin` 重算；
- routed-score 梯度返回；
- 无 score 时不重算 `y_origin`；
- dispatcher 缺失时的错误信息；
- 共享专家原地转换与 `w1/w2/w3` 梯度；
- `npu_gmm` 继续使用 native activation；
- 非 A5 拒绝融合；
- `npu_gmm_swiglu` 必须在 `npu_gmm` 之后。

`tests/unit_tests/converters/test_registry.py` 还断言：

- `npu_gmm_swiglu` 已注册；
- 它的 `state_dict_updater is None`；
- 过渡期使用过的旧名不再注册。

`tests/unit_tests/models/test_deepseek_v4_gmm_compile.py` 覆盖：

- activation 确实在两次 GMM 之间执行；
- native/fused activation 都可进入 compile wrapper；
- compile 共享、幂等和动态 token 标记；
- DeepSeek-V4 parallelize 只查询 `npu_gmm` 判断 GMM/AC/compile 路径。

### 10.3 关键提交

| Commit | 内容 |
| --- | --- |
| `bd64a61` | 首次接入 A5 `SwigluGroup` 专家融合 |
| `3c89cd3` | 将共享专家接入融合前向与反向路径 |
| `7944c82` / `8f4656b` | 处理测试 CodeCheck、空输入和其他 review 意见 |
| `2fbadeb` | 把 `SwigluGroup` 从 GMM 结构转换中解耦为独立 converter |
| `437fb0d` | 实现文件名恢复为 `gmm_swiglu.py` |
| `12693c8` | 恢复 `deepseek_v4/moe.py` 基线，移除该文件的 PR 差异 |
| `676e2ae` | 对外 converter 名统一为 `npu_gmm_swiglu` |
| `1eae07e` | 更新融合算子用户文档格式 |

## 附录 A：关键概念

### A.1 GMM 与 `SwigluGroup` 的关系

GMM（Grouped Matrix Multiplication）把多组 shape 兼容的矩阵乘法放进一次 grouped op。
MoE 中不同专家执行相同结构但使用不同权重，非常适合 GMM：

```text
expert 0: tokens_0 @ weight_0
expert 1: tokens_1 @ weight_1
expert 2: tokens_2 @ weight_2
```

`SwigluGroup` 不是 GMM。它位于两次 GMM 之间，负责逐行激活和可选 score 乘法：

```text
GMM-1 → SwigluGroup → GMM-2
```

### A.2 `w1`、`w3` 与 `w13`

普通 SwiGLU FFN 使用两个独立升维权重：

```text
w1 → gate
w3 → up
```

路由专家为了让一次 GMM 同时产生 gate/up，把两个权重融合为 `w13`：

```text
w13 = concat(w1, w3)
x @ w13 → [gate | up]
```

共享专家保留独立 `w1/w3`，在输出侧执行 `cat`。两种布局数学上可等价，但
参数对象、state dict、TP plan 和其他 converter 的观察结果不同。

### A.3 `weight` 为什么是 routed score

MoE combine 需要把每个选中专家的输出乘以 router 产生的 score。原路径在 SwiGLU 后
执行：

```python
h = h * routed_scores.to(h.dtype)
```

`SwigluGroup` 的可选 `weight` 正好表达这一逐 token 乘法。这里的 weight 不是线性层
权重，也不是 `w1/w2/w3`。

### A.4 为什么 `group_index=None`

当前路由流程在进入激活前已经：

1. 根据 router 结果重排 token；
2. 根据 `num_tokens_per_expert` 构造 GMM offsets；
3. 只生成实际参与专家计算的 rows。

因此 `SwigluGroup` 处理所有现有输入行即可。`group_index=None` 不会取消 GMM 的专家
分组，因为分组已在前一个算子完成。

### A.5 `y_origin` 是什么

```text
y_origin = silu(A) * B
y = y_origin * weight
```

`y_origin` 是未乘 routed score 的 SwiGLU 激活，主要用于计算 `grad_weight`。它不是未
clamp 的输入，也不是整个 FFN 的原始输出。

### A.6 前向能运行与反向能运行是两件事

设备 dispatcher 存在只证明可以执行对应 kernel。训练还需要 Autograd 知道：

- forward 要保存哪些 Tensor；
- backward 调用哪个 dispatcher；
- 每个 forward 输入对应返回哪个梯度；
- 哪些输入不可微。

这正是局部 `_NpuSwigluGroup` 的职责。

### A.7 Python op 名、dispatcher 名和硬件 kernel 名

同一次计算可能在不同层级显示不同名称：

```text
Python wrapper:  torch.ops.cann_ops_nn.swiglu_group.default
Dispatcher:      cann_ops_nn::swiglu_group
Hardware kernel: SwigluGroup / 包内实现名
```

反向也可能在 Python 层显示 `swiglu_group_quant_backward`，在硬件层显示
`SwigluGroupQuantGrad` 或包内 kernel 名。Profiling 判断应结合调用位置、shape 和前后算子。

### A.8 为什么空 Tensor 要保留可微分解路径

直接返回 `torch.empty` 虽然 shape 正确，但可能丢失与输入和参数的 Autograd 关系。
当前代码仍在空 Tensor 上执行 chunk/clamp/silu/mul，使结果保留完整计算图，只是
因元素个数为零而没有实际数值计算。

### A.9 activation checkpointing 对 Profiling 的影响

activation checkpointing 会在 backward 中重新执行一段 forward。因此 backward 期间出现
`SwigluGroup` 可能有两个来源：

1. checkpoint 的整段 forward recompute；
2. `_NpuSwigluGroup.backward` 为 routed-score 梯度进行的局部 `y_origin` recompute。

可通过是否连同整个 block 重算、调用栈、shape 和是否紧邻 quant backward 来区分。

## 参考资料

- [ops-nn：`SwigluGroup`](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group/README.md)
- [ops-nn：`SwigluGroupQuantGrad`](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group_quant_grad/README.md)
- [ops-nn 算子列表](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/op_list.md)
- [PyTorch `torch.autograd.Function`](https://docs.pytorch.org/docs/stable/autograd.html#function)
- [torchtitan-npu PR #468](https://gitcode.com/cann/torchtitan-npu/pull/468)

> 说明：上述 `ops-nn` 链接指向可变的 `master`。长期归档时应替换为与目标 CANN
> 9.2.0 环境匹配的 tag 或 commit 链接。
