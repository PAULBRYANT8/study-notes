---
title: 深度学习训练中的反向传播、混合精度与显存账本
type: concept
created: 2026-07-26
updated: 2026-08-04
tags: [深度学习, 数值, 反向传播, 混合精度, fp16, bf16, loss-scale, 优化器, 显存]
sources:
  - https://docs.pytorch.org/docs/stable/notes/autograd.html
  - https://docs.pytorch.org/docs/stable/autograd.html
  - https://docs.pytorch.org/docs/stable/amp.html
  - https://docs.pytorch.org/docs/stable/type_info.html
  - https://docs.pytorch.org/docs/stable/tensor_attributes.html
  - https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
---

**训练数值排障的底座是同时看懂三本账：梯度如何沿计算图反传、每个算子实际使用什么精度、参数/梯度/优化器状态/激活分别占多少内存。**

只会看 loss 曲线无法定位精度问题：loss 是许多层、许多算子和许多 step 误差汇总后的迟钝指标。适配工作要能找到“第一个不一致的中间量”“第一个非有限梯度”和“是哪一种状态把显存占满”。

---

## 一、反向传播是计算图上的链式法则

### 1. 标量链式法则

若：

$$
y=f(x),\quad L=g(y)
$$

则：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

反向传播从 loss 开始，把上游梯度乘以当前局部导数，再传给输入。

例：

$$
y=x^2,\quad L=3y
$$

则：

$$
\frac{\partial L}{\partial y}=3,\quad
\frac{\partial y}{\partial x}=2x,\quad
\frac{\partial L}{\partial x}=6x
$$

PyTorch 的 `Tensor.backward()` 按链式法则对图求导；见 [`torch.Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)。

### 2. 计算图不是 module 树

```python
y = x * w
z = y + y
loss = z.sum()
```

这是运行时由 tensor 运算组成的有向无环图。module 只是组织参数和 forward 的 Python 结构；同一个 module 可调用多次、分支可动态选择，真正反向依据的是本次 forward 产生的 autograd graph。

PyTorch 是 reverse-mode automatic differentiation 系统，forward 同时记录产生 tensor 的操作，输出 tensor 的 `grad_fn` 是进入反向图的入口。对应文档：[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

### 3. 分支处梯度相加

上例中 `y` 被使用两次：

$$
z=y+y
$$

所以：

$$
\frac{\partial L}{\partial y}
=
\left.\frac{\partial L}{\partial y}\right|_{\text{第一条边}}
+
\left.\frac{\partial L}{\partial y}\right|_{\text{第二条边}}
$$

这解释了：

- 一个参数在 forward 中多次使用，梯度会累加所有路径贡献。
- 多次调用 `backward()` 而不清空 `.grad`，叶子 tensor 的梯度继续累积。
- 梯度累积训练需要明确何时 `zero_grad`。

`backward()` 累积叶子梯度的行为见 [`torch.Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)。

### 4. 向量场景是 VJP，不是显式构造完整 Jacobian

若 $y=f(x)$ 是向量，反向接收上游向量 $v=\partial L/\partial y$，计算：

$$
v^\mathsf{T}J_f
$$

即 vector-Jacobian product（VJP）。框架通常不显式构造巨大 Jacobian，而是每个算子的 backward 实现局部 VJP。

若输出不是标量，调用：

```python
y.backward(gradient=v)
```

传入的 `gradient` 就是相对于 `y` 的上游梯度，shape 要与 `y` 兼容。对应文档：[`Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)。

### 5. 一个 Linear + 激活的反向结构

前向：

$$
z = xW^\mathsf{T} + b,\quad
h = \mathrm{ReLU}(z),\quad
L = \ell(h)
$$

反向：

$$
g_h = \frac{\partial L}{\partial h}
$$

$$
g_z = g_h \odot \mathbf{1}(z>0)
$$

$$
\frac{\partial L}{\partial x}=g_zW
$$

$$
\frac{\partial L}{\partial W}=g_z^\mathsf{T}x
$$

$$
\frac{\partial L}{\partial b}=\operatorname{sum}_{batch}(g_z)
$$

这里至少涉及：

- ReLU backward 需要知道前向哪些元素大于 0。
- Linear backward 需要前向输入或等价信息。
- bias 梯度包含归约，归约顺序会影响低位舍入。
- 权重被 batch 中所有样本共享，因此其梯度是贡献之和。

### 6. 为什么 forward 要保存 tensor

许多 backward 需要 forward 中间量。autograd node 会按需保存 tensor，反向时取回。保存什么决定了：

- backward 能否计算。
- 激活显存占用。
- 原地修改是否安全。
- activation checkpointing 能省多少内存。

PyTorch 保存 tensor 的机制见 [Saved tensors](https://docs.pytorch.org/docs/stable/notes/autograd.html#saved-tensors)。

### 7. 叶子、非叶子和 `.grad`

```python
x = torch.randn(4, requires_grad=True)  # leaf
y = x * 2                              # non-leaf
loss = y.sum()
loss.backward()
```

- 叶子 `x.grad` 默认累积。
- 非叶子 `y.grad` 默认不保留；需要时调用 `y.retain_grad()`。
- `y.grad_fn` 指向产生它的反向节点。
- `detach()` 返回与当前 autograd graph 脱离的 tensor view/handle 语义，不能用它“修复”本应存在的梯度。

接口见 [`is_leaf`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html) · [`retain_grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html) · [`detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)。

### 8. 原地操作为何危险

autograd 可能保存某 tensor 供 backward 使用。若之后原地修改，保存值不再是 forward 当时的值。PyTorch 用 version counter 做正确性检查，发现被保存 tensor 被改写时通常在 backward 报错。

不要用 `.data` 绕过检查；这可能生成“能跑但梯度错误”的结果。PyTorch 对 in-place correctness 的说明见 [In-place operations on Tensors](https://docs.pytorch.org/docs/stable/autograd.html#in-place-operations-on-tensors)。

---

## 二、梯度为何会消失、爆炸或变成非有限值

### 1. 链上反复相乘

深层网络的梯度包含多个局部 Jacobian 的乘积：

$$
\frac{\partial L}{\partial x_0}
=
\frac{\partial L}{\partial x_n}
\prod_{i=1}^{n}
\frac{\partial x_i}{\partial x_{i-1}}
$$

若乘数的典型尺度长期小于 1，梯度趋向消失；大于 1，趋向爆炸。实际网络是矩阵乘积，其谱性质、归一化、残差、初始化和激活共同决定传播。

### 2. 数值问题与优化问题要分开

| 现象 | 数值问题可能性 | 优化问题可能性 |
|---|---|---|
| 梯度变成 `inf`/`nan` | 溢出、非法运算、未稳定化公式 | 学习率过大也可先放大值再溢出 |
| 梯度精确变成大量 0 | fp16 下溢、flush-to-zero | 饱和激活、结构本身也可为 0 |
| 梯度有限但极大 | 尚未溢出 | 爆炸梯度、错误归一化 |
| loss 缓慢不降 | 误差累积可能 | 数据、模型、学习率等更常见 |
| CPU/GPU/NPU 从某层开始偏离 | 算子/精度/归约差异 | 若输入权重相同，先查数值路径 |

第一步不是调学习率，而是找首个异常 tensor 和异常 step。

### 3. 稳定公式比提高 dtype 更优先

例如：

$$
\log\sum_i e^{x_i}
$$

直接计算可能在 $x_i$ 较大时溢出。稳定形式：

$$
m+\log\sum_i e^{x_i-m},\quad m=\max_i x_i
$$

softmax、logsumexp、交叉熵、方差、归一化都有类似稳定实现。若公式本身不稳定，仅把 fp16 换 fp32 可能只是推迟失败。

---

## 三、fp16、bf16 与 fp32 到底差在哪里

### 1. 位分配

| dtype | 符号位 | 指数位 | fraction 位 | 有效精度约 | 最大有限值 | 最小正 normal | `eps`（1 附近间距） |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 5 | 10 | 约 3–4 位十进制 | 65504 | $2^{-14}\approx6.10\times10^{-5}$ | $2^{-10}\approx9.77\times10^{-4}$ |
| bf16 | 1 | 8 | 7 | 约 2–3 位十进制 | 约 $3.39\times10^{38}$ | $2^{-126}\approx1.18\times10^{-38}$ | $2^{-7}=0.0078125$ |
| fp32 | 1 | 8 | 23 | 约 7 位十进制 | 约 $3.40\times10^{38}$ | $2^{-126}\approx1.18\times10^{-38}$ | $2^{-23}\approx1.19\times10^{-7}$ |

`fraction` 不含 normal 数隐含的前导 1；因此 fp16 的 normal significand 精度为 11 bit，bf16 为 8 bit，fp32 为 24 bit。

PyTorch dtype 的 S-E-M 位分配见 [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)；当前运行环境的 `eps`、`max`、`tiny` 应直接查询 [`torch.finfo`](https://docs.pytorch.org/docs/stable/type_info.html#torch-finfo)。

```python
for dtype in [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]:
    info = torch.finfo(dtype)
    print(
        dtype,
        info.bits,
        info.eps,
        info.max,
        info.tiny,
    )
```

### 2. bf16 相比 fp16 牺牲什么、换来什么

bf16：

- 指数位与 fp32 相同，动态范围接近 fp32。
- fraction 只有 7 位，比 fp16 的 10 位少。
- 因此更不容易 overflow/underflow，但同一数量级内的间距更粗。

fp16：

- fraction 更多，所以在数值范围允许时比 bf16 有更细的相对精度。
- 指数范围窄，65504 以上溢出，微小 normal/subnormal 也更容易下溢。

结论不是“bf16 总比 fp16 准”，而是：

```text
bf16：范围优先，精度更粗
fp16：范围更窄，1 附近精度更细
```

格式对比也见 [NVIDIA low precision training introduction](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/introduction/introduction.html)。

### 3. normal、subnormal、flush-to-zero

normal 数使用隐含前导 1。指数到最小后，subnormal 用逐渐减小的有效位数继续靠近 0。

理论格式支持 subnormal 不等于所有硬件路径都完整保留。某些设备或算子为性能会把 subnormal flush to zero（FTZ），不同后端的结果可能因此在很小的梯度上分叉。判断时要看具体硬件、指令和运行模式，不能只看 dtype 名。

`torch.finfo.tiny` 是最小正 normal，不是最小 subnormal。文档见 [`torch.finfo`](https://docs.pytorch.org/docs/stable/type_info.html#torch-finfo)。

### 4. 舍入

实数或高精度结果落到有限格式时必须舍入。常见默认是 round-to-nearest, ties-to-even：

- 取最近的可表示数。
- 正好位于两者中点时，选择最低保留位为偶数者。

舍入使每一步产生小误差；连续 cast、长归约和不同融合方式会让误差路径不同。特定设备是否支持其他舍入或随机舍入，要看其文档和指令。

### 5. 范围与精度是两个轴

以下两种失败不同：

- **overflow**：绝对值超过最大有限值，常变成 `inf`。
- **rounding/quantization**：值仍在范围内，但低位被舍掉。

bf16 能避免很多 fp16 overflow，却可能因精度更粗让小更新量加不到大权重上：

```text
weight = 1.0
update = 0.001
```

bf16 在 1 附近的间距约为 0.0078125，若直接用 bf16 保存和更新权重，某些小更新可能完全被舍掉。这就是训练中常保留 fp32 master 参数/优化器状态的理由之一。

---

## 四、混合精度训练不是“全模型转成 fp16”

### 1. 基本目标

混合精度让适合低精度的算子使用 fp16/bf16，提高吞吐并降低内存/带宽压力，同时让数值敏感的算子或累加保留 fp32。

常见划分：

- 矩阵乘、卷积：低精度输入，硬件可能用更高精度累加。
- reduction、norm、softmax、loss：经常需要 fp32 中间量或输出。
- 参数更新和优化器状态：经常保留 fp32。

PyTorch AMP 的总体设计见 [`torch.amp`](https://docs.pytorch.org/docs/stable/amp.html)。

### 2. autocast 做什么

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    output = model(input)
    loss = loss_fn(output, target)
```

autocast 根据算子策略选择输入/计算 dtype。它不是递归把模型所有参数永久转换成半精度；常规做法是模型参数仍为 fp32，在 eligible op 处按策略转换。

明确传 `dtype=`、使用某些 in-place 或 `out=` 形式时，autocast 行为可能不同。具体 eligible op 和策略见 [`torch.amp` Autocast Op Reference](https://docs.pytorch.org/docs/stable/amp.html#autocast-op-reference)。

### 3. 累加 dtype 很关键

矩阵乘的乘法输入可以是 fp16/bf16，但部分积可能在 fp32 累加；归约也可能先转 fp32。两种实现都叫“输入 fp16”，结果精度却可能显著不同。

跨 NPU/GPU 对比时必须问：

- 输入 dtype。
- 输出 dtype。
- 内部 accumulator dtype。
- 是否使用 fused kernel。
- 是否允许降精度累加模式。
- 是否用了不同归约树。

只比较 Python 表面 dtype 不够。

### 4. 不应手工对所有东西 `.half()`

```python
model.half()
input = input.half()
```

这种纯半精度转换会让 norm、loss、reduction、参数更新等敏感路径也进入低精度，且未自动提供 loss scaling 和算子白/黑名单策略。它可用于特定推理或受控实验，不等于 AMP 训练。

---

## 五、loss scaling 为什么存在

### 1. 目标是防 fp16 梯度下溢

若某个 backward op 产生 fp16 梯度，小于可表示范围的值可能变为 0。把 loss 乘以比例 $S$：

$$
L' = S L
$$

链式法则使所有梯度一起放大：

$$
\nabla_\theta L' = S\nabla_\theta L
$$

optimizer 更新前再除以 $S$，数学上恢复原梯度：

$$
\frac{1}{S}\nabla_\theta L'=\nabla_\theta L
$$

这样中间反向传播中的小梯度更可能落在 fp16 可表示范围内。PyTorch 的解释见 [`torch.amp`: Gradient Scaling](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling)。

### 2. 典型训练顺序

```python
scaler = torch.amp.GradScaler("cuda")

for input, target in loader:
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        output = model(input)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )
    scaler.step(optimizer)
    scaler.update()
```

关键顺序：

1. scale loss。
2. backward 得到 scaled grads。
3. 在查看/裁剪真实梯度前 unscale。
4. 若发现非有限梯度，跳过 optimizer step。
5. update 调整 scale。

AMP 完整示例见 [Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

### 3. 动态 loss scaling

一种典型策略：

- 从较大 scale 开始。
- 一段时间没有 overflow，增大 scale。
- 检测到 `inf`/`nan`，减小 scale 并跳过本次参数更新。

scale 不保证永远大于 1；如果模型的数值超出 fp16 范围，动态 scaler 可能持续减小。动态算法说明见 [NVIDIA Mixed Precision Training: Loss Scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling) 和 PyTorch [`GradScaler`](https://docs.pytorch.org/docs/stable/amp.html#torch.amp.GradScaler)。

### 4. 为什么 bf16 往往不需要 loss scaling

bf16 的指数范围与 fp32 接近，小梯度因“范围不足”而下溢的风险远小于 fp16，所以通常不需要专门通过 loss scaling 扩大动态范围。

但“通常不需要”不等于“bf16 没有数值问题”：

- bf16 fraction 少，舍入更粗。
- 某些硬件会 flush subnormal。
- 不稳定公式仍会产生 `inf`/`nan`。
- 低精度累加仍会放大误差。
- 梯度可能因模型数学结构而真正接近 0。

因此自测题的完整答案是：**bf16 用更少 significand 精度换取接近 fp32 的指数范围；loss scaling 主要解决 fp16 动态范围导致的梯度下溢，所以 bf16 通常可不使用，但仍需验证具体模型和后端。**

### 5. loss scaling 的常见错误

- 梯度未 unscale 就做 clipping，阈值语义被 scale 改变。
- 对同一个 optimizer 重复 unscale。
- 手工 `optimizer.step()` 绕过 scaler 的非有限检查。
- 梯度累积时每个 microbatch 随意改变 scale，导致累积的梯度不在同一尺度。
- 多 rank 是否跳步不同步，造成参数立即分叉。
- 把 forward activation overflow 误当成 loss scaling 能解决；scaling 只直接放大反向梯度，不能修复 forward 已经溢出的值。

---

## 六、优化器状态到底占多少显存

### 1. 通用计算公式

设参数个数为 $N$，某状态 dtype 每元素字节数为 $B$，则：

$$
\text{memory}=N\times B
$$

十进制 GB 与二进制 GiB 不同：

$$
1\text{ GB}=10^9\text{ bytes},\quad
1\text{ GiB}=2^{30}\text{ bytes}
$$

估算时明确单位，避免“70B 参数 × 多少字节”的口径混乱。

### 2. 常见 dtype 字节数

| dtype | bytes/element |
|---|---:|
| fp64 | 8 |
| fp32 | 4 |
| fp16 | 2 |
| bf16 | 2 |
| int64 | 8 |
| int32 | 4 |
| int8 | 1 |

PyTorch dtype 列表见 [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)。

### 3. 只算模型状态的基础账

假设参数和梯度都是 fp32：

| 训练配置 | 参数 | 梯度 | optimizer state | 合计 |
|---|---:|---:|---:|---:|
| SGD，无 momentum | 4N | 4N | 0 | 8N bytes |
| SGD + momentum | 4N | 4N | 4N | 12N bytes |
| Adam/AdamW，m/v 为 fp32 | 4N | 4N | 8N | 16N bytes |

Adam 有一阶矩 $m$ 和二阶矩 $v$，各一个与参数同 shape 的状态 tensor；算法见 [`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)。每个 parameter tensor 还可能有 step 等小型元数据，但大模型中主要量级由逐元素状态决定。

### 4. “混合精度是 16 bytes/param”不是永恒定律

不同实现可能是：

#### A. autocast，模型参数保持 fp32

```text
fp32 parameter     4
fp32 gradient      4
fp32 Adam m/v      8
--------------------
合计              16 bytes/param
```

低精度 op 输入可能有临时 cast/cache，但不应机械地再加一份永久 fp16 参数。

#### B. 低精度模型参数 + 低精度梯度 + fp32 master 参数

```text
fp16/bf16 parameter  2
fp16/bf16 gradient   2
fp32 master param    4
fp32 Adam m/v        8
----------------------
合计                16 bytes/param
```

#### C. 低精度参数，但梯度/状态策略不同

若 gradient 是 fp32，或 optimizer state 量化到 8 bit，结果会变化。FSDP/ZeRO 还会分片这些状态；offload 会把部分状态移到 host，但增加传输和 host 内存。

所以正确方法是**枚举实际存在的 tensor、dtype、是否分片、是否 offload**，不要死背单一数字。

### 5. 例：10 亿参数 Adam

按 16 bytes/param：

$$
10^9\times16=16\times10^9\text{ bytes}
$$

即约 16 GB，或：

$$
\frac{16\times10^9}{2^{30}}\approx14.90\text{ GiB}
$$

这只含参数、梯度和 optimizer states，不含：

- forward 保存的 activation。
- 临时 workspace。
- allocator reserved 但未 active 的块。
- 通信 bucket。
- fused kernel 临时 buffer。
- CUDA/NPU context 与库工作区。
- 数据输入和 host pinned memory。

### 6. activation 为什么常是训练显存大头

activation 规模大致随：

$$
\text{batch}
\times
\text{sequence length}
\times
\text{hidden size}
\times
\text{layers}
$$

增长，并受 attention 中间量、MLP 扩张比例、保存策略和 dtype 影响。长序列时 attention 朴素中间矩阵还可能按序列长度平方增长。

activation checkpointing 用额外重算换取少保存中间量；原理和 API 见 [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html)。

### 7. allocated、reserved、峰值不要混淆

caching allocator 通常会保留已向设备申请、当前未被 tensor 使用的块。因此：

```text
tensor active/allocated memory
≤ allocator reserved memory
≤ 设备进程可见总占用（通常还含上下文/外部库）
```

“删了 tensor 但系统工具占用没降”可能只是块回到 allocator cache，不代表仍有 tensor 引用。具体 NPU allocator 指标应以当前 torch_npu 版本为准；概念可对照 PyTorch [Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)。

---

## 七、精度问题的分层定位

### 1. 先建立 fp32 基线

同一份：

- 模型结构。
- 权重。
- 输入和 label。
- 随机种子。
- train/eval 状态。

先在 fp32 单卡执行一个或少量 step。fp32 基线也不等于数学真值，但它能显著缩小低精度和后端差异的搜索空间。

### 2. 找第一个偏离点

按 module 或 operator 保存：

- 输入。
- 输出。
- dtype/device/shape/stride。
- `max_abs`、`mean_abs`、有限性。
- 与参考结果的 `max_abs_error`、`max_rel_error`、mismatch 比例。

找到第一层明显偏离后，再把该层缩成单算子复现。不要从最终 loss 倒猜。

### 3. 前向与反向分开

| 检查 | 目的 |
|---|---|
| 同权重同输入对比 forward | 判断偏差是否已经在前向 |
| 前向对齐后对比 loss | 查 loss/stability |
| 对同一个 output 注入相同 `grad_output` | 隔离 backward 实现 |
| 比较每个参数 grad | 找第一个反向分叉 |
| 固定 grad 直接跑 optimizer step | 隔离 optimizer |

自定义 backward 应用 [`gradcheck`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) 在 double precision、小规模输入上验证解析梯度；低精度不适合作为 finite difference 的默认检查 dtype。

### 4. 监测有限性与尺度

```python
def summarize(name, tensor):
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    print(
        name,
        "shape=", tuple(value.shape),
        "finite=", finite.float().mean().item(),
        "max_abs=", value.abs().max().item(),
        "mean_abs=", value.abs().mean().item(),
    )
```

在 NPU 上调用 `.item()` 会同步设备；排障时可接受，性能测量时应移除。若 tensor 很大，采样或在设备上归约后只取少量标量。

### 5. 第一处 `nan` 的常见来源

- `log(x)` 的 `x <= 0`。
- `sqrt(x)` 的 `x < 0`。
- 除零或极小分母。
- `exp(x)` 溢出。
- `inf - inf`。
- `0 * inf`。
- softmax/logsumexp 未稳定化。
- norm 的方差/epsilon 路径差异。
- 未初始化内存或越界写。
- 通信前某 rank 已产生非有限值，归约后扩散。

### 6. 看分布，不只看最大误差

最大误差可能由参考值接近 0 的单个元素主导。至少同时看：

- max/mean absolute error。
- max/median/高分位 relative error。
- cosine similarity。
- 超过容差的元素比例。
- 按幅值区间分桶后的误差。
- `nan`/`inf` 数量与位置。

误差度量的边界见 [[floating-point-error-analysis]]。

---

## 八、跨设备适配时必须控制的变量

### 1. dtype 路径

- 输入和参数表面 dtype。
- autocast 是否开启，目标 dtype。
- 敏感 op 是否强制 fp32。
- accumulator dtype。
- optimizer state dtype。
- cast 的位置和次数。

### 2. 算法路径

- 是否使用 fused op。
- eager 与图模式是否不同。
- dynamic shape 是否选择不同 kernel。
- reduce/tree/ring 顺序是否不同。
- deterministic 配置是否相同。
- 是否存在 CPU fallback。

### 3. 随机性

- 参数初始化。
- dropout。
- 数据顺序。
- 分布式 sampler。
- 后端 RNG 算法和随机数消费顺序。

固定相同 seed 不保证不同后端逐 bit 使用相同随机流。精度对比时可先关闭 dropout，或直接保存并复用输入与 mask。

### 4. 异步执行

真正失败的算子可能在稍后的同步点报错。插入同步做二分，与 [[linux-debugging-for-npu-adaptation]] 配合；定位完成后移除同步。

---

## 九、常见误区

1. **bf16 范围大，所以比 fp16 更精确。** 范围大，但 significand 更短。
2. **AMP 就是 `model.half()`。** AMP 是按算子选择精度，并常与 scaler 配合。
3. **loss scaling 能修 forward overflow。** 它主要防反向 fp16 梯度下溢。
4. **scaler 的 scale 必须大于 1。** 动态 scaler 不作此保证。
5. **clip scaled gradient。** 应先 unscale，再按真实梯度尺度裁剪。
6. **Adam 永远 16 bytes/param。** 要看参数、grad、master、state 的实际 dtype 和分片。
7. **模型参数显存就是训练显存。** activation、workspace、bucket 和 allocator cache 可能更大。
8. **最终 loss 接近就说明每层正确。** 误差可能抵消，也可能尚未传播到 loss。
9. **不同设备不逐 bit 一致就是 bug。** 浮点归约与算法路径不同会产生合理差异；要看容差、误差增长形态和首个分叉点。

---

## 十、两周学习与动手

### 第 1–3 天：计算图和 VJP

- 手算两层 MLP 的 forward/backward。
- 用 `grad_fn`、`next_functions` 观察图。
- 验证分支处梯度累加。

### 第 4–5 天：dtype

- 打印 fp16/bf16/fp32 的 `torch.finfo`。
- 构造接近 max、tiny、eps 的值，观察 cast。
- 比较 fp16 与 bf16 在“范围”和“1 附近间距”上的差别。

### 第 6–8 天：AMP 与 loss scaling

- 跑同一小模型的 fp32、fp16 AMP、bf16 AMP。
- 每步记录 loss scale、是否跳步、grad norm。
- 故意制造 fp16 overflow 和 underflow。

### 第 9–10 天：显存账本

- 枚举 `model.parameters()` 与 optimizer state 的 shape/dtype。
- 手算字节数并与运行时统计对照。
- 分开记录 allocated、reserved、峰值。

### 第 11–14 天：精度二分

- 给小模型注册 forward hook。
- 对比两个后端每层输出。
- 找第一个偏离层，拆成单算子。
- 用相同 `grad_output` 单独验证 backward。

---

## 十一、自测题

- 为什么一个 tensor 被两条后续路径使用时，反向梯度要相加？
- 非标量输出调用 `backward` 时传入的 gradient 是什么？
- fp16 和 bf16 的指数位、fraction 位分别是多少？
- bf16 相比 fp16 牺牲什么、换来什么？
- 为什么 bf16 通常不需要 loss scaling，但仍可能有精度问题？
- loss scaling 为什么不能修复 forward 中已经出现的 `inf`？
- 为什么 gradient clipping 要在 unscale 之后？
- fp32 Adam 的参数、梯度和 m/v 各占多少 bytes/param？
- autocast 下为什么不能机械地额外计算一份永久 fp16 参数？
- activation checkpointing 省了什么，付出了什么？
- GPU/NPU 最终 loss 不同，怎样区分 forward、backward 和 optimizer 三段？

---

## 相关

[[npu-training-adaptation-learning-path]] · [[floating-point-error-analysis]] · [[npu-precision-debugging]] · [[linux-debugging-for-npu-adaptation]] · [[cpp-reading-for-pytorch-backends]] · [[python-advanced-mechanisms-for-pytorch]] · [[zero-and-fsdp]]
