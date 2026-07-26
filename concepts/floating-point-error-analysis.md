---
title: 训练适配所需的浮点误差分析常识
type: concept
created: 2026-07-26
updated: 2026-07-26
tags: [数值分析, 浮点数, 舍入误差, 相对误差, 绝对误差, 精度对比]
sources:
  - https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
  - https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
  - https://docs.python.org/3/library/math.html#math.isclose
  - https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose
  - https://docs.pytorch.org/docs/stable/type_info.html
---

**浮点误差分析的核心是区分“实数问题本身有多敏感”和“有限精度算法额外放大了多少误差”，并用同时含绝对项与相对项的尺度化标准判断差异。**

浮点结果不同不自动等于 bug，结果接近也不自动证明实现正确。NPU 适配中最重要的判断是：误差是否符合 dtype、归约顺序和算法路径所能解释的范围，还是从某个算子开始出现了突变、非有限值或结构性偏差。

---

## 一、浮点数是有限集合，不是实数

### 1. 大多数实数无法精确表示

有限 bit 只能编码有限个值。十进制 `0.1` 在二进制中是无限循环，存成 binary floating point 时必须舍入到邻近的可表示值。

因此：

```python
0.1 + 0.2 == 0.3
```

常为 `False`。这不是 Python 或硬件算错，而是三个十进制字面值都先映射到了近似二进制数。浮点表示和舍入的经典说明见 Goldberg 的 [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)。

### 2. 可表示数的间距随量级变化

对 normal binary floating point，同一指数区间内相邻数间距固定；数值跨到更大指数区间，间距也按 2 的幂增大。

这意味着：

- 在 1 附近能表达的小增量，到 \(10^8\) 附近可能完全加不上去。
- 绝对精度不是全范围固定的。
- `eps` 只描述 1.0 与下一个可表示数的间距，不是任意数附近的绝对误差上界。

当前 dtype 的 `eps`、`max`、`tiny` 可用 [`torch.finfo`](https://docs.pytorch.org/docs/stable/type_info.html#torch-finfo) 查询。

### 3. ULP

ULP（unit in the last place）可理解为某数附近相邻可表示浮点数的间距。用“差多少 ULP”衡量误差，比固定绝对误差更贴近表示精度，但：

- 跨 0、subnormal、符号和 NaN 时要谨慎定义。
- 对应用正确性，ULP 小不一定代表业务误差小。
- 对不同 dtype，1 ULP 的绝对大小不同。

Goldberg 论文的 Relative Error and Ulps 小节见 [Rounding Error](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#689)。

---

## 二、一个常用舍入误差模型

对处于 normal 范围、没有 overflow/underflow 且采用合理舍入的基本运算，可用：

\[
\operatorname{fl}(x\circ y)
=
(x\circ y)(1+\delta),
\quad |\delta|\le u
\]

其中：

- \(\circ\) 是加、减、乘、除等运算。
- `fl` 表示结果舍入到目标浮点格式。
- \(u\) 是 unit roundoff，量级与机器精度相关。

这是分析模型，不是所有场景的无条件保证：

- 精确结果接近 0 时，相对形式可能失效。
- subnormal、flush-to-zero 会改变模型。
- overflow 直接产生 `inf`，不能用小 \(\delta\) 解释。
- 超越函数、低精度近似和 fast math 有各自误差。
- 复合算子可能使用 FMA 或更高精度中间量。

浮点基础与异常情况见 [Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)。

---

## 三、浮点加法为什么不满足结合律

### 1. 数学结合律与机器运算

实数中：

\[
(a+b)+c=a+(b+c)
\]

浮点中，每次加法后都舍入：

\[
\operatorname{fl}(
  \operatorname{fl}(a+b)+c
)
\]

与：

\[
\operatorname{fl}(
  a+\operatorname{fl}(b+c)
)
\]

中间结果不同，所以最终可能不同。

例：

```python
import torch

a = torch.tensor(1e20, dtype=torch.float32)
b = torch.tensor(-1e20, dtype=torch.float32)
c = torch.tensor(3.14, dtype=torch.float32)

left = (a + b) + c
right = a + (b + c)
```

第一种通常先得到 0，再加 3.14；第二种在 `b + c` 时 3.14 相对 \(10^{20}\) 太小而被舍掉，再与 \(10^{20}\) 相消，通常得到 0。

PyTorch 明确不保证数学等价的浮点计算逐 bit 相同，因为加法和乘法不满足结合律。见 [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

### 2. reduction 顺序为何会变

求和：

\[
s=\sum_{i=1}^n x_i
\]

可按多种顺序执行：

```text
顺序累加：(((x1+x2)+x3)+...)
树形归约：(x1+x2) + (x3+x4) + ...
分块归约：每个线程/AI Core 算一块，再合并
原子累加：到达顺序可能不固定
```

GPU/NPU 为并行性能通常使用树形或分块归约；不同 shape、线程数、kernel、融合和通信算法都可能改变求和树。即使每次基本加法都正确舍入，最终低位仍可能不同。

### 3. 分布式归约同样如此

allreduce 的 ring、tree、分层拓扑会以不同顺序合并各 rank 数据。单卡一致、多卡出现小差异时，应检查：

- 归约算法。
- rank 顺序。
- 累加 dtype。
- 分块大小。
- 是否存在非确定性原子操作。

不同不是自动错误；误差是否随 world size、层数和 step 平滑增长，是判断的重要证据。

---

## 四、误差如何累积

### 1. 最坏界与典型行为不同

若每一步都有量级 \(u\) 的舍入误差，简单地说“做 \(n\) 次就是 \(nu\)”只是一种小 \(nu\) 条件下的最坏量级直觉。更严谨的经典界常写为：

\[
\gamma_n=\frac{nu}{1-nu}
\]

前提是 \(nu<1\)，并依赖具体运算结构。

若误差符号近似随机，实际增长有时更接近 \(\sqrt{n}u\) 的统计量级；但训练中的误差可能相关、被非线性放大，不能默认随机抵消。

### 2. 长链乘法与长归约是不同机制

- 长归约主要积累加法舍入和 cancellation。
- 深层反向主要包含 Jacobian 连乘，可能在数学上放大或缩小输入扰动。
- optimizer 跨 step 反馈：本 step 的权重差异改变下 step 的激活和梯度，误差不再只是独立加法。

因此“单算子误差 1e-3，100 层就是 0.1”通常不成立。需要测量误差传播曲线，而不是机械线性外推。

### 3. 灾难性消去

两个接近的大数相减：

\[
x-y
\]

若 \(x\) 和 \(y\) 本身已有舍入误差，相减后有效高位抵消，留下的结果可能主要由原误差构成。

例：

```text
x = 1.234567
y = 1.234566
x - y = 0.000001
```

输入末位一点误差，相对于很小的差值会被巨大放大。

常见场景：

- `variance = E[x^2] - E[x]^2`。
- 两个接近 loss 的差。
- 小角度/小距离公式。
- 直接求二次方程某个根。
- `log(1+x)` 在很小 \(x\) 时若先算 `1+x`。

稳定实现常使用 Welford variance、`log1p`、`expm1` 等专门公式。

### 4. 小数加到大数可能完全消失

若大数附近的 ULP 大于小增量：

\[
\operatorname{fl}(x+\Delta)=x
\]

这对低精度参数更新很重要：bf16 权重为 1.0 时，远小于其局部间距的 update 可能无法改变存储值。保留 fp32 master 参数能让小更新累积，而不是每步都在 bf16 落盘时消失。

---

## 五、问题条件数与算法稳定性

### 1. 条件数：问题本身对输入扰动多敏感

若输入相对扰动很小，输出相对变化很大，则问题 ill-conditioned。对标量函数的局部相对条件数可写成：

\[
\kappa(x)
=
\left|
\frac{x f'(x)}{f(x)}
\right|
\]

\(\kappa\) 大时，哪怕算法非常稳定，输入中不可避免的舍入也会造成明显输出误差。

例：计算 \(f(x)=x-c\) 且 \(x\approx c\)。结果接近 0，相对条件数很大。

### 2. 稳定性：算法引入了多少额外误差

同一个数学问题可有稳定或不稳定算法。好的 backward-stable 算法可理解为：算出的结果等于“一个略微扰动输入”的精确答案。

排障时要分：

- 输入本身是否靠近奇异点/边界。
- 参考实现是否也不稳定。
- NPU 实现是否使用了不同但稳定的算法。
- 偏差是否来自低精度输入，还是算法额外放大。

PyTorch 对线性代数中 ill-conditioned 输入的提醒见 [Extremal values and linear algebra](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#extremal-values)。

### 3. 参考值不一定是真值

GPU fp32 输出只是常用参考，不是数学真值。严谨验证可逐级提高：

1. 同 dtype 的 CPU/GPU/NPU 交叉比较。
2. fp64 参考。
3. 小规模高精度库或符号结果。
4. 已知不变量、解析解和 metamorphic test。

若 fp32 两种实现不同，用 fp64 或更稳定公式判断哪一个更接近真值。

---

## 六、绝对误差与相对误差

### 1. 定义

参考值 \(r\)，待测值 \(a\)。

绝对误差：

\[
E_\text{abs}=|a-r|
\]

相对误差：

\[
E_\text{rel}
=
\frac{|a-r|}{|r|}
\]

### 2. 绝对误差什么时候好用

适合：

- 参考值接近 0。
- 业务量有固定物理尺度。
- 小于某个固定噪声地板即无意义。

失效场景：

- 数值跨度跨许多数量级。
- `atol=1e-5` 对 \(10^8\) 的量太严，对 \(10^{-10}\) 的量又可能太松。

### 3. 相对误差什么时候好用

适合：

- 参考值明显非零。
- 关注“误差占信号的比例”。
- tensor 元素跨尺度，但各自相对精度要求近似。

失效场景：

- \(r=0\) 时分母为 0。
- \(r\) 很小时，相对误差爆炸，即使绝对差极小。
- 只用参考值作分母会产生方向不对称：交换 actual/reference 得到不同结果。

### 4. 组合容差

NumPy `assert_allclose` 使用：

\[
|a-r|
\le
\mathrm{atol}
+
\mathrm{rtol}|r|
\]

其中：

- `atol` 给零附近一个绝对误差地板。
- `rtol` 控制非零大数的相对误差。

对应文档：[`numpy.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose)。

Python `math.isclose` 使用对称相对尺度：

\[
|a-b|
\le
\max(
  \mathrm{rel\_tol}\max(|a|,|b|),
  \mathrm{abs\_tol}
)
\]

对应文档：[`math.isclose`](https://docs.python.org/3/library/math.html#math.isclose)。

PyTorch 的 `torch.testing.assert_close` 采用逐元素 close 判定，并能检查 dtype/device/layout 等属性；具体默认值随 dtype 而异，测试中最好显式说明容差依据。见 [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。

### 5. 容差不是拍脑袋常数

设置容差前考虑：

- 输入/输出 dtype 的 unit roundoff。
- 运算次数和 reduction 长度。
- 条件数。
- 参考值是否更高精度。
- accumulator dtype。
- 是否跨设备、融合或分布式。
- 业务允许的误差。

好的测试会解释：

```text
为什么选择这个 rtol/atol
哪个 dtype 和算法假设支持它
失败时输出哪些误差统计
```

### 6. 对 tensor 不应只报一个 max relative error

建议同时计算：

```python
diff = (actual - reference).abs()
scale = reference.abs()
combined_limit = atol + rtol * scale
mismatch = diff > combined_limit

report = {
    "max_abs": diff.max(),
    "mean_abs": diff.mean(),
    "max_rel_nonzero": (
        diff / scale.clamp_min(rel_floor)
    ).max(),
    "mismatch_fraction": mismatch.float().mean(),
}
```

`rel_floor` 只用于统计稳定，不应偷偷替代测试定义。报告还应包含 NaN/Inf 数量、最差元素的 reference/actual/索引。

---

## 七、NaN、Inf、signed zero 和 subnormal

### 1. NaN

NaN 与任何值（包括自身）的普通相等比较都为 false。比较 tensor 时必须明确：

- 两边同位置 NaN 是否视为一致。
- NaN 的出现本身是否应直接失败。
- quiet/signaling NaN 和 payload 是否相关。

训练精度测试通常应把“出现任何意外 NaN”作为独立失败，不要让 allclose 的 `equal_nan=True` 掩盖。

### 2. Inf

同号 `inf` 在某些 close 函数中可判相等，但两个实现都 overflow 不代表算法正确。若参考高精度结果有限，而两边低精度都为 `inf`，仍是范围/稳定性问题。

### 3. signed zero

`+0.0 == -0.0` 为真，但某些函数如倒数、复数 branch cut 可能区分符号。深度学习大多数比较无需区分，但底层数值库测试可能需要检查 bit pattern。

### 4. subnormal

subnormal 提供逐渐下溢，但有效精度降低。硬件 FTZ/DAZ 模式可能把它们当 0，造成：

- 小梯度直接消失。
- CPU 与加速器在零附近不同。
- 相对误差失去意义。

此时优先报告绝对量级、非零比例以及是否处在 `tiny` 以下。

---

## 八、改善数值稳定性的常见方法

### 1. 更高精度累加

低精度输入、fp32 accumulator 是矩阵乘和 reduction 的常见折中。它减少累加误差，但输入在 cast 时丢失的信息无法恢复。

### 2. pairwise/tree summation

顺序累加长序列时，小数不断加到越来越大的 partial sum。pairwise summation 让相近规模的部分和先合并，通常把误差增长从线性量级改善到对数深度量级。

并行 reduction 的树形结构因此既是性能设计，也是数值路径的一部分。

### 3. Kahan compensated summation

```python
def kahan_sum(values):
    total = 0.0
    correction = 0.0
    for value in values:
        adjusted = value - correction
        next_total = total + adjusted
        correction = (
            next_total - total
        ) - adjusted
        total = next_total
    return total
```

补偿变量保留普通加法丢失的低位。它增加运算和数据依赖，未必适合所有设备 kernel，但能作为高精度参考或理解工具。

Python 的 [`math.fsum`](https://docs.python.org/3/library/math.html#math.fsum) 提供精确跟踪多个中间部分和的浮点求和实现，可用于小规模 CPU 参考。

### 4. 稳定等价变形

| 不稳定形式 | 更稳定思路 |
|---|---|
| `log(1 + x)`，`x` 很小 | `log1p(x)` |
| `exp(x) - 1`，`x` 很小 | `expm1(x)` |
| `log(sum(exp(x)))` | 减去 `max(x)` |
| `E[x²] - E[x]²` | Welford / centered two-pass |
| sigmoid 直接 `exp(-x)` 全范围计算 | 按符号分支或用稳定库函数 |
| softmax 后再 log | 直接 log-softmax |

### 5. 缩放

- loss scaling 把小梯度移入 fp16 可表示范围。
- norm/standardization 控制激活尺度。
- 线性代数中可先对输入做尺度归一化，再恢复量纲。

缩放要配套逆变换，且不能改变优化器/正则化语义。

### 6. 避免反复 cast

```text
fp32 → bf16 → fp32 → bf16
```

每次降精度都会舍入。第二次转回 fp32 只是补 0/扩展表示，不能找回已丢位。检查图中冗余 cast 和格式转换既有性能价值，也有精度价值。

---

## 九、融合、FMA 与“更不同但可能更准”

### 1. FMA

普通：

\[
\operatorname{fl}(
  \operatorname{fl}(a\times b)+c
)
\]

FMA：

\[
\operatorname{fl}(a\times b+c)
\]

FMA 中乘加只在最终舍入一次，通常更准确，但与分开乘加的 bit pattern 不同。编译器、硬件和 fused kernel 是否使用 FMA，会导致跨平台差异。

### 2. 图融合

融合可：

- 避免中间结果落盘和低精度 round-trip。
- 在寄存器或更高精度中保留中间量。
- 改变运算顺序。
- 使用近似数学函数。

所以“开图模式结果不同”既可能是合理舍入差异，也可能是融合规则 bug。应找到第一处 fused region，分别运行融合前后并用高精度参考判断。

### 3. fast math

fast math 可能允许：

- 近似超越函数。
- 重排假设。
- 忽略 signed zero、NaN/Inf 的某些严格语义。
- flush subnormal。

性能适配中不能只比较吞吐，必须明确精度合同。具体开关和保证以编译器/后端文档为准。

---

## 十、确定性、可复现与数值接近是三件事

### 1. bitwise deterministic

同一输入、同一环境、多次运行得到完全相同 bit pattern。它要求：

- 算法确定。
- 并发归约顺序固定。
- RNG 状态一致。
- 编译和库路径一致。

### 2. reproducible within tolerance

结果不逐 bit 相同，但统计或逐元素误差始终在合同范围内。这是很多并行训练更现实的目标。

### 3. mathematically/semantically correct

算法满足业务或数学误差要求。即使 bitwise deterministic，也可能稳定地产生错误结果；即使不确定，也可能每次都在可接受误差内。

PyTorch 不保证不同 release、平台或批量/切片路径的浮点结果逐 bit 相同；见 [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) 与 [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

---

## 十一、NPU/GPU 精度对比的实用流程

### 1. 固定输入和执行路径

- 直接保存同一输入与权重，不只固定 seed。
- 关闭 dropout 或复用相同 mask。
- 固定 train/eval。
- 记录 autocast、dtype、精度模式、图/单算子模式。
- 单卡先比较。

### 2. 建立参考层级

```text
fp64 CPU（小规模）
    ↓
fp32 CPU/GPU
    ↓
fp32 NPU
    ↓
bf16/fp16 GPU 与 NPU
```

逐级比较能区分：

- 公式/实现错误。
- 后端 fp32 路径差异。
- 低精度 cast/accumulation 差异。

### 3. 逐层找到首个分叉

记录每层：

```text
shape / dtype / device
finite ratio
min / max / mean / norm
max abs error
高分位 relative error
mismatch fraction
```

若误差从第一层开始小且随深度平滑增长，可能是合理累积；若某层突然扩大多个数量级，优先查该层算子、输入范围、融合和 accumulator。

### 4. 设计有针对性的输入

随机正态输入不覆盖所有边界。加入：

- 0、正负极小值。
- dtype 的 `tiny`、`eps`、`max` 附近。
- 大小量级混合。
- 正负相消。
- 非 contiguous、不同 stride。
- 极长 reduction 维。
- 包含 NaN/Inf（若 API 定义了行为）。

### 5. 区分结构性错误和舍入误差

结构性错误的常见信号：

- shape/stride/layout 不对。
- 只有某些索引或 tile 错。
- 误差与输入值无关，出现固定偏移。
- 输出含未初始化样式的随机大值。
- 原地/alias 行为不对。
- 某 dtype 完全错误，其他 dtype 正常。

舍入误差常见信号：

- 误差与输出量级相关。
- reduction 越长误差越大。
- 改高精度 accumulator 明显改善。
- 没有突变、越界或离散索引模式。

---

## 十二、测试容差的设计模板

### 1. 先写误差预算

```text
输入 dtype：bf16
参考：fp32 同算法
核心运算：长度 K 的 dot product
被测路径：bf16 输入、fp32 accumulate
风险：输入量化 + K 项累加 + 输出 cast
```

再通过：

- 理论量级。
- 多组 shape/dtype 实测。
- 高精度参考。
- 业务阈值。

确定 rtol/atol。

### 2. 失败信息要能定位

测试失败至少输出：

- dtype、shape、seed。
- rtol、atol。
- max absolute/relative error。
- mismatch 数量和比例。
- 最差位置的 actual/reference。
- NaN/Inf 数量。

### 3. 不要用过宽容差隐藏 bug

若某算子理论应精确（如整数索引、copy、某些排列），就不应使用普通浮点宽容差。对浮点算子，也应分别测试：

- 通常范围。
- 边界范围。
- 不同 reduction 长度。
- 不同 layout。

容差随 shape 任意扩大，往往说明误差模型或实现尚未理解。

---

## 十三、常见误区

1. **浮点误差就是随机噪声。** 它可能有系统偏差并被非线性放大。
2. **所有运算每步最多 `eps` 绝对误差。** `eps` 是 1 附近相对间距量级。
3. **加法数学上结合，所以编译器/设备可随意重排且结果相同。** 浮点 bit 结果会变。
4. **relative error 能处理所有尺度。** reference 接近 0 时会失效。
5. **absolute error 足够。** 大数范围下固定 atol 没有尺度感。
6. **两边都是 inf 就算一致。** 可能只是两边都溢出。
7. **GPU fp32 是真值。** 它仍是有限精度实现。
8. **逐 bit 不同就是 bug。** 并行归约、FMA、融合都会合理改变低位。
9. **提高 dtype 一定修复。** ill-conditioned 问题或不稳定公式仍可能失败。
10. **最终 loss close 就完成验证。** 中间结构性错误可能被抵消或尚未放大。

---

## 十四、两周学习与实验

### 第 1–3 天：表示与舍入

- 打印各 dtype 的 `finfo`。
- 观察 1、\(10^4\)、\(10^8\) 附近相邻可表示数。
- 演示小数加到大数后消失。

### 第 4–5 天：非结合律与归约

- 用不同顺序求同一数组之和。
- 比较顺序、pairwise、fp64 accumulator、`math.fsum`。
- 改变 reduction 长度并画误差。

### 第 6–7 天：消去与稳定公式

- 比较 `log(1+x)` 与 `log1p(x)`。
- 比较 naive variance 与 Welford。
- 构造相近大数相减。

### 第 8–9 天：误差指标

- 构造 reference 为 0、极小、大数的案例。
- 比较只用 rtol、只用 atol、组合标准。
- 写一个包含分位数和 mismatch ratio 的 tensor 报告。

### 第 10–12 天：后端比较

- 对长 reduction 比较 CPU/GPU/NPU。
- 分别使用 fp16、bf16、fp32。
- 记录 accumulator/fusion/shape 对误差的影响。

### 第 13–14 天：测试设计

- 为一个算子写正常值、边界值、非 contiguous 和长归约测试。
- 给每个容差写出依据。
- 故意注入固定偏移或索引错误，确认测试不会被宽容差放过。

---

## 十五、自测题

- 为什么 `eps` 不是全数轴上的固定绝对误差？
- 用三个数举例说明浮点加法不满足结合律。
- parallel reduction 为什么会让 GPU/NPU 结果低位不同？
- 误差什么时候可能近似按 \(nu\) 增长，什么时候这个直觉不可靠？
- 灾难性消去为何会放大输入的相对误差？
- reference 接近 0 时相对误差为什么失效？
- 大数比较时只用固定 atol 为什么失效？
- `atol + rtol * |reference|` 两项分别负责什么？
- 问题 ill-conditioned 与算法不稳定有什么区别？
- FMA 为什么可能与分开乘加不同，却反而更准确？
- bitwise deterministic、tolerance reproducible、semantically correct 有什么区别？
- 如何判断某层突然增大的误差更像结构性 bug，而不是正常舍入？

---

## 相关

[[npu-training-adaptation-learning-path]] · [[deep-learning-training-numerics]] · [[npu-precision-debugging]] · [[linux-debugging-for-npu-adaptation]] · [[cpp-reading-for-pytorch-backends]]
