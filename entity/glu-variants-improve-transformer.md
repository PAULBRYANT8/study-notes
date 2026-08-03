---
title: GLU Variants Improve Transformer
type: entity
created: 2026-07-27
updated: 2026-07-27
tags: [transformer, GLU, GEGLU, SwiGLU, FFN, 激活函数, 论文]
sources:
  - "[[2026-07-27-glu-variants-improve-transformer]]"
  - https://arxiv.org/abs/2002.05202
---

《GLU Variants Improve Transformer》是 Noam Shazeer 于 2020 年提出的一篇论文，核心结论是：在参数量和计算量大致匹配时，用 GEGLU、SwiGLU 等门控结构替换 Transformer 的普通 FFN，能改善 T5 式预训练的困惑度，并在多个下游任务上带来小幅增益。

## 一句话理解

这篇论文的关键不是“换一个更好的激活函数”，而是“把 FFN 的中间表示改成两条投影分支的逐元相乘”；长程预训练中，非门控的 GELU 和 Swish 没有超过 ReLU，而论文测试的所有门控/双线性变体都超过了 ReLU。

## 核心改动

标准 Transformer FFN 可写成：

`FFNφ(x) = φ(xW₁)W₂`

论文把它改成：

`FFNφ-GLU(x) = (φ(xW) ⊙ xV)W₂`

其中 `⊙` 是逐元相乘，`φ(xW)` 可理解为门，`xV` 是被门调制的内容分支。论文遵循 T5 的实现，省略 bias。

| 名称 | `φ(z)` | 形式 |
|---|---|---|
| GLU | `sigmoid(z)` | `sigmoid(xW) ⊙ xV` |
| Bilinear | `z` | `xW ⊙ xV` |
| ReGLU | `ReLU(z)` | `ReLU(xW) ⊙ xV` |
| GEGLU | `GELU(z)` | `GELU(xW) ⊙ xV` |
| SwiGLU | `Swish₁(z)` | `Swish₁(xW) ⊙ xV` |

普通 FFN 有两个权重矩阵，门控 FFN 有三个。为了尽量保持参数和计算量不变，作者把门控版的中间维度缩小为原来的 `2/3`：实验中普通 FFN 的 `d_ff = 3072`，门控 FFN 的 `d_ff = 2048`。忽略 bias 时，两者的主要矩阵参数都是 `2 × 768 × 3072 = 3 × 768 × 2048`。

## 实验设置

- 模型：T5 Base 规模的 encoder-decoder Transformer，encoder 和 decoder 各 12 层，`d_model = 768`，12 个 attention heads。
- 预训练：在 C4 上做 span-filling 去噪目标，训练 524,288 步；每批 128 个样本，输入 512 tokens，输出 114 tokens。
- 优化：Adafactor，反平方根学习率调度，最后 10% 步数线性衰减；预训练不用 dropout。
- 下游任务：将 SQuAD、GLUE 和 SuperGLUE 按样本数混合后联合微调 131,072 步，而不是每个任务分开微调。

## 主要结果

### 预训练

指标是 C4 留出集上的 log-perplexity，越低越好。65,536 步的数据来自 4 次运行，括号中为运行间波动；524,288 步列是完整预训练结果。

| FFN | 65,536 步 | 524,288 步 |
|---|---:|---:|
| ReLU（基线） | 1.997 (0.005) | 1.677 |
| GELU | 1.983 (0.005) | 1.679 |
| Swish | 1.994 (0.003) | 1.683 |
| GLU | 1.982 (0.006) | 1.663 |
| Bilinear | 1.960 (0.005) | 1.648 |
| GEGLU | **1.942 (0.004)** | **1.633** |
| SwiGLU | 1.944 (0.010) | 1.636 |
| ReGLU | 1.953 (0.003) | 1.645 |

GEGLU 在两个训练长度上都最好，SwiGLU 非常接近。更有说服力的对比是：完整预训练后，只换激活函数的 GELU/Swish 略差于 ReLU，而 GLU、Bilinear、GEGLU、SwiGLU 和 ReGLU 全部优于 ReLU。

### 下游任务

论文报告的是开发集上各任务的最优 checkpoint；数据有噪声，且每个完整预训练模型只做了一次混合微调。

| 指标 | ReLU 基线 | 最优变体 | 结果 |
|---|---:|---|---:|
| GLUE 平均分 | 83.80 | ReGLU | **84.67** |
| SuperGLUE 平均分 | 72.76 | SwiGLU | **74.56** |
| SQuAD EM | 83.18 | Bilinear | **83.82** |
| SQuAD F1 | 90.87 | ReGLU | **91.18** |

不同指标的赢家不同，所以论文不支持“某一个变体在所有场景都最优”。它支持的更稳健结论是：门控 FFN 这个家族值得取代普通 ReLU/GELU FFN。

## 如何解读

1. **结构可能比激活函数本身更重要。** GEGLU 与普通 GELU、SwiGLU 与普通 Swish 的差别是额外的线性分支和逐元乘法；实验改善主要出现在加入门控后。
2. **改善不是靠增加主要矩阵参数换来的。** 作者主动把中间维度缩到 `2/3`，让三矩阵门控版与两矩阵基线匹配。
3. **实用默认值是一个工程取舍。** 如果更看重本论文的预训练指标，GEGLU 和 SwiGLU 是最直接的候选；但下游结果没有单一胜者。
4. **机制并未被解释。** 作者在结论中明说没有给出这些架构为何有效的解释，因此“门产生了什么表示优势”仍是实验现象，不是论文已证明的机理。

## 局限

- 只在一种 T5 Base 规模的 encoder-decoder 架构、一个 C4 去噪预训练设置上验证，不能仅凭本文推广到所有模型规模和任务。
- 只有 65,536 步的短程实验做了 4 次运行以估计波动；完整预训练和下游微调没有同等的多种子统计。
- 下游表格选取每个任务的最优 checkpoint，又没有显著性检验，小差异不宜过度解读。
- 论文相比原始 T5 实验取消了预训练 dropout，因此它与原 T5 论文的跨论文差异不能只归因于 FFN；不过本文内各 FFN 变体使用相同配方，内部对比仍然有效。
- 作者用维度缩放匹配了理论参数量和主要计算量，但没有报告不同硬件上的墙钟时间、内存占用或 kernel 效率。

## 实践要点

在希望保持主要参数量和计算量的前提下，可将 `φ(xW₁)W₂` 改为 `(φ(xW) ⊙ xV)W₂`，并把门控版 `d_ff` 设为原来的约 `2/3`。但这个建议来自本文的 T5 Base 实验，具体模型仍应在自己的训练配方上做对照实验。

## 相关

- 这类 FFN 的权重切分背景：[[tensor-parallel]]
- 原始 PDF 与书目信息：[[2026-07-27-glu-variants-improve-transformer]]
