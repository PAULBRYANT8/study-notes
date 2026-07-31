# DeepSeek-V4 原始问答记录

本文只记录在对话中提出的 DeepSeek-V4 相关原始问题及其回答，不回填已有专题笔记。问题编号和块锚点一经创建便不再修改，方便长期引用。

## 问题索引

- [[#^q001-router|Q001：Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下]]

## Q001：Router 指的是什么

- 记录日期：2026-07-31
- 主题：MoE、Router、routed score、Top-K、共享专家与路由专家

> **原问题：** Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下 ^q001-router

### 回答

#### 1. 核心定义

Router 是 MoE（Mixture-of-Experts，混合专家）层中的一个可学习门控模块，也常被称为 gating network。它接收每个 token 在当前 Transformer 层中的隐藏向量，为所有路由专家计算匹配分数，然后选择其中若干专家处理该 token。

Router 本身：

- 不是某一个专家；
- 不负责完成专家 FFN 的主要计算；
- 不是模型外部的任务调度器；
- 通常只是一个参数量很小的线性映射，加上 Softmax、Top-K 和 dispatch 逻辑。

可以把它理解为 MoE 层内的“分诊台”：专家负责具体计算，Router 决定每个 token 应该送到哪些专家，以及这些专家的结果应占多大权重。

Router 通常会在每一个 MoE 层重新执行。某个 token 在不同层的隐藏状态不同，因此可能在不同层选择完全不同的专家。

#### 2. Router 的输入和输出

设第 \(l\) 层中，第 \(t\) 个 token 经过注意力模块后得到隐藏向量：

\[
u_t^{(l)} \in \mathbb{R}^{d}
\]

这里的 \(u_t^{(l)}\) 已经包含上下文信息，而不只是 token 的字面含义。例如，同一个 `bank` 在 `river bank` 和 `bank account` 中会产生不同的隐藏向量，因此可以被 Router 分配给不同专家。

Router 主要产生以下信息：

| 名称 | 含义 |
|---|---|
| Router logits | Softmax 前，token 与各专家的原始匹配值 |
| Routing scores | Softmax 或其他归一化之后，各专家的相对匹配分数 |
| Top-K indices | 最终被选中的 K 个路由专家编号 |
| Gate weights | 被选中专家在输出合并时使用的权重 |
| Dispatch metadata | token 如何复制、重排并发送给对应专家的信息 |

在很多实现中，Router 可以视为一个从隐藏维度 \(d\) 映射到路由专家数量 \(N\) 的线性层：

\[
z_t = W_{router}u_t
\]

其中：

\[
W_{router} \in \mathbb{R}^{N \times d}
\]

输出 \(z_t\) 中的第 \(i\) 个值就是 token 对路由专家 \(i\) 的原始匹配值。

#### 3. 从隐藏向量到 Top-K 专家的计算过程

DeepSeekMoE 一类常见路由机制可以写成以下形式。先计算 token 和每个专家路由向量之间的匹配值：

\[
z_{i,t} = \left(u_t^{(l)}\right)^T e_i^{(l)}
\]

其中 \(e_i^{(l)}\) 是当前层中专家 \(i\) 对应的可学习路由向量。然后在专家维度执行 Softmax：

\[
s_{i,t} = \operatorname{Softmax}_i(z_{i,t})
\]

Router 从所有 \(s_{i,t}\) 中选择最大的 K 个。稀疏门控权重为：

\[
g_{i,t} =
\begin{cases}
s_{i,t}, & i \in \operatorname{TopK}(s_{1,t},\ldots,s_{N,t}) \\
0, & \text{其他专家}
\end{cases}
\]

路由专家的加权输出可以抽象为：

\[
y_t = \sum_{i=1}^{N}g_{i,t}\operatorname{Expert}_i(u_t)
\]

所以 Router 实际完成了两件事：

1. 离散选择：用 Top-K 决定执行哪些专家；
2. 连续加权：用 routed score 决定选中专家结果的相对贡献。

不同实现可能会在 Top-K 之后重新归一化选中的分数，也可能采用 Sigmoid 等其他打分形式。因此上述公式用于说明通用机制，不等同于对 DeepSeek-V4 所有实现细节的断言。

#### 4. 一个数字示例

假设当前层有 4 个路由专家，某个 token 的归一化 routing scores 为：

\[
[0.58,\ 0.13,\ 0.03,\ 0.26]
\]

采用 Top-2 路由时，Router 会选择：

- 专家 1，score 为 0.58；
- 专家 4，score 为 0.26。

专家 2 和专家 3 不为这个 token 执行 FFN。抽象的合并结果为：

\[
y_t = 0.58E_1(u_t)+0.26E_4(u_t)
\]

如果实现要求对 Top-2 分数重新归一化，则会先将 0.58 和 0.26 除以二者之和，再作为最终 gate weights。

#### 5. DeepSeek-V4 当前代码语境中的完整路径

在本知识库记录的 DeepSeek-V4 `SwigluGroup` 接入路径中，Router 位于专家计算之前：

```text
token hidden states
  │
  ├─ Router：选择专家并产生 routed score
  │
  ├─ dispatch / token reorder
  │    将一个 token 为每个被选中专家生成对应 routed row，
  │    再按专家分组和重排
  │
  ├─ GMM-1：使用 w13 执行第一组 grouped matmul
  │
  ├─ clamp + SwiGLU + routed-score scaling
  │
  ├─ GMM-2：使用 w2 执行第二组 grouped matmul
  │
  └─ combine：把专家输出恢复并合并回原 token 顺序
```

Router 输出的专家编号决定 dispatch 如何复制和重排 token。经过 dispatch 后，一行数据不再只表示“原始 token”，而是表示一个具体的“token—专家”组合。与这些 routed rows 对齐的分数就是后续看到的 `routed_scores`。

当前接入路径中，原始计算会在 SwiGLU 后执行：

```python
h = h * routed_scores.to(h.dtype)
```

融合路径则把 routed score 作为 `SwigluGroup` 的可选 `weight`，将 SwiGLU 与逐行 score scaling 合并。这里的 `weight`：

- 是 Router 为 token—专家组合产生的动态分数；
- 不是模型线性层的权重；
- 不是专家的 `w1`、`w2` 或 `w3` 参数；
- 会随 token、上下文和 MoE 层发生变化。

之所以可以把 score scaling 放在第二次线性变换之前，是因为对每个 routed row 的标量 \(s\) 而言：

\[
W_2(s h)=s(W_2h)
\]

因此在没有其他破坏线性的操作时，先缩放激活再执行 `w2`，与在 `w2` 后缩放该专家输出是数学等价的。

#### 6. Router、dispatch、GMM 和 combine 的职责边界

这几个模块经常一起出现，但职责不同：

| 模块 | 主要职责 |
|---|---|
| Router | 计算专家分数并选择 Top-K 专家 |
| Dispatch | 根据选择结果复制、重排 token，并发送到对应专家 |
| GMM | 对按专家分组后的数据执行 grouped matrix multiplication |
| SwiGLU | 在专家 FFN 中执行门控激活 |
| routed-score scaling | 按 Router 分数缩放每个 token—专家分支 |
| Combine | 汇总多个专家分支并恢复原 token 顺序 |

Router 决定“去哪里”，dispatch 负责“怎么送过去”，专家和 GMM 负责“怎么算”，combine 负责“怎么把多路结果收回来”。

#### 7. 共享专家为什么不经过 Router

当前记录的 DeepSeek-V4 MoE 路径包含路由专家和共享专家：

- 路由专家只处理被 Router 选中的 token，因此需要 routed score、dispatch、专家分组和 combine；
- 共享专家由所有 token 固定经过，不参与 Top-K 竞争，所以没有 routed score，也不需要根据 Router 结果进行 GMM 分组。

共享专家通常负责承载各种 token 都可能需要的通用知识；路由专家则更适合学习差异化、专门化知识。把通用知识交给共享专家，可以减少多个路由专家重复保存同一类知识的现象。

#### 8. Router 是怎样学习出来的

Router 和其他模型参数一起进行端到端训练。开始训练时，它通常并不知道哪个专家适合代码、数学或自然语言。训练过程大致形成以下反馈：

1. Router 根据当前参数选择专家；
2. 被选中的专家处理对应 token；
3. 模型根据最终预测计算训练损失；
4. 反向传播更新专家参数和 Router 参数；
5. 能够降低损失的 token—专家组合逐渐得到更合适的分数。

专家分工一般是从数据中自然形成的，并非人工指定“专家 3 负责代码”。因此 routed score 应理解为：

> 在当前层和当前上下文中，某个 token 隐藏表示与某个专家之间的相对匹配程度。

它不一定是经过校准的概率，也不能直接解释成“专家有多少概率回答正确”。

Top-K 专家编号本身是离散选择；训练时通常通过被选中路径中的连续 gate weights，以及路由辅助损失，更新 Router 参数。

#### 9. 为什么需要负载均衡

如果只优化语言模型损失，Router 可能长期偏向少数暂时表现较好的专家，形成路由塌缩：

- 少数专家接收大量 token，成为计算瓶颈；
- 其他专家接收的 token 太少，无法充分训练；
- 模型虽然拥有很多专家，实际只使用其中少数；
- 在专家并行场景下，不同设备负载严重不均。

MoE 训练通常增加专家级或设备级负载均衡约束，鼓励 Router 更合理地利用专家。但约束也不能过强，否则 Router 会为了平均分配而放弃更适合的专家，从而损害模型效果。

#### 10. Router 与 Attention 的区别

二者都会计算分数，但选择对象不同：

| 模块 | 选择对象 | 主要问题 |
|---|---|---|
| Attention | 序列中的其他 token | 当前 token 应读取哪些上下文信息 |
| Router | 当前 MoE 层中的 FFN 专家 | 当前 token 应使用哪些专家参数处理 |
| Expert FFN | 不负责选择 | 对收到的 token 表示执行非线性变换 |

一句话区分：

> Attention 决定“看哪些 token”，Router 决定“让哪些专家处理这个 token”。

#### 11. 术语速查

| 术语 | 简要解释 |
|---|---|
| Router / gating network | 为 token 计算专家分数并选择专家的可学习模块 |
| Router logits | 归一化前的原始专家匹配值 |
| Routing score / routed score | token 被送往某个专家时携带的相对权重 |
| Top-K indices | 当前 token 选中的 K 个专家编号 |
| Gate weight | 合并专家结果时实际使用的权重 |
| Routed row | dispatch 后的一个 token—专家计算分支 |
| Dispatch | 将 token 按路由结果复制、重排并发给专家 |
| Combine | 将多个专家分支恢复并合并为 token 输出 |
| Shared expert | 所有 token 固定经过、不参与 Router Top-K 的专家 |
| Routed expert | 只有被 Router 选中时才执行的专家 |

### 相关笔记

- [[swiglu-group-接入复盘#1.3 路由专家与共享专家|路由专家与共享专家]]
- [[swiglu-group-接入复盘#4.1 两次 GMM 之间的融合点|两次 GMM 之间的融合点]]
- [[swiglu-group-接入复盘#A.3 weight 为什么是 routed score|weight 为什么是 routed score]]

[[#问题索引|返回问题索引]]
