# DeepSeek-V4 原始问答记录

本文只记录在对话中提出的 DeepSeek-V4 相关原始问题及其回答，不回填已有专题笔记。问题编号和块锚点一经创建便不再修改，方便长期引用。

## 问题索引

- [[#^q001-router|Q001：Router 为每个 token 选择若干专家，并产生 routed score。这其中的 Router 指的是什么，请详细说明一下]]
- [[#^q002-shared-routed-experts|Q002：为什么要拆分成共享专家和路由专家？两条 FFN 路径分别起什么作用？]]
- [[#^q003-experts-zero-compile-config|Q003：使用 experts[0] 统一代表全局专家编译配置是否合理？]]
- [[#^q004-profile-routed-shared-experts|Q004：路由专家和共享专家在采集的 profiling 上怎么区分？它们有什么区别？]]

## Q001：Router 指的是什么

- 记录日期：2026-07-31
- 主题：MoE、Router、routed score、Top-K、共享专家与路由专家

> **原问题：** Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下
>
> ^q001-router

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

设第 $l$ 层中，第 $t$ 个 token 经过注意力模块后得到隐藏向量：

$$
u_t^{(l)} \in \mathbb{R}^{d}
$$

这里的 $u_t^{(l)}$ 已经包含上下文信息，而不只是 token 的字面含义。例如，同一个 `bank` 在 `river bank` 和 `bank account` 中会产生不同的隐藏向量，因此可以被 Router 分配给不同专家。

Router 主要产生以下信息：

| 名称 | 含义 |
|---|---|
| Router logits | Softmax 前，token 与各专家的原始匹配值 |
| Routing scores | Softmax 或其他归一化之后，各专家的相对匹配分数 |
| Top-K indices | 最终被选中的 K 个路由专家编号 |
| Gate weights | 被选中专家在输出合并时使用的权重 |
| Dispatch metadata | token 如何复制、重排并发送给对应专家的信息 |

在很多实现中，Router 可以视为一个从隐藏维度 $d$ 映射到路由专家数量 $N$ 的线性层：

$$
z_t = W_{router}u_t
$$

其中：

$$
W_{router} \in \mathbb{R}^{N \times d}
$$

输出 $z_t$ 中的第 $i$ 个值就是 token 对路由专家 $i$ 的原始匹配值。

#### 3. 从隐藏向量到 Top-K 专家的计算过程

DeepSeekMoE 一类常见路由机制可以写成以下形式。先计算 token 和每个专家路由向量之间的匹配值：

$$
z_{i,t} = \left(u_t^{(l)}\right)^T e_i^{(l)}
$$

其中 $e_i^{(l)}$ 是当前层中专家 $i$ 对应的可学习路由向量。然后在专家维度执行 Softmax：

$$
s_{i,t} = \operatorname{Softmax}_i(z_{i,t})
$$

Router 从所有 $s_{i,t}$ 中选择最大的 K 个。稀疏门控权重为：

$$
g_{i,t} =
\begin{cases}
s_{i,t}, & i \in \operatorname{TopK}(s_{1,t},\ldots,s_{N,t}) \\
0, & \text{其他专家}
\end{cases}
$$

路由专家的加权输出可以抽象为：

$$
y_t = \sum_{i=1}^{N}g_{i,t}\operatorname{Expert}_i(u_t)
$$

所以 Router 实际完成了两件事：

1. 离散选择：用 Top-K 决定执行哪些专家；
2. 连续加权：用 routed score 决定选中专家结果的相对贡献。

不同实现可能会在 Top-K 之后重新归一化选中的分数，也可能采用 Sigmoid 等其他打分形式。因此上述公式用于说明通用机制，不等同于对 DeepSeek-V4 所有实现细节的断言。

#### 4. 一个数字示例

假设当前层有 4 个路由专家，某个 token 的归一化 routing scores 为：

$$
[0.58,\ 0.13,\ 0.03,\ 0.26]
$$

采用 Top-2 路由时，Router 会选择：

- 专家 1，score 为 0.58；
- 专家 4，score 为 0.26。

专家 2 和专家 3 不为这个 token 执行 FFN。抽象的合并结果为：

$$
y_t = 0.58E_1(u_t)+0.26E_4(u_t)
$$

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

之所以可以把 score scaling 放在第二次线性变换之前，是因为对每个 routed row 的标量 $s$ 而言：

$$
W_2(s h)=s(W_2h)
$$

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

- [[swiglu-group-接入复盘#2. 背景：融合的到底是哪一段|路由专家与共享专家]]
- [[swiglu-group-接入复盘#3.2 npu_gmm：为路由专家提供 activation 注入点|两次 GMM 之间的融合点]]
- [[swiglu-group-接入复盘#4.1 Python 入口|weight 为什么是 routed score]]

[[#问题索引|返回问题索引]]

## Q003：使用 `experts[0]` 统一代表全局专家编译配置是否合理

- 记录日期：2026-08-04
- 主题：DeepSeek-V4、NpuGroupedExperts、torch.compile、hash routing、激活函数共享、编译配置

> **原问题：** 对于deepseek-v4模型，使用experts[0]统一代表全局专家编译配置是否合理？
>
> ^q003-experts-zero-compile-config

### 回答

#### 1. 结论

这条评审意见并不是说“代码已经确定编译错了”，而是在质疑一个没有被代码显式保证的前提：

> 所有 `NpuGroupedExperts` 是否真的都使用同一种 activation？如果不是，为什么可以拿 `experts[0]` 代表全部层？

当前 DeepSeek-V4 的前三层虽然使用 hash routing，但差异发生在“Router 怎么选专家”，不是“专家内部怎么执行 SwiGLU”。因此，**当前支持配置下共享一份 activation 编译结果是合理的**。

不过，当前代码没有检查“所有 activation 相同”这个前提。如果未来部分层使用原生 activation、部分层使用 `SwigluGroup`，代码就可能拿第一层的实现静默覆盖其他层。因此，这条评审属于合理的健壮性和可维护性意见。

#### 2. 先区分 MoE 层中的 Router 和 Experts

一层 MoE 可以先简化成两部分：

```text
MoE 层
├─ Router：决定每个 token 送给哪些专家
└─ Experts：对已经分发过来的 token 执行 FFN
```

完整数据流大致是：

```text
输入 token
  -> Router 选择专家
  -> dispatch/reorder
  -> GMM-1
  -> SwiGLU activation
  -> GMM-2
  -> combine
```

`compile_expert_activation()` 只编译中间这一小段：

```text
SwiGLU activation
```

它不编译：

- Router；
- hash routing table；
- token dispatch/reorder；
- GMM-1/GMM-2；
- expert 权重；
- 完整的 MoE 层。

这正是理解“前三层是 hash expert，为什么还可能共享编译结果”的关键。

#### 3. `experts[0]` 不是第 0 号专家

当前代码先遍历传入的模型对象：

```python
experts = [
    module
    for module in model.modules()
    if isinstance(module, NpuGroupedExperts)
]
```

这里的 `experts` 并不是某一层中的：

```text
expert 0、expert 1、expert 2……
```

列表中的每个元素是一个 `NpuGroupedExperts` 容器，通常对应一个 MoE 层中的整组本地路由专家。例如一个 6 层模型可能得到：

```text
experts[0] = 第 0 层的 NpuGroupedExperts 容器
experts[1] = 第 1 层的 NpuGroupedExperts 容器
experts[2] = 第 2 层的 NpuGroupedExperts 容器
experts[3] = 第 3 层的 NpuGroupedExperts 容器
experts[4] = 第 4 层的 NpuGroupedExperts 容器
experts[5] = 第 5 层的 NpuGroupedExperts 容器
```

每个容器内部的 `w13` 和 `w2` 才打包了这一层中的多个具体专家参数。

另外，“全局”在这里也不是跨所有机器、所有 PP stage 的真正全局；函数只处理传入的当前 `model` 对象，所以更准确地说，是在当前进程、当前模型分片可见的所有 `NpuGroupedExperts` 中共享。

#### 4. 当前代码如何使用 `experts[0]`

核心逻辑相当于：

```python
activation_fn = experts[0]._expert_activation_fn
compiled_activation = torch.compile(activation_fn)

for expert_module in experts:
    expert_module._expert_activation_fn = compiled_activation
```

也就是：

1. 从第一个路由专家容器取得 activation；
2. 只编译这一份；
3. 把编译结果写回全部路由专家容器。

调用关系可以画成：

```text
compile_config.backend ───────────────┐
parallel_dims.ep_enabled ─────────────┼─► 编译选项
experts[0]._expert_activation_fn ─────┘   待编译函数
                                          │
                                          ▼
                                  torch.compile 一次
                                          │
                                          ▼
                              写回全部 NpuGroupedExperts
```

因此，`experts[0]` 并没有代表完整的“全局编译配置”。

- `backend` 来自模型 compile 配置；
- `dynamic_tokens` 来自是否启用 Expert Parallel；
- `experts[0]` 只负责提供“要编译哪个 activation 函数”。

#### 5. hash 层与普通层到底哪里不一样

假设一层有 4 个路由专家。

普通 top-k Router 会根据当前 token 的隐藏向量计算分数：

```text
token A 的专家分数：

expert 0 = 0.1
expert 1 = 0.7
expert 2 = 0.3
expert 3 = 0.9
```

如果选择 top-2：

```text
token A -> expert 3、expert 1
```

hash Router 则根据 `input_ids` 查询 `tid2eid` 表：

```text
token_id = 123
tid2eid[123] = [expert 0, expert 2]

token A -> expert 0、expert 2
```

两种路由的差别是：

```text
普通层：根据 router score 的 top-k 选择专家
hash 层：根据 token ID 查表选择专家
```

但选择完成以后，专家内部仍可以执行完全相同的计算：

```text
hash Router ─────┐
                 ├─► routed_input/scores/counts
普通 top-k Router ┘           │
                              ▼
                  相同的 NpuGroupedExperts
                              │
                              ▼
                   GMM-1 -> activation -> GMM-2
```

所以“前三层 Router 是 hash 模式”并不能直接推出“前三层 expert activation 不同”。

#### 6. 一个当前安全的四层例子

假设 DeepSeek-V4 有 4 层，前三层是 hash routing：

| 层 | Router 类型 | `_expert_activation_fn` |
|---|---|---|
| 第 0 层 | hash | `_expert_activation` |
| 第 1 层 | hash | `_expert_activation` |
| 第 2 层 | hash | `_expert_activation` |
| 第 3 层 | top-k | `_expert_activation` |

虽然 Router 类型不同，但四层 activation 是同一个 Python 函数 `F`：

```text
第 0 层：F
第 1 层：F
第 2 层：F
第 3 层：F
```

这时没有必要执行四次：

```text
compile(F)
compile(F)
compile(F)
compile(F)
```

可以只编译一次：

```text
第 0 层 ─┐
第 1 层 ─┤
第 2 层 ─┼─► compiled(F)
第 3 层 ─┘
```

这就是当前使用 `experts[0]` 的优化目的：第一个容器只是提供大家共同使用的函数 `F`。

#### 7. 为什么每层输入不同仍然可以共享函数

不同层传给 activation 的数据当然不同：

```text
第 0 层：h0、limit0、scores0
第 3 层：h3、limit3、scores3
```

但 activation 的程序结构相同：

```python
def activation(h, swiglu_limit, routed_scores):
    ...
```

这些数据是在调用时传入的运行时参数，不是被写死在函数中的层级状态。类似于：

```python
def add(a, b):
    return a + b
```

`add(1, 2)` 和 `add(100, 200)` 的输入值不同，但不需要定义两个不同的 `add` 函数。

同样，hash 层和普通层的 routed rows、routed score 数值不同，不代表必须使用不同的 activation 程序。

需要注意：只调用一次 `torch.compile` 表示共享同一个编译包装器，不绝对保证底层永远只有一个图。如果输入 shape、可选参数或静态值触发不同 guard，编译器仍可能生成不同的特化版本。EP 场景下，当前 wrapper 会把 token 数所在的第 0 维标记为动态维度。

#### 8. A5 `SwigluGroup` 路径为什么也能共享

没有启用 `npu_swiglu_group` 时，所有 `NpuGroupedExperts` 默认使用：

```text
_expert_activation
```

A5 启用 `npu_swiglu_group` 后，converter 会遍历全部路由专家容器：

```python
for module in experts:
    module.set_expert_activation(swiglu_group_activation)
```

因此会变成：

| 层 | Router 类型 | `_expert_activation_fn` |
|---|---|---|
| 第 0 层 | hash | `swiglu_group_activation` |
| 第 1 层 | hash | `swiglu_group_activation` |
| 第 2 层 | hash | `swiglu_group_activation` |
| 第 3 层 | top-k | `swiglu_group_activation` |

当前 converter 不会产生“部分层 native、部分层 fused”的混合状态，所以从 `experts[0]` 取得 fused activation 后共享，在当前支持配置下也是合理的。

#### 9. 一个真正会出错的例子

假设未来支持按层选择 activation：

| 层 | 原本配置的 activation |
|---|---|
| 第 0 层 | `_expert_activation` |
| 第 1 层 | `_expert_activation` |
| 第 2 层 | `_expert_activation` |
| 第 3 层 | `swiglu_group_activation` |

这时：

```python
experts[0]._expert_activation_fn
```

取到的是 `_expert_activation`。当前代码会编译 native 实现，然后写回所有层：

```text
compiled_native = compile(_expert_activation)

第 0 层 -> compiled_native
第 1 层 -> compiled_native
第 2 层 -> compiled_native
第 3 层 -> compiled_native  # 错误
```

最终结果是：

| 层 | 原本应该使用 | 最终实际使用 |
|---|---|---|
| 第 0 层 | native | native |
| 第 1 层 | native | native |
| 第 2 层 | native | native |
| 第 3 层 | SwigluGroup | native，配置被静默覆盖 |

此时模型遍历顺序甚至会影响最终语义：如果 fused 层刚好排在 `experts[0]`，就可能反过来把所有 native 层覆盖成 fused。

这才是评审意见真正担心的问题。

#### 10. 当前代码缺少的显式保证

当前实现依赖这个不变量：

```text
所有 NpuGroupedExperts 的 activation 函数完全相同
```

但代码没有验证它，而且 `compile_key` 只有：

```python
(backend, dynamic_tokens)
```

没有包含 activation identity 或稳定的 `activation_kind`。

现有单元测试验证的是：

```text
两个容器都使用 F
  -> F 只编译一次
  -> 两个容器共享 compiled(F)
```

尚未覆盖：

```text
一个容器使用 F，另一个使用 G
  -> 应明确报错或分组编译
```

因此当前没有明确的 hash-layer 正确性 bug，但存在一个没有被代码表达出来的全局同质性假设。

#### 11. 为什么评审要求和 `panchao-gitcode` 确认

原来的 `compile_expert_activation()` 就采用“编译一份 activation，再分发给所有路由专家容器”的设计。当时待编译函数固定为 `_expert_activation`，不存在 native/fused 选择，所以全局同质性是天然成立的。

接入 `SwigluGroup` 后，待编译函数可能是：

```text
_expert_activation
或者
swiglu_group_activation
```

代码因此改为从 `experts[0]` 取得当前选择的函数。评审者是在确认：

> 原来“全模型共享一份 activation compile”的设计，在引入多个 activation 实现以后是否仍然成立？前三层 hash routing 是否会破坏这个前提？

从当前代码看，hash routing 不会破坏它；但最好由原设计者确认意图，并把不变量写进代码或测试，而不是只依赖大家对模型结构的理解。

#### 12. 推荐的处理方式

如果当前只支持“全模型统一 native”或“全模型统一 SwigluGroup”，最小修正是在编译前检查函数一致性：

```python
activation_fn = getattr(
    experts[0],
    _EXPERT_ACTIVATION_FN_ATTR,
)

if any(
    getattr(module, _EXPERT_ACTIVATION_FN_ATTR) is not activation_fn
    for module in experts[1:]
):
    raise ValueError(
        "All NpuGroupedExperts must use the same "
        "activation before shared compilation."
    )
```

这样：

```text
F、F、F、F -> 校验通过 -> 编译一次并共享
F、F、F、G -> 校验失败 -> 明确报错，不静默覆盖
```

同时应增加“不同 activation 明确报错”的单元测试。

如果未来确实需要不同层选择不同 activation，则不能继续简单使用 `experts[0]`，而应按下面的键分组，每组编译一次：

```text
(activation_kind, backend, dynamic_tokens)
```

#### 13. 最终判断

| 判断对象 | 结论 |
|---|---|
| 前三层 hash Router 与后续 top-k Router 不同 | 是，但差异在路由选择，不在 expert activation |
| 当前所有路由专家共享 native activation | 合理 |
| A5 下所有路由专家共享 `swiglu_group_activation` | 合理 |
| `experts[0]` 代表完整的全局编译配置 | 不准确，它只提供待编译 activation |
| 当前 hash 层会直接导致编译结果错误 | 没有代码证据支持 |
| 不校验 activation 一致性就覆盖全部模块 | 不够稳健 |
| 未来支持混合 activation 时仍只看 `experts[0]` | 不合理，应显式拒绝或分组编译 |

一句话概括：

> 当前可以共享的是“所有层共同使用、且不捕获层级状态的 activation 程序”，不是 `experts[0]` 这个模块本身。前三层的 hash 差异发生在 Router，所以当前共享合理；但代码应该显式验证所有 activation 相同，避免未来混合配置被第一层静默覆盖。

### 相关问题

- [[#^q001-router|Q001：Router 指的是什么]]
- [[#^q002-shared-routed-experts|Q002：为什么需要共享专家和路由专家]]

[[#问题索引|返回问题索引]]

## Q002：为什么需要共享专家和路由专家

- 记录日期：2026-07-31
- 主题：MoE、共享专家、路由专家、参数分工、稀疏计算

> **问题（精简）：** 为什么要拆分成共享专家和路由专家？共享专家已经计算了 `w1/w2/w3`，路由专家还要对选中的 token 计算 `w13/w2`，这两条路径分别起什么作用？
>
> ^q002-shared-routed-experts

### 回答

#### 1. 核心结论

共享专家和路由专家不是重复计算同一套参数，而是两组参数独立、输出相加的 FFN 分支：

$$
y(x)=F_{shared}(x)+\sum_{i\in\operatorname{TopK}(x)}g_i(x)F_i(x)
$$

- 共享专家提供所有 token 都需要的公共底座；
- 路由专家根据 token 和上下文提供专门化增量；
- Router 产生的 $g_i(x)$ 决定各路由专家对当前 token 的贡献。

因此整体思想是：

> MoE 输出 = 共享的基础变换 + 动态选择的专业变换。

#### 2. 共享专家算完 FFN，为什么还不够

共享专家计算的是自己的参数：

$$
F_{shared}(x)=W_{2,s}\left(\operatorname{SiLU}(xW_{1,s})\odot xW_{3,s}\right)
$$

路由专家计算的是另一套独立参数：

$$
F_i(x)=W_{2,i}\left(\operatorname{SiLU}(xW_{1,i})\odot xW_{3,i}\right)
$$

即使两条路径都有完整的 SwiGLU FFN，也有：

$$
W_{1,s}\neq W_{1,i},\qquad W_{2,s}\neq W_{2,i},\qquad W_{3,s}\neq W_{3,i}
$$

相同的算子结构不代表相同的知识。共享专家只完成了一套公共参数的变换，并没有执行所有路由专家学习到的专业能力。

路由路径中的 `w13` 通常只是：

$$
W_{13}=\operatorname{concat}(W_1,W_3)
$$

它将 `w1` 和 `w3` 合并成一次 GMM，再切分为 gate 和 up 两部分。这是参数布局和算子融合方式，不是与共享专家不同的神经网络原理。

#### 3. 两条路径如何配合

```text
                         ┌─► shared w1/w3 ─► SwiGLU ─► shared w2 ─┐
token hidden states ─────┤                                      ├─► 相加
                         └─► Router ─► dispatch                  │
                                      └─► routed w13             │
                                          └─► SwiGLU × score     │
                                              └─► routed w2 ─────┘
```

在当前 DeepSeek-V4 接入语境中：

1. 共享分支不经过 Router，所有 token 都执行同一套共享 FFN；
2. Router 为每个 token 选择 Top-K 路由专家并产生 routed score；
3. dispatch 将 token 复制、重排为对应的 token—专家 routed rows；
4. 路由专家通过 `w13 → SwiGLU × score → w2` 计算专业增量；
5. combine 恢复 token 顺序，再与共享专家输出相加。

这里并不是只有少数“特殊 token”才进入路由路径。通常每个 token 都会经过共享专家，同时还会选择若干路由专家；只是从某个路由专家的角度看，它只接收与自己匹配的 token 子集。

#### 4. 为什么两类专家都需要

| 方案 | 主要问题 |
|---|---|
| 只有共享专家 | 退化为稠密 FFN，所有知识都压缩进同一套参数，无法利用大型稀疏专家池 |
| 只有路由专家 | 每个专家都要重复学习语法、常见词义等公共知识，专家参数冗余，路由错误时也缺少稳定底座 |
| 共享 + 路由专家 | 共享专家学习公共能力，路由专家集中学习差异化能力，在激活少量参数的同时扩大模型总容量 |

这种拆分主要带来四个作用：

1. **减少知识冗余**：公共知识不必在多个路由专家中重复保存；
2. **提高专家专门化**：路由专家可以更多地学习代码、数学或特定语境中的增量特征；
3. **扩大条件容量**：模型可以拥有很多路由专家，但每个 token 只计算 Top-K 个；
4. **提高路由鲁棒性**：即使 Router 没选到最理想的专家，共享路径仍提供基础输出。

#### 5. 计算代价与取舍

共享专家不是免费的，它对所有 token 都产生固定计算量：

$$
\text{激活计算量}=\text{共享专家计算量}+\text{Top-K 路由专家计算量}
$$

为了控制总计算量，实际模型通常需要在共享专家数量、专家宽度和路由 Top-K 之间重新分配预算。共享专家过多会增加固定成本，并可能让模型过度依赖共享路径；共享专家过少，则公共知识仍会在路由专家中重复出现。因此两类专家的比例需要通过实验确定。

#### 6. 一句话总结

> 共享专家虽然完整计算了自己的一套 `w1/w3/w2`，但只负责公共基础能力；路由专家使用另一套独立参数，为每个 token 添加动态选择的专业增量。两者相加，才能同时获得稳定的通用能力、大规模稀疏参数容量和更强的专家专门化。

### 相关问题

- [[#^q001-router|Q001：Router 指的是什么]]

[[#问题索引|返回问题索引]]

## Q004：路由专家和共享专家在 profiling 上怎么区分

- 记录日期：2026-08-03
- 主题：DeepSeek-V4、profiling、路由专家、共享专家、GMM、SwigluGroup、Expert Parallel

> **原问题：** 路由专家和共享专家在采集的profiling上怎么区分呢？它们有什么区别？
>
> ^q004-profile-routed-shared-experts

### 回答

#### 1. 核心结论

在 profiling 中，最可靠的办法不是只看一个 `SwigluGroup` 内核名称，而是看它所在的**完整算子链、输入形状和 routed-score 参数**：

- 路由专家位于 `token permute/EP dispatch → 两次 Grouped MatMul → token unpermute/combine` 这条链路中；
- 共享专家位于普通的 `w1/w3 MatMul → activation → w2 MatMul` 稠密 FFN 链路中；
- 启用 `npu_swiglu_group` 后，两条路径都会出现同名的 `SwigluGroup`，但路由专家会传入 `[R, 1]` 的 routed-score `weight`，共享专家的 `weight` 为 `None`。

所以，应当用“上下文 + shape + 参数”联合判断，而不能把所有 `SwigluGroup` 都算作路由专家。

#### 2. 当前 DeepSeek-V4 前向算子链

设：

- (T)：当前 rank 上进入 MoE 层的 token 数；
- (K)：Router 的 Top-K；
- (R)：当前 rank 实际收到并处理的 routed rows 数；
- (H)：单个路由专家的中间维度；
- (H_s)：共享专家合并后的中间维度。

路由专家的典型 profiling 链路是：

```text
Router
  ↓
npu_moe_token_permute(tokens)
npu_moe_token_permute(routed_scores)
  ↓
[EP AllToAll + local re-routing]
  ↓
GroupedMatMul / aten::_grouped_mm       # w13，输入约 [R, D]
  ↓
SwigluGroup(weight=[R, 1])             # clamp + SwiGLU + score scaling
  ↓
GroupedMatMul / aten::_grouped_mm       # w2
  ↓
[local unpermute + EP AllToAll]
  ↓
npu_moe_token_unpermute / Top-K combine
```

共享专家的典型链路是：

```text
MatMul / Linear                         # w1，输入 [T, D]
MatMul / Linear                         # w3，输入 [T, D]
  ↓
Concat                                  # 得到 [T, 2H_s]
  ↓
SwigluGroup(weight=None)                # clamp + SwiGLU，无 routed score
  ↓
MatMul / Linear                         # w2
  ↓
与路由专家输出相加
```

算子名称可能随 CANN、torch_npu 和编译器版本变化，例如显示为 `GroupedMatmul`、`GroupedMatMul` 或关联到框架侧的 `aten::_grouped_mm`。应根据算子语义和前后关系识别，不依赖某一种精确拼写。

#### 3. 两条路径在 profiling 中的主要差异

| 观察项 | 路由专家 | 共享专家 |
|---|---|---|
| 处理对象 | 每个 token 的 Top-K 路由分支 | 当前 rank 上的全部 token |
| 激活输入行数 | (R)，随路由和 EP 收发变化 | (T)，通常相对固定 |
| 权重布局 | 多个专家打包的三维 `w13/w2` | 普通二维 `w1/w3/w2` |
| 矩阵乘法 | 两次 Grouped MatMul | 三次普通 MatMul/Linear，`w1` 和 `w3` 各一次 |
| routed score | 有；融合时作为 `weight=[R,1]` | 没有，`weight=None` |
| token 重排 | 有 permute、unpermute | 没有路由重排 |
| EP 通信 | 可能有 AllToAll 和本地 re-routing | 不参与路由产生的 EP AllToAll |
| 工作量 | 随各专家 token 分布动态变化 | 对全部 token 固定执行 |
| 性能特征 | 易受负载不均、短分组和通信影响 | shape 稳定，通常更接近普通稠密 GEMM |
| 对 Router 的梯度 | routed score 路径会把梯度传回 Router | 与 Router 无关 |

共享专家仍可能因为 Tensor Parallel 或其他并行策略出现通信算子；“没有通信”并不是它的定义。真正的区别是：共享专家不需要由路由产生的 token dispatch、EP AllToAll 和 combine。

#### 4. 为什么只看 `SwigluGroup` 名字区分不了

启用 `npu_swiglu_group` 后，两条路径最终调用的是同一个公共算子：

```python
torch.ops.cann_ops_nn.swiglu_group.default(...)
```

但调用参数不同：

```text
路由专家：x=[R, 2H]，weight=[R, 1]
共享专家：x=[T, 2H_s]，weight=None
```

路由专家的 `weight` 是 Router 为每个 token—专家分支产生的动态分数；共享专家对所有 token 固定执行，因此不需要该分数。

这也意味着：

- `api_statistic.csv` 如果只按 API 名聚合，通常不能可靠地把两个 `SwigluGroup` 拆开；
- `kernel_details.csv` 如果只有内核名、没有完整调用上下文，也可能无法可靠区分；
- `trace_view.json` 中的时间线、CPU API 与 NPU kernel 关联、输入 shape 更适合做归属判断。

#### 5. 没启用 `npu_swiglu_group` 时的区别

如果只启用了 `npu_moe_dispatch + npu_gmm`，两条路径的激活算子本身也不同：

- 路由专家默认使用 `torch_npu.npu_swiglu`，随后执行 routed-score 乘法；
- 共享专家默认使用拆分后的 `Silu + Mul` 小算子路径；
- 两者仍分别位于 Grouped MatMul 链和普通 MatMul 链中。

这种配置下，仅从激活算子名称通常也能初步区分；启用统一的 `SwigluGroup` 后，则必须再看 `weight` 和上下文。

#### 6. 实际查看 trace 的推荐步骤

1. 在 `trace_view.json` 中先定位 `npu_moe_token_permute`；
2. 沿时间线向后查看，找到相邻的两次 Grouped MatMul；
3. 两次 GMM 中间的激活属于路由专家；
4. 检查其输入行数是否为 (R)，以及是否存在 `[R,1]` 的 score/weight；
5. 再定位后面的 `w1/w3` 普通 MatMul、Concat、激活和 `w2` MatMul，这一段属于共享专家；
6. 在反向阶段，用 `GroupedMatMul` 梯度、路由重排反向和 EP 通信识别路由分支，用普通 MatMul 梯度识别共享分支。

当前 DeepSeek-V4 NPU forward 的 Python 调用顺序是先运行路由专家路径，再运行共享专家路径，最后相加。不过 NPU 异步执行、不同 stream 和 `torch.compile` 可能让设备时间线出现重叠或融合，所以调用先后只能作为辅助依据。

#### 7. 如果想让 profiling 一眼可辨

当前实现没有为两条路径设置独立 profiler range。最直接的改进是在 profiling 专用路径中增加两个明确范围：

```python
with torch.profiler.record_function("deepseek_v4.moe.routed_experts"):
    expert_out = _run_local_experts(...)

with torch.profiler.record_function("deepseek_v4.moe.shared_experts"):
    out = _shared_expert_output(...)
```

这样可以在 trace 中直接根据父级范围归属 NPU kernel。若启用了 `torch.compile`，需要先验证这些范围能否保留且不会造成额外 graph break；否则应在编译区域外增加标记，或使用当前编译栈支持的 profiler annotation。

在不修改代码的情况下，应至少打开：

```text
profile_record_shapes = True
profile_with_stack = True
```

其中 shape 是最关键的信息；stack 在编译模式下可能主要显示编译包装层，因此不能完全替代显式 range。

#### 8. 一句话总结

> 路由专家的 profiling 指纹是“动态 routed rows + token 重排/EP 通信 + 两次 GMM + 带 routed-score weight 的激活”；共享专家的指纹是“全部 token + 普通 w1/w3/w2 MatMul + 不带 weight 的激活”。两者可以使用同一个 `SwigluGroup` 内核，但调用语义并不相同。

### 相关问题

- [[#^q001-router|Q001：Router 指的是什么]]
- [[#^q002-shared-routed-experts|Q002：为什么需要共享专家和路由专家]]
- [[#^q003-experts-zero-compile-config|Q003：使用 experts[0] 统一代表全局专家编译配置是否合理]]

[[#问题索引|返回问题索引]]
