---
title: DTensor Placement（Shard / Replicate / Partial）
type: concept
created: 2026-07-26
updated: 2026-11-06
tags: [pytorch, dtensor, 分布式, 集合通信]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

Placement 描述一个全局张量在 DeviceMesh 的某一维上**怎么分布**。它是 [[tensor-parallel]] 和 [[context-parallel]] 共同的底座——两者都只是在选不同的 placement。

定义在 `torch/distributed/tensor/placement_types.py`，Python 层是 C++ 类型的薄封装。

## 三种基本 placement

| Placement | 含义 | 本地张量与全局的关系 |
|---|---|---|
| `Shard(dim)` | 沿张量第 `dim` 维切开 | 拼接（concat）得全局 |
| `Replicate()` | 每个 rank 持有完整副本 | 相等 |
| `Partial(reduce_op)` | 每个 rank 持有**待归约的部分值** | 归约（默认 sum）得全局 |

`Partial` 是最容易被忽略但最关键的一个：它表示"计算已经做完，但还没归约"。[[rowwise-parallel]] 的输出天然就是 `Partial`，归约推迟到真正需要值的时候。

## redistribute 与集合通信的映射

`DTensor.redistribute(placements=...)` 改变布局，底层自动选集合通信（`_redistribute.py:191-195`）：

| 转换 | 集合通信 |
|---|---|
| `Partial` → `Replicate` | `all_reduce` |
| `Partial` → `Shard` | `reduce_scatter` |
| `Shard` → `Replicate` | `all_gather` |
| `Shard(i)` → `Shard(j)` | all-to-all（本质是 gather + reshard） |
| `Replicate` → `Shard` | 本地切片，**无通信** |

**这张表是理解所有 TP 通信开销的钥匙。** 看一份 parallelize_plan 时，把每层的 output_layouts 和下一层的 desired_input_layouts 对上，不一致的地方需要查看具体转换；例如 Replicate 到 Shard 可以是本地切片，并非所有转换都发生集合通信。

Partial 到 Shard 的输出只保留本地分片，可配合 [[sequence-parallel]] 减少复制的激活；通信流量不能简单等同于输出大小，需考虑集合算法与统计口径。

## Partial 的线性性

```python
# reduce_ops that distribute over addition, enabling per-input linearity
# for bilinear ops like mm: reduce_op(A_i @ B) = reduce_op(A_i) @ B
LINEAR_REDUCE_OPS = ("sum", "avg")
ALL_REDUCE_OPS = ("sum", "avg", "min", "max", "product")
```

只有 `sum` 和 `avg` 对加法满足分配律，因此只有它们允许把归约**推迟到 matmul 之后**——这正是 rowwise 能"先算再 allreduce"的数学依据。`min`/`max`/`product` 可以 allreduce，但不能穿过 matmul。

源码注明 `Partial` 一般由算子产生，用户侧只能通过 `DTensor.from_local` 构造。

## 两个特殊变体

**`_MaskPartial`**（`placement_types.py:1868`）——为 rowwise 切分的 embedding 设计。词表按行切开后，每个 rank 只有一段词表，需要先把不属于本地分片的索引 mask 掉、把剩下的索引减去偏移，查完表再 allreduce。mask buffer 的生命周期跟随 DTensor。

**`_StridedShard`**（`placement_types.py:799`）——同一个张量维度被多个 mesh 维切分时（比如 FSDP 和 TP 叠加），描述非连续的切分布局。

## 与 CP 的联系

[[cp-sequence-sharding]] 里 CP 的整个切分动作就是一行 `distribute_tensor(buffer, mesh, [Shard(seq_dim)]).to_local()`——用完立刻退回普通张量。TP 则**全程持有 DTensor**，靠 placement 在算子间传播来自动推导通信。这是两者实现风格上最大的分歧。

## 补充：实现边界与核查

**布局变化与通信：**对 `2.14.0a0`、提交 `a07d9d50489`，本文表格自身已列出 `Replicate → Shard` 为本地切片，因此布局变化不必然产生集合通信。原始摘录 `style.py:292-300` 明确支持 Rowwise 输出转复制布局时用 allreduce、转分片布局时用 reduce_scatter，但不能据此断言后者通信量恒为前者的 $1/p$。通信量还须明确统计口径、算法和并行规模 $p$。`_redistribute.py:191-195` 仅给出位置锚点，未展示实现；具体映射条件、`_StridedShard` 行为及 mask 生命周期仍待核实。
