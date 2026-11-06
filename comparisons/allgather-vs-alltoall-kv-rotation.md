---
title: CP 中 KV 轮转：allgather vs all-to-all
type: comparison
created: 2026-07-25
updated: 2026-11-06
tags: [pytorch, context-parallel, 集合通信, 显存]
sources: ["[[2026-07-25-pytorch-cp-source]]"]
---

[[ring-attention]] 每步都要拿到别的 rank 的 KV 分片，PyTorch 提供两种实现，用**显存换通信次数**，默认是 allgather。

接口是 `_RingRotater`（`exchange_buffers` / `next_buffer`），`set_rotate_method("allgather" | "alltoall")` 切换。

## 对比

| | `_AllToAllRotater` | `_AllGatherRotater`（默认） |
|---|---|---|
| 通信次数 | 每步一次，共 `world_size - 1` 次 | **只有第一步一次** |
| 峰值显存 | 1 份 KV 分片 | **全量 KV** |
| 原语 | `permute_tensor(dsts=[1..n-1, 0])` | `all_gather_single(gather_dim=0)` |
| 与计算重叠 | 每步都能和 SDPA 重叠 | 只有首次能重叠 |

## 实现差异

**all-to-all** 是真正的环形轮转，每个 rank 把当前 buffer 发给下一个 rank：

```python
dsts = list(range(1, size)) + [0]
self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)
```

**allgather** 一次性把全量 KV 收齐，之后每步只是从本地缓存里切片：

```python
def next_buffer(self):
    idx = rank - self._idx
    return self._aggregated_buffer.chunk(world_size)[idx]
```

`idx` 可以是负数，靠 Python 的负索引绕回，效果等价于 `(rank - step) % world_size`。

注意 allgather 用的是 `gather_dim=0`——这里传的是已经 flatten 并拼接的 kv buffer，不是原始的 seq 维。

## 怎么选

**默认 allgather 是有道理的**：CP 本来就是为长序列设计的，此时激活值才是显存大头，KV 相对小；而 `world_size - 1` 次串行通信容易成为瓶颈。

**KV 大到装不下时换 all-to-all**：比如 KV head 数多、或 CP 与其他并行叠加导致单卡预算紧张。代价是通信次数线性增长，但每次都能和当步的 SDPA 计算重叠，实际开销未必线性上升。

需要注意 allgather 的显存优势会随 `world_size` 恶化：全量 KV 的大小是本地分片的 `world_size` 倍，CP 规模越大越吃亏。

## flex_attention 不走这条路

`_ContextParallel` 的 FLEX 分支直接在 forward pre-hook 里 `flex_cp_allgather(key, value, seq_dim, pg)` 拿全量 KV，Q 保持分片，根本没有轮转循环。掩码正确性靠 `_create_cp_block_mask` 改写 `mask_mod` 保证。

所以 `set_rotate_method` 只对 SDPA 路径有效。

## 顺序与内存统计的边界

[[2026-07-25-pytorch-cp-source]] 的因果分类使用 `(rank - i) % world_size` 标识第 `i` 轮 KV 来源。以 `world_size=4`、当前 rank 1 为例，四轮逻辑来源依次为 `1, 0, 3, 2`；两种 rotater 若用于同一轮次分类，取出的来源顺序应与该分类保持一致。allgather 缓存的负索引写法来自本笔记，原始摘录没有给出 rotater 实现。

表格中“一份分片”与“全量 KV”应理解为示意性的 KV 通信缓存规模比较：等长分片时，全量元素数是单个本地分片的 $W$ 倍。总峰值还受输入、拼接 buffer、输出、工作区、反向保存张量及生命周期影响，不能仅据此推断峰值显存比例。通信与计算实际重叠程度、默认配置的性能优势均需在目标设备上测量。
