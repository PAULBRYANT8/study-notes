---
title: Ring Attention（PyTorch CP 实现）
type: concept
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, attention, sdpa]
sources: [[[2026-07-25-pytorch-cp-source]]]
---

Ring attention 是 CP 下计算 attention 的方法：Q 固定在本地，KV 分片沿环形拓扑轮转 world_size 步，每步算一次局部 SDPA，用 logsumexp 增量合并成全局结果。

实现在 `_templated_ring_attention()`（`_attention.py:317-488`），"templated" 指它接收一个 `op` 参数，flash / efficient / cudnn 三种 SDPA 后端共用同一套循环。

## 主循环

```python
rotater = _create_rotater(group, 2)

for i in range(size):
    if i > 0:
        next_kv = rotater.next_buffer()          # 收上一步转来的 kv
        key   = next_kv[: key.numel()].reshape(key.shape)
        value = next_kv[key.numel() :].reshape(value.shape)

    if i < (size - 1):
        next_kv = torch.cat([key.flatten(), value.flatten()])
        rotater.exchange_buffers(next_kv)        # 转给下一个 rank

    ...
    out, logsumexp, *rest = op(q, k, v, is_causal=..., **kwargs)
    sdpa_merger.step(out, logsumexp, partial)
```

k 和 v 拼成一个扁平 buffer 一起通信，减少一半的集合通信调用次数。

`key`/`value` 在循环前强制 `.contiguous()`，源码 TODO 注明：不做的话 loss 曲线会坏掉，原因未查明（SDPA 本身没这个要求）。

## 因果掩码下的三种块行为

`_is_causal_behavior()` 决定第 i 步该怎么算：

| 条件 | 行为 |
|---|---|
| `is_causal == False` | `NOT_IS_CAUSAL`，全算 |
| `i == 0` | `IS_CAUSAL`，本地 QKV，带因果掩码 |
| `(rank - i) % world_size < rank` 或启用负载均衡 | `NOT_IS_CAUSAL`，全算 |
| 其余 | `SKIP`，整块被掩掉，直接 `continue` |

**未启用负载均衡时**，`SKIP` 是那个跳过的分支：rank 0 只在第一步有活干，后面全 skip；rank N-1 每步都要算。这正是不均衡的来源。

**启用负载均衡后永不 SKIP**——因为每个 rank 同时持有序列的头部和尾部块，任何一个 KV 分片都至少有一半 Q 需要它。

## 负载均衡下的分块计算

启用 head-tail 均衡后，本地 q/k/v 各自包含**两个不连续的块**（头块 + 尾块），拼在一起以便单次 SDPA 调用。循环里据此三分支：

```python
if i == 0 or (not enable_load_balance or not is_causal):
    q, k, v, partial = query, key, value, False        # 全量本地
elif i <= rank:
    ROUND_ROBIN_CYCLE = 2
    q, k, v, partial = query, key.chunk(2, dim=2)[0], value.chunk(2, dim=2)[0], False
else:
    q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True
```

- `i <= rank`：转来的 KV 只有前半块（头块）被需要，后半块（尾块）在因果序上靠后，丢弃
- `i > rank`：本地 Q 只有后半块（尾块）需要这批 KV，前半块用不上；因为只更新了输出的一部分，`partial=True`

`partial` 传给 `_SDPAMerger`，它用 `_partial_update()` 只把结果写回输出张量对应的那半块。

源码里 `Note [Context parallelism load balance algorithm for causal masking]` 用 seq_len=4 / 2 ranks 的例子逐步推演了这段，是理解这部分最快的入口。

## Softmax 合并

`_SDPAMerger` 做的是标准的在线 softmax 合并：每步拿到局部 `out` 和 `logsumexp`，按 logsumexp 加权累积。`_cp_options.convert_to_f32` 控制是否升到 fp32 累加，默认 True。

## 相关

- KV 怎么转：[[allgather-vs-alltoall-kv-rotation]]
- 为什么需要头尾配对：[[head-tail-load-balancer]]
- 输入怎么切：[[cp-sequence-sharding]]
- flex_attention 走的是另一条路（allgather 全量 KV，不轮转），见 [[context-parallel]]
