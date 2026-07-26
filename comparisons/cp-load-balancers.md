---
title: CP 三种负载均衡策略对比
type: comparison
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 负载均衡]
sources: [[[2026-07-25-pytorch-cp-source]]]
---

单文档因果掩码用 [[head-tail-load-balancer]]（默认，不用配置）；多文档打包用 [[per-document-head-tail-load-balancer]]；掩码不是标准因果形状、且用 flex_attention，才需要 [[ptrr-load-balancer]]。

三者都实现 `_LoadBalancer._generate_indices(restore)`，返回 `(1, seq_len)` 或 `(B, seq_len)` 的重排索引，被 [[cp-sequence-sharding]] 统一消费。

## 对比

| | HeadTail | PerDocumentHeadTail | PTRR |
|---|---|---|---|
| 原理 | 解析式：头尾对称配对 | 解析式：每文档内配对 | 测量式：读 BlockMask 算负载后调度 |
| 适用掩码 | 单文档因果 | 文档块对角因果 | 任意 |
| 后端 | SDPA + flex | SDPA + flex | **仅 flex_attention** |
| 粒度 | token（`seq/(2·ws)`） | token（每文档独立） | 块（默认 128 token） |
| 整除要求 | `seq % (2·ws) == 0` | **每篇文档** `len % (2·ws) == 0` | Q 块数 `% ws == 0`，且 `q_blk == kv_blk` |
| 均衡质量 | 精确（因果下恰好互补） | 精确（每文档内） | 近似（贪心，docstring 示例 45/45/43/47） |
| 是否默认 | **是** | 否，须显式传入 | 否，须显式传入 |
| 索引 batch 维 | 恒为 1 | 可为 B | 可为 B |

## 选择顺序

1. **不传 load_balancer** → `_context_parallel_shard` 关闭均衡；旧版 `context_parallel` 则调 `_create_default_load_balancer` 自动上 HeadTail
2. 序列里打包了多篇文档 → PerDocumentHeadTail，否则跨文档配对失效
3. 掩码不是因果三角（滑窗、前缀 LM、自定义稀疏）→ 只有 PTRR 能处理，代价是必须迁到 flex_attention

## 几个容易踩的差异

**整除条件的严格程度差很多。** HeadTail 只看总长；PerDocument 要求每篇文档单独满足。打包的短文档很容易不足 `2 * world_size`，直接 `AssertionError`。

**PTRR 不看因果性。** 前两者在 `is_causal=False` 时会被 `_templated_ring_attention` 拒绝（"Load balancing requires `is_causal=True`"），但 PTRR 走的是 flex 路径，不经过这个检查。

**只有 HeadTail 会静默退化。** `_create_default_load_balancer` 在整除条件不满足时返回 `None`，不报错不警告，训练照常跑，只是 rank 之间失衡。另外两个是显式构造的，构造时就会抛。

**PTRR 的调度是贪心近似。** 前两者在标准因果掩码下是精确均衡（每 rank 1 的个数完全相等），PTRR 只做到接近。掩码越不规则差距越明显。
