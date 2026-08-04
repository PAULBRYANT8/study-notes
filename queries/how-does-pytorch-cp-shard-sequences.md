---
title: PyTorch 的 CP 是怎么切分序列的？
type: query
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 分片]
sources:
  - "[[2026-07-25-pytorch-cp-source]]"
  - 本地仓库 commit a07d9d50489 (2026-07-25), version.txt 2.14.0a0
---

**结论：切分本身是死的——永远按 `torch.chunk` 语义连续等分成 world_size 份，第 r 份给 rank r。所有灵活性都在切分之前的「重排」那一步。**

## 一句话流程

```
rearrange_idx = load_balancer._generate_indices()
buffer = gather/index_select(buffer, seq_dim, rearrange_idx)
local  = distribute_tensor(buffer, mesh, [Shard(seq_dim)]).to_local()
```

想改变哪些序列位置归哪个 rank，唯一的手段是改 `_generate_indices()` 返回的顺序。这条约定写在 `_LoadBalancer` docstring 的 "Sharding contract" 里，要求索引必须 **rank-major** 布局。详见 [[cp-sequence-sharding]]。

## 为什么要重排

因果掩码下越靠后的 Q 计算量越大，直接等分会让最后一个 rank 成为 straggler（8×8 掩码 2 卡时是 26 vs 10 个有效元素）。重排把序列头部和尾部配对，让每个连续分片的负载相当。三种策略见 [[cp-load-balancers]]。

## 分层回答

| 层 | 做什么 | 关键代码 |
|---|---|---|
| 入口 | `_context_parallel_shard` / `context_parallel` | `_attention.py:1433` / `:1523` |
| 切分 | 重排 + `Shard(seq_dim)` + `to_local()` | `_context_parallel_buffers`，`:1074-1181` |
| 均衡 | 生成重排索引 | `_load_balancer.py` |
| 计算 | ring attention 轮转 KV | `_templated_ring_attention`，`:317` |

DTensor 在这里只是**一次性的切分工具**：`distribute_tensor(...).to_local()` 切完立刻取回本地张量，CP 运行时并不持有 DTensor。[[tensor-parallel]] 的做法相反，对照见 [[tp-vs-cp-sharding]]。

## 值得记住的三个坑

1. **shard 和 unshard 的 `load_balancer` 必须成对传。** `_context_parallel_shard` 会隐式改写全局 `_cp_options.enable_load_balance`，`context_parallel_unshard` 在没传参时又去读它。传错不报错，静默算错。
2. **序列长度不满足 `% (2 * world_size) == 0` 时，默认均衡器静默退化为不均衡**，无警告。
3. **`BlockMask` 不走这条路**，`seq_dim` 参数被忽略，改由 `_create_cp_block_mask` 重新生成局部掩码。

## 状态

除 `context_parallel` 这个 context manager 外，全部 API 都是 `_` 前缀的私有 prototype，包括新版的 `_context_parallel_shard`。测试和实际用法都在用私有 API，说明公开接口还没定型。

`torch/distributed/tensor/experimental/_attention.py` 已退化成 44 行的 BC stub，实体全部迁到 `_context_parallel/` 子包。TODO 里写着待最终接口确定后再加 deprecation。

## 相关

[[context-parallel]] · [[cp-sequence-sharding]] · [[ring-attention]] · [[cp-load-balancers]] · [[allgather-vs-alltoall-kv-rotation]]
