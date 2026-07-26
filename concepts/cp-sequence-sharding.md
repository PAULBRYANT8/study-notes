---
title: CP 序列切分流程
type: concept
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 分片, dtensor]
sources: [[[2026-07-25-pytorch-cp-source]]]
---

CP 的切分是**「先重排、后等分」**两步：先按负载均衡策略把序列位置打乱成 rank-major 顺序，再用 `torch.chunk` 语义切成 world_size 段，第 r 段给 rank r。

真正干活的是 `_context_parallel_buffers()`（`_attention.py:1074-1181`），两个公开入口都收敛到它。

## 完整流程

```
buffers ──► load_balancer._generate_indices()   得到 rearrange_idx
        ──► index_select / gather 沿 seq 维重排
        ──► distribute_tensor(mesh, [Shard(seq_dim)]).to_local()
        ──► 每 rank 的本地分片
```

### 1. 生成重排索引

```python
load_balance_indices = load_balancer._generate_indices() if load_balancer else None
```

形状必须是 `(1, seq_len)`（batch 内策略一致）或 `(B, seq_len)`（batch 内不同）。`load_balancer is None` 表示不重排，直接等分。

### 2. 重排

按 `seq_dim` 分两条路径：

- **`seq_dim == 0`**（形如 `[seq_len]` 或 `[seq_len, ...]`）：`torch.index_select(buffer, 0, indices[0])`
- **`seq_dim > 0`**：把 `(B, seq_len)` 的索引 unsqueeze 到与 buffer 同维、expand 到同形，然后 `torch.gather(buffer, seq_dim, indices)`

索引 batch 数为 1 时会 expand 到数据的 batch 数；不为 1 且与数据 batch 数不等则报错。

### 3. 切分

```python
distribute_tensor(buffer, mesh, [Shard(seq_dim)], src_data_rank=None).to_local()
```

借 DTensor 的 `Shard` placement（见 [[dtensor-placement]]）完成切分后立刻 `to_local()` ——**CP 内部并不持有 DTensor**，只是拿它当一个切分工具。这和 [[tp-module-sharding]] 全程持有 DTensor、靠 placement 传播推导通信的风格正好相反。`src_data_rank=None` 表示各 rank 自己切自己的本地数据，不做广播。

### BlockMask 走另一条路

buffer 是 `BlockMask`（flex_attention 用）时不走上面的重排 + 切分，而是调 `_create_cp_block_mask()` 重新生成一个局部 BlockMask，`seq_dim` 参数被忽略。它内部改写 `mask_mod`，把局部 q 索引映射回全局 q 索引，这样局部 Q 能正确地对全局 KV 做掩码。

约束：`Q_LEN % (world_size * BLOCK_SIZE) == 0`，`BLOCK_SIZE` 默认 128。不满足会抛 `NotImplementedError`，因为局部 BlockMask 的 padding 还没支持。

## Sharding contract

这是整套设计的核心约定，写在 `_LoadBalancer` 的 docstring 里：

> 重排之后 CP 用 `Shard(seq_dim)` 切分，遵循 `torch.chunk` 语义——序列被切成 world_size 段**连续等长**的块，第 r 块归 rank r。因此 `rearrange_idx` 的布局必须让每个连续块自身是均衡的，即**按 rank 聚集（rank-major）**，不能按任何其他轴（比如 document）交错。

一句话：**切分逻辑是死的（永远连续等分），所有的灵活性都在重排索引里**。想改变哪些位置归哪个 rank，只能改 `_generate_indices()` 的输出顺序。

commit `53db8cca157` 修的就是违反这条约定的 bug：per-document 均衡器原本按 document-major 排布，导致切割线落在文档中间，文档长度不齐时负载失衡。

## 还原

`context_parallel_unshard()` 反向操作：

1. `all_gather_single(b, dim, mesh)` 沿 seq 维聚合
2. 用 `_generate_indices(restore=True)`（即 `argsort(rearrange_idx)`）做 `index_select` 还原原始顺序

`Q[rearrange_idx][restore_idx] == Q`。

## 坑：全局状态被隐式改写

`_context_parallel_shard()` 会**根据你有没有传 load_balancer 去改全局** `_cp_options.enable_load_balance`：

```python
if load_balancer is not None:
    _cp_options.enable_load_balance = True
else:
    _cp_options.enable_load_balance = False
```

而 `context_parallel_unshard()` 在你没传 load_balancer 时，会去读这个全局值来决定是否用默认的 head-tail 均衡器还原。

所以 shard 和 unshard 的 load_balancer 参数必须成对传，否则还原顺序会错，而且**不会报错，只会静默算错**。旧版 `context_parallel` context manager 有 `old_enable_load_balance` 保存/恢复，新版 `_context_parallel_shard` 没有——它只写不还原。

## 相关

- 重排索引怎么算：[[cp-load-balancers]]
- 切完之后 attention 怎么算：[[ring-attention]]
- 上层概念：[[context-parallel]]
