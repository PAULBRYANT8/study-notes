---
title: _PTRRLoadBalancer
type: entity
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 负载均衡, flex-attention, 调度]
sources: ["[[2026-07-25-pytorch-cp-source]]"]
---

`_PTRRLoadBalancer`（Processing-Time based Round-Robin）不假设掩码形状，而是**从 `BlockMask` 里读出每个 Q 块的实际计算量**，当作调度问题求解。只能配 `flex_attention()` 使用。

位置：`_load_balancer.py:300-472`。

## 和头尾配对的本质区别

[[head-tail-load-balancer]] 和 [[per-document-head-tail-load-balancer]] 都是**解析式**的——预设掩码是因果三角形，据此推出对称配对能均衡。掩码换个形状（滑窗、前缀 LM、任意稀疏）就不成立了。

PTRR 是**测量式**的：把每个 Q 块要算的 KV 块数当作任务的处理时间，然后做一次贪心调度，不关心掩码为什么长这样。

## 计算量从哪来

```python
non_sparse_kv_num_blocks = kv_num_blocks + full_kv_num_blocks
```

`BlockMask` 里 `kv_num_blocks` 是部分掩码的块数、`full_kv_num_blocks` 是全满的块数，相加就是该 Q 块行需要计算的 KV 块总数。形状 `(B, H, Q_BLK)`，要求 `H == 1`（各 head 掩码一致）。

## PTRR 调度

`ptrr_scheduling(process_time, group_size)` 是个静态方法，本质是**蛇形（boustrophedon）分配**：

```python
_, sorted_desc = torch.sort(process_time, descending=True, stable=True)
sorted_desc_reversed = torch.flip(sorted_desc.view(-1, group_size), dims=[1]).view(-1)
tasks_in_group = torch.where(
    torch.arange(num_tasks) // group_size % 2 == 0,
    sorted_desc,               # 偶数轮：正序
    sorted_desc_reversed,      # 奇数轮：逆序
)
tasks_in_group = tasks_in_group.view(-1, group_size).transpose(0, 1)
```

按处理时间降序排好后，每 `group_size` 个任务为一轮，偶数轮正着发、奇数轮倒着发。这样上一轮拿到最重任务的组，下一轮拿最轻的，长期趋于平衡。这是经典的贪心近似解，不保证最优。

docstring 里的例子：16 个任务分 4 组，各组总时长 45 / 45 / 43 / 47，接近但不完全相等。

最后每组内部再 `torch.sort` 一次——源码注明这步对正确性和性能都没影响，纯粹为了让用户看掩码时顺眼。

## 块索引展开成 token 索引

调度出来的是块编号，要展开成序列位置：

```python
indices = torch.arange(q_blk_size * ptrr_indices.size(1)).view(-1, q_blk_size)
indices = indices[ptrr_indices].view(B, -1)
```

batch 维用 `torch.vmap` 并行处理。

## 约束

- **仅 flex_attention**，因为要 `BlockMask`
- `num_tasks % group_size == 0`，即 Q 块数必须被 world_size 整除，否则 `NotImplementedError`
- `q_blk_size == kv_blk_size`，否则 `AssertionError`（源码注明 "for now only support"）
- `BlockMask` 形状须为 `(B, 1, seq_len, seq_len)` 或 `(1, 1, seq_len, seq_len)`
- 粒度是块（默认 128 token），比头尾配对粗

## 相关

- 对比：[[cp-load-balancers]]
- 上游概念：[[context-parallel]]
