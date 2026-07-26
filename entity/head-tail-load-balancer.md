---
title: _HeadTailLoadBalancer
type: entity
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 负载均衡]
sources: [[[2026-07-25-pytorch-cp-source]]]
---

`_HeadTailLoadBalancer` 是 PyTorch CP 的默认负载均衡策略：把序列切成 `2 * world_size` 个等长块，让每个 rank 各拿一个靠前的块和一个对称靠后的块。

位置：`_load_balancer.py:87-173`。

## 问题

因果掩码下，Q 在序列越靠后需要算的 KV 越多。按原始顺序等分给 2 个 rank，掩码矩阵的 1 的个数是这样分的：

```
        [1, 0, 0, 0, 0, 0, 0, 0]
        [1, 1, 0, 0, 0, 0, 0, 0]
        [1, 1, 1, 0, 0, 0, 0, 0]   rank 0   ← 10 个 1
        [1, 1, 1, 1, 0, 0, 0, 0]
        ------------------------
        [1, 1, 1, 1, 1, 0, 0, 0]
        [1, 1, 1, 1, 1, 1, 0, 0]   rank 1   ← 26 个 1
        [1, 1, 1, 1, 1, 1, 1, 0]
        [1, 1, 1, 1, 1, 1, 1, 1]
```

rank 1 的计算量是 rank 0 的 2.6 倍，成为拖后腿的 straggler。

## 算法

配对第 `r` 块和第 `2*world_size - 1 - r` 块：

```python
chunk_size = seq_length // (world_size * 2)
indices = torch.arange(seq_length).view(world_size * 2, chunk_size)
head_idx = torch.arange(world_size)
tail_idx = 2 * world_size - 1 - head_idx
paired = torch.stack([indices[head_idx], indices[tail_idx]], dim=1)
all_indices = paired.reshape(-1)
```

`seq_length=8, world_size=2` 时得到 `[0, 1, 6, 7, 2, 3, 4, 5]`：rank 0 拿 `{0,1,6,7}`，rank 1 拿 `{2,3,4,5}`。重排后掩码矩阵每个 rank 各 18 个 1，完全均衡。

一个短的行长 + 一个长的行，互补成常数——这就是头尾配对能均衡的原因。

> docstring 里给的示例索引是 `[0, 7, 1, 6, 2, 5, 3, 4]`，和代码实际输出的块内顺序不同（示例是逐元素交错，代码保持每个块内部连续）。两者给各 rank 的**位置集合完全相同**，所以负载均衡效果一致，只是 shard 内部的排列不同。以代码为准。

## 约束

- `seq_length % (world_size * 2) == 0`，否则 `AssertionError`
- 返回形状固定为 `(1, seq_len)`——batch 内所有样本用同一套重排（因为只依赖序列长度）
- 只对因果掩码有意义。`_templated_ring_attention` 在 `is_causal=False` 且启用均衡时直接抛 `RuntimeError`

## 默认启用条件

```python
def _create_default_load_balancer(seq_length, world_size, device):
    if _cp_options.enable_load_balance and seq_length % (world_size * 2) == 0:
        return _HeadTailLoadBalancer(seq_length, world_size, device)
    else:
        return None
```

序列长度不满足整除时**静默退化为不均衡**，不报错也不警告。

## 相关

- 变体：[[per-document-head-tail-load-balancer]]
- 对比：[[cp-load-balancers]]
- 重排之后怎么用：[[ring-attention]]
