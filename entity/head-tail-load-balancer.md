---
title: _HeadTailLoadBalancer
type: entity
created: 2026-07-25
updated: 2026-11-06
tags: [pytorch, context-parallel, 负载均衡]
sources: ["[[2026-07-25-pytorch-cp-source]]"]
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

> 笔记记录的 docstring 示例索引 `[0, 7, 1, 6, 2, 5, 3, 4]` 与摘录中的代码输出顺序不同。以代码为准；此处的 docstring 示例未包含在原始源码摘录中，需回到对应提交核实。

**顺序影响正确性（依据源码摘录及本笔记中的示例推导）**：序列长 8、两个 rank 时，rank 0 的代码输出是 `[0, 1, 6, 7]`。若改成相同位置集合但顺序不同的 `[0, 7, 1, 6]`，仍对本地 Q/K 使用普通下三角掩码，原位置 7 的 Q 只能看到局部前两项 0 和 7，却遗漏本应可见的位置 1 和 6。位置集合和可见对数统计相同，并不说明它们可沿用同一种局部掩码和半块处理逻辑直接互换。

**均衡的量（数学推导）**：令序列长度为 $L$、rank 数为 $W$，小块长度 $c=L/(2W)$。在标准单文档因果掩码下，每个头位置和对称尾位置的可见 K 数之和为 $L+1$，每个 rank 的可见 QK 对数为 $c(L+1)=L(L+1)/(2W)$。例如 $L=8$、$W=2$ 时各为 18；这不代表 kernel 或端到端耗时一定相等。

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
