---
title: loss_parallel
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, 损失函数, 显存]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

`loss_parallel()` 是个 context manager，让 cross entropy 在 **logits 按词表维分片**的状态下直接计算，不必先 all_gather 成完整 logits。

位置：`torch/distributed/tensor/parallel/loss.py:31-`，公开 API。

## 解决什么问题

LLM 的输出层是 `(batch, seq, vocab)`，vocab 通常几万到几十万。这个张量往往是整个前向里最大的一个。TP 下它天然按 vocab 维分片（输出层用 [[colwise-parallel]]），但标准的 `cross_entropy` 需要完整 logits——一次 all_gather 就把刚省下的显存又吐回去了。

loss_parallel 重写了 cross entropy 的分布式实现：softmax 的分母只需要各分片 max 和 sum 的 allreduce（两个标量级别的量），完全不用聚合 logits 本身。

## 用法

```python
with loss_parallel():
    loss = F.cross_entropy(logits, target)
    loss.backward()
```

**backward 也必须在 context 内**，docstring 明确要求。

## 输入约定

- `input`：DTensor，在**恰好一个** mesh 维上按 class 维分片
- `target`：类别索引（不支持类别概率）。布局由 input 推导——TP 那维的 `Shard(class_dim)` 换成 `Replicate()`，其余 `Shard(d)` 中 `d > class_dim` 的下移一位（因为输出少了 class 这一维）
- `weight`：如果给，假定已复制
- `label_smoothing`：**不支持**

N-D mesh 现在是支持的：切 class 维的那个 mesh 维被当作 "TP" 维，**可以在任意位置**，不必是最后一维。其余 mesh 维可以是 `Shard`（DP/CP 的 batch 切分）或 `Replicate`。

普通 `torch.Tensor` 的 target 只在推导出的布局全是 `Replicate` 时才接受（即 1-D mesh），否则必须显式传 DTensor，避免歧义。

## 返回值布局随 reduction 变化

| `reduction` | 返回 |
|---|---|
| `"none"` | 逐样本 loss，继承 target 的布局 |
| `"sum"` | 标量；TP 维 `Replicate`，其余 `Shard` 维为 `Partial("sum")`（跨 rank 归约推迟到物化时），`Replicate` 维保持 `Replicate` |
| `"mean"` | **仅支持 1-D mesh**，返回完全复制的 DTensor |

1-D mesh 下三种情况都简化为完全复制的 DTensor，也就是这个 API 最初的行为。

`"sum"` 返回 `Partial` 而不是立即 allreduce，是把归约交给 [[dtensor-placement]] 的惰性机制——真正需要值时才通信。

## 相关

- 上游：[[colwise-parallel]]（输出层按 vocab 切）
- 概念：[[tensor-parallel]]
