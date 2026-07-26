---
title: ColwiseParallel vs RowwiseParallel
type: comparison
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, linear]
sources: [[[2026-07-26-pytorch-tp-source]]]
---

两者是一对：**先 colwise 后 rowwise**，中间那层零通信，整段只在末尾付一次归约。单独用任何一个都会在两端各产生一次通信，得不偿失。

详细各自见 [[colwise-parallel]] 和 [[rowwise-parallel]]。

## 对比

| | ColwiseParallel | RowwiseParallel |
|---|---|---|
| Linear weight | `Shard(0)` | `Shard(1)` |
| Linear bias | `Shard(0)`（跟着切） | **`Replicate()`** |
| Embedding weight | `Shard(1)` | `Shard(0)`（词表维） |
| 数学含义 | 切输出维 | 切输入维 |
| `input_layouts` 默认 | `Replicate()` | `Shard(-1)` |
| `desired_input_layouts` | **写死** `Replicate()` | Linear `Shard(-1)`；Embedding `Replicate()` |
| `output_layouts` 默认 | `Shard(-1)` | `Replicate()` |
| 输出的中间状态 | 直接就是 `Shard(-1)`，无需归约 | `Partial`，必须归约 |
| 归约通信 | 无 | allreduce 或 reduce_scatter |
| `use_local_output` | `True` | `True` |

## 为什么必须配对

```
x (Replicate)
  ──Colwise(w1)──►  Shard(-1)        无通信
  ──activation──►   Shard(-1)        逐元素，无通信
  ──Rowwise(w2)──►  Partial          无通信
  ──redistribute──► Replicate        ← 唯一一次 allreduce
```

Colwise 的输出布局 `Shard(-1)` 恰好等于 Rowwise 的 `desired_input_layouts`，衔接处 `redistribute` 是恒等操作。

反过来（先 rowwise 后 colwise）就要在中间归约一次、末尾再 gather 一次，通信翻倍。

## 三个不对称的地方

**bias 的处理完全相反。** Colwise 把 bias 一起切（形状是 `(out_features,)`，跟输出维走）；Rowwise 必须复制 bias，否则每卡各加一次，归约后变成 `N * b`。

**Embedding 的 desired input 相反。** Colwise embedding 要求输入 `Replicate()`（切的是 embedding 维，索引要完整）；Rowwise embedding 也要求 `Replicate()`（切的是词表维，每卡拿全部索引查自己那段）。**两者巧合地一致**，但原因不同。真正的差别是 Rowwise 的 `desired_input_layouts` 会随模块类型在 `_apply` 里动态改写，Colwise 是写死的。

**Rowwise embedding 用的不是普通 Partial。** 词表切分需要 mask 掉不在本地分片的索引，走 `_MaskPartial`（见 [[dtensor-placement]]）。

## 一个副作用陷阱

`RowwiseParallel._apply` 会写实例属性 `self.desired_input_layouts`。同一个 style 实例被复用到不同类型的模块上（比如 plan 里 Linear 和 Embedding 共用一个 `RowwiseParallel()` 对象），后一次 `_apply` 会覆盖前一次的设置。**plan 里每条目新建实例**，不要复用。

Colwise 没有这个问题，它的 `desired_input_layouts` 在 `__init__` 里定死。
