---
title: ColwiseParallel vs RowwiseParallel
type: comparison
created: 2026-07-26
updated: 2026-11-06
tags: [pytorch, tensor-parallel, linear]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

在常见 MLP 布局中，先 colwise 后 rowwise 能让中间分片自然衔接；实际通信取决于输入、输出布局和算子，单独使用也可能适合具体模型。

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

## 一种常见的配对

```
x (Replicate)
  ──Colwise(w1)──►  Shard(-1)        无通信
  ──activation──►   Shard(-1)        逐元素，无通信
  ──Rowwise(w2)──►  Partial          无通信
  ──redistribute──► Replicate        ← 唯一一次 allreduce
```

Colwise 的输出布局 `Shard(-1)` 恰好等于 Rowwise 的 `desired_input_layouts`，衔接处 `redistribute` 是恒等操作。

若改变顺序，是否需要归约或 gather 取决于前后布局及目标输出，不能只根据策略顺序断定通信次数翻倍。

## 三个不对称的地方

**bias 的处理不同。** Colwise 把 bias 随输出维切分；Rowwise 的 bias 采用复制布局。具体算子须保证最终 bias 只贡献一次：如果各 rank 先加完整 bias 再求和，则会错误地得到多倍 bias。

**Embedding 的 desired input 相反。** Colwise embedding 要求输入 `Replicate()`（切的是 embedding 维，索引要完整）；Rowwise embedding 也要求 `Replicate()`（切的是词表维，每卡拿全部索引查自己那段）。**两者巧合地一致**，但原因不同。真正的差别是 Rowwise 的 `desired_input_layouts` 会随模块类型在 `_apply` 里动态改写，Colwise 是写死的。

**Rowwise embedding 用的不是普通 Partial。** 词表切分需要 mask 掉不在本地分片的索引，走 `_MaskPartial`（见 [[dtensor-placement]]）。

## 一个副作用陷阱

`RowwiseParallel._apply` 会写实例属性 `self.desired_input_layouts`。同一个 style 实例被复用到不同类型的模块上（比如 plan 里 Linear 和 Embedding 共用一个 `RowwiseParallel()` 对象），后一次 `_apply` 会覆盖前一次的设置。**plan 里每条目新建实例**，不要复用。

Colwise 没有这个问题，它的 `desired_input_layouts` 在 `__init__` 里定死。

## 补充：实现边界与核查

**配对与通信条件：**以 `2.14.0a0`、提交 `a07d9d50489` 为限，是否通信取决于已有输入布局及目标输出布局，而非策略名称本身。来源 `style.py:292-300` 仅明确 Rowwise 部分结果的两种归约去向。偏置解释也需谨慎：`style.py:250-279` 证明 Rowwise 偏置采用 `Replicate()`，却未展示算子如何避免重复累加。复制偏置不等于每卡先加偏置再求和；若如此仍会得到多倍偏置。因此，正确性机制及偏置参与计算的具体时机需继续核对算子分片规则。
