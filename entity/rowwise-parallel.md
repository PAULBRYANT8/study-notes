---
title: RowwiseParallel
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, linear, embedding]
sources: [[[2026-07-26-pytorch-tp-source]]]
---

`RowwiseParallel` 把权重按**输入维度**（行）切开，输入要求按最后一维分片，输出是 `Partial`，归约后才是正确结果。

位置：`torch/distributed/tensor/parallel/style.py:186-336`。

## 切法

| 模块 | 参数 | placement |
|---|---|---|
| `nn.Linear` | weight | `Shard(1)` |
| `nn.Linear` | bias | **`Replicate()`** |
| `nn.Embedding` | weight | `Shard(0)` |

同样是存储视角与数学视角的错位：`weight` 存储形状 `(out, in)`，切 dim 1 就是切 `in`，数学上 `weight^T` 变成 `Shard(0)`，即按行切。

**bias 必须复制而不能切分。** 数学上 `y = sum_i (x_i @ W_i^T) + b`：每卡算出的是部分和，如果每卡都加一次 bias，归约后会变成 `N * b`。所以 bias 只能在归约完成之后加一次。（DTensor 的算子规则负责保证这个顺序——`Partial + Replicate` 不满足线性性，会先物化再相加。这一步是从 [[dtensor-placement]] 的 `LINEAR_REDUCE_OPS` 语义推出来的，未逐行核对算子规则实现。）

## 布局

| | 默认值 | 说明 |
|---|---|---|
| `input_layouts` | `Shard(-1)` | 声明进来的张量已按最后一维切好 |
| `desired_input_layouts` | Linear → `Shard(-1)`；Embedding → `Replicate()` | 在 `_apply` 里按模块类型动态设置 |
| `output_layouts` | `Replicate()` | 触发 all_reduce |
| `use_local_output` | `True` | |

`desired_input_layouts` 是唯一一个随模块类型变化的——这是它和 [[colwise-parallel]] 结构上最大的不同。Embedding 需要完整的索引张量（每个 rank 都要拿全部索引去查自己那段词表），Linear 需要分片的激活。

## 输出归约由 output_layouts 决定

```python
# Rowwise sharding produces partial output, depending on output layouts:
# 1. to replicate -> allreduce
# 2. to shard -> reduce_scatter
```

这是**调 TP 通信量最重要的一个旋钮**：

- `output_layouts=Replicate()`（默认）→ `all_reduce`，通信量 = 完整张量
- `output_layouts=Shard(1)` → `reduce_scatter`，通信量 = 完整张量 / world_size

后者正是 [[sequence-parallel]] 的做法：既然下一层 LayerNorm 反正要按序列维切，不如直接 reduce_scatter 到位，省掉 `(world_size-1)/world_size` 的通信量。

## Embedding 的特殊性

rowwise 切 embedding 是按**词表维**切：每个 rank 只有一段词表。查表时不属于本地分片的索引必须被 mask 掉，输出置零，再 allreduce 拼回来。这个逻辑封装在 `_MaskPartial` 里（见 [[dtensor-placement]]），不是普通的 `Partial`。

## 约束

- 只支持 `nn.Linear` 和 `nn.Embedding`
- `_apply` 会**写实例属性** `self.desired_input_layouts`——同一个 style 实例复用到不同类型的模块上会互相干扰。plan 里每个条目建议新建实例

## 相关

- 配对使用：[[colwise-parallel]]，对比见 [[colwise-vs-rowwise-parallel]]
- 上层机制：[[tp-module-sharding]]
