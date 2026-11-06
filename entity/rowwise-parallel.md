---
title: RowwiseParallel
type: entity
created: 2026-07-26
updated: 2026-11-06
tags: [pytorch, tensor-parallel, linear, embedding]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
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

**bias 的复制与计算顺序需要区分。** 摘录中的 Rowwise Linear 将 bias 标为 `Replicate()`。数学上 `y = sum_i (x_i @ W_i^T) + b`，最终 bias 必须只贡献一次；如果各 rank 先加完整 bias 再对部分结果求和，就会错误地得到 `N * b`。如何实现正确的 bias 语义，仍需核对对应版本 DTensor 算子规则，不能仅由 placement 推断具体顺序。

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

- `output_layouts=Replicate()`（默认）→ `all_reduce`，结果在各 rank 复制
- `output_layouts=Shard(1)` → `reduce_scatter`，结果按指定维度分片

后者可配合 [[sequence-parallel]]：若下一层沿序列维切分，可考虑直接 reduce-scatter 到对应布局。具体通信量和耗时取决于实现、拓扑与统计口径。

## Embedding 的特殊性

rowwise 切 embedding 是按**词表维**切：每个 rank 只有一段词表。查表时不属于本地分片的索引必须被 mask 掉，输出置零，再 allreduce 拼回来。这个逻辑封装在 `_MaskPartial` 里（见 [[dtensor-placement]]），不是普通的 `Partial`。

## 约束

- 只支持 `nn.Linear` 和 `nn.Embedding`
- `_apply` 会**写实例属性** `self.desired_input_layouts`——同一个 style 实例复用到不同类型的模块上会互相干扰。plan 里每个条目建议新建实例

## 相关

- 配对使用：[[colwise-parallel]]，对比见 [[colwise-vs-rowwise-parallel]]
- 上层机制：[[tp-module-sharding]]

## 参数分片与通信量的边界

源码摘录 `style.py:250-279` 对 Linear 的权重标记 `Shard(1)`，bias 标记 `Replicate()`；`style.py:292-300` 根据目标布局触发重分布。数学上需要保证偏置对最终结果只贡献一次，但摘录没有展开 DTensor 内部的加法规则，不能仅凭参数 placement 推断具体是在归约前后由哪一步实现。

`Partial → Replicate` 与 `Partial → Shard` 的目标输出布局不同。输出分片变为原来的 $1/W$，不等于跨设备传输字节数必为 all-reduce 的 $1/W$；后者依赖通信算法、拓扑、数据类型和统计口径，应在目标环境测量。
