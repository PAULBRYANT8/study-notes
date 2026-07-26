---
title: SequenceParallel
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, layernorm, 显存]
sources: [[[2026-07-26-pytorch-tp-source]]]
---

`SequenceParallel` 处理 TP 覆盖不到的那些层（LayerNorm、Dropout、RMSNorm）：**参数复制，激活沿序列维切分**，让这些层的 activation 也不必每卡存一份完整的。

位置：`torch/distributed/tensor/parallel/style.py:339-439`。出自论文 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)。

## 解决什么问题

纯 TP 下，Linear 的 activation 是切开的，但 LayerNorm 前后需要 `Replicate`——于是每张卡都要存一份完整的 `(batch, seq, hidden)`。序列一长，这部分 activation 就成了显存大头，而且它是纯冗余的。

SequenceParallel 把这些层的输入也按序列维切开。配合 [[rowwise-parallel]] 输出改成 `Shard(1)`（reduce_scatter 而非 allreduce），整条链路上再没有完整复制的 activation。

**通信量不增反减**：原本 allreduce 一次，现在 reduce_scatter + all_gather 各一次，两者加起来的数据量和一次 allreduce 相同（allreduce 本身就是这两步实现的）。省下的是显存。

## 行为

| | 值 |
|---|---|
| `sequence_dim` | 默认 `1`，即 `(batch, seq, hidden)` |
| 参数 | `Replicate()` |
| 输入 | 重分布到 `Shard(sequence_dim)` |
| 输出 | `Shard(sequence_dim)` |
| `use_local_output` | **`False`**（与 Colwise/Rowwise 相反） |

输入处理分两种情况：传进来是 `DTensor` 且布局不对 → `redistribute`；传进来是普通 `Tensor` → **假定已经切好**，直接 `from_local` 打标签，不校验。

`use_local_output` 默认 `False` 是有道理的——SequenceParallel 一般夹在 TP 层中间，保持 DTensor 才能让下游继续自动推导布局。

## 参数复制用的是 from_local，不是 distribute_tensor

```python
# simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
# us to simply just use from_local
replicated_param = torch.nn.Parameter(
    DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
)
```

`from_local` 不做任何通信，纯粹给本地张量贴个"我是复制的"标签。这依赖一个假设：LayerNorm / RMSNorm 默认 `ones_` 初始化，各卡本来就相同。

**自定义初始化会静默破坏这个假设**——各卡权重不同却被声明为 Replicate，训练不报错但结果错。docstring 专门警告了这点，要求手动在 parallelize 前后广播权重。这是整个 TP 里最隐蔽的坑之一。

## 支持的模块

`nn.LayerNorm`、`nn.Dropout`，以及 RMSNorm 的 Python 实现。共同点是**逐元素或沿最后一维归约**，序列维上各位置独立，所以切开互不影响。

## 与 CP 的区别

两者都沿序列维切 activation，但目的完全不同：

- SequenceParallel 是 **TP 的补丁**，作用范围是 norm/dropout 这类层，跨层边界要 all_gather 回 `Replicate` 给 Linear 用
- [[context-parallel]] 是**独立的并行维度**，整个模型（包括 attention）全程保持序列切分，靠 ring attention 解决跨分片依赖

详见 [[tp-vs-cp-sharding]]。

## 相关

- 配合使用：[[rowwise-parallel]]（output_layouts 改 `Shard(1)`）、[[prepare-module-input-output]]
- 上层机制：[[tp-module-sharding]]
