---
title: SequenceParallel
type: entity
created: 2026-07-26
updated: 2026-11-06
tags: [pytorch, tensor-parallel, layernorm, 显存]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

`SequenceParallel` 处理 TP 覆盖不到的那些层（LayerNorm、Dropout、RMSNorm）：**参数复制，激活沿序列维切分**，让这些层的 activation 也不必每卡存一份完整的。

位置：`torch/distributed/tensor/parallel/style.py:339-439`。出自论文 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)。

## 解决什么问题

纯 TP 下，Linear 的 activation 是切开的，但 LayerNorm 前后需要 `Replicate`——于是每张卡都要存一份完整的 `(batch, seq, hidden)`。序列一长，这部分 activation 就成了显存大头，而且它是纯冗余的。

SequenceParallel 让这些层可以处理按序列维分片的输入。与 [[rowwise-parallel]] 的 `Shard(1)` 输出衔接时，可减少这些层中完整复制的 activation；其他模块或后续布局转换是否仍需要完整副本，取决于具体链路。

**通信与显存的权衡**：在常见环形集合通信模型和合适的布局衔接下，一次 all-reduce 可与 reduce-scatter 加 all-gather 作理论字节量比较，序列分片可减少部分层的复制激活。实际通信算法、额外布局转换和重叠情况不同，不能据此保证总通信量或耗时下降。

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

## 复制布局与复制操作

[[2026-07-26-pytorch-tp-source]] 中 `style.py:389-398` 使用 `DTensor.from_local(..., [Replicate()], run_check=False)` 包装已有本地参数；该片段本身没有展示跨 rank 同步。因此，自定义初始化或加载各 rank 不一致的检查点时，应先确保参数一致，再把它声明为复制布局。

例如两个 rank 的 norm 权重分别为全 1 和全 2，即使都被标成 `Replicate()`，本地前向结果仍可能不同。摘录没有证明这一错误必然被检测或必然静默；验证时应比较 rank 间参数和输出。
