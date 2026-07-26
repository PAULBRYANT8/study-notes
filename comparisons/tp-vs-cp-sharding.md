---
title: TP 切分 vs CP 切分
type: comparison
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, context-parallel, 分布式]
sources:
  - [[2026-07-26-pytorch-tp-source]]
  - [[2026-07-25-pytorch-cp-source]]
---

TP 切**参数**，CP 切**激活的序列维**。两者正交，可以叠在同一个模型的不同 mesh 维上。切的东西不同导致实现风格也完全不同：TP 声明式、全程持有 DTensor；CP 命令式、切完立刻退回本地张量。

各自详见 [[tensor-parallel]] 和 [[context-parallel]]。

## 对比

| | TP | CP |
|---|---|---|
| 切什么 | 权重矩阵 | activation 的序列维 |
| 参数 | 分片（DTensor） | 完整复制 |
| 激活 | 按 placement 在算子间流转 | 全程沿 seq 维分片 |
| 切分粒度 | 模块（`nn.Linear` / `nn.Embedding`） | 张量 buffer |
| 谁决定分配 | `ParallelStyle` 声明布局 | load balancer 生成重排索引 |
| 应用方式 | `parallelize_module(m, mesh, plan)` 挂 hook | `_context_parallel_shard(mesh, buffers, dims)` 直接返回分片 |
| DTensor 的角色 | **全程持有**，靠 placement 传播推导通信 | **一次性工具**，`.to_local()` 后即弃 |
| 主要通信 | allreduce / reduce_scatter / all_gather（层边界） | KV 环形轮转或 allgather（attention 内部） |
| 负载均衡 | 天然均衡（矩阵等分） | **需要专门策略**，因果掩码下不均衡 |
| 省什么 | 参数显存 + 激活显存 | 激活显存（长序列） |
| 扩展瓶颈 | 通信在关键路径，通常不超过单机 8 卡 | 序列长度 |
| API 成熟度 | **公开稳定** | 私有 prototype（除 `context_parallel`） |

## 为什么 CP 需要负载均衡而 TP 不需要

TP 切的是权重矩阵，`(out, in)` 等分成 N 份，每份计算量严格相等。

CP 切的是序列，而因果掩码让序列位置的计算量**线性递增**——直接等分会让持有尾部的 rank 算 2.6 倍的量（8×8 掩码 2 卡）。所以 CP 必须先重排再切，见 [[cp-sequence-sharding]] 和 [[cp-load-balancers]]。

这是两者最本质的差异：**TP 的切分对象是齐次的，CP 的不是。**

## SequenceParallel 不是 CP

名字容易混淆。[[sequence-parallel]] 是 TP 的组成部分，只作用于 LayerNorm/Dropout 这类逐元素层，跨层边界要 all_gather 回 `Replicate` 给 Linear 用；attention 本身仍然是 TP 按 head 切的。

CP 则是让 attention 本身跨卡工作，靠 [[ring-attention]] 解决 Q 需要看到全部 KV 的问题。序列维的分片贯穿整个模型，不在层边界还原。

一个粗略的判断：**能不能处理单卡装不下的序列长度**——SequenceParallel 不能（attention 那步仍需完整序列），CP 能。

## 组合使用

典型的 4-D 并行是 `(dp, cp, tp, pp)`。TP 和 CP 用不同的 mesh 维：

```python
mesh = init_device_mesh("cuda", (dp, cp, tp), mesh_dim_names=("dp", "cp", "tp"))
parallelize_module(model, mesh["tp"], plan)      # TP 只吃 1-D mesh
_context_parallel_shard(mesh["cp"], buffers, dims)
```

`parallelize_module` 明确要求 1-D mesh，N-D 必须先切片。

叠加时同一个张量维度可能被两个 mesh 维切（比如 FSDP 和 TP 都切 dim 0），这时需要 `_StridedShard` 来描述非连续布局，见 [[dtensor-placement]]。

## 相同的地方

两者最终都落到 `distribute_tensor(t, mesh, [Shard(dim)])`。区别只在于 **dim 是权重维还是序列维**，以及切完之后还持不持有 DTensor。理解了 [[dtensor-placement]]，两套代码读起来都是同一件事的变体。
