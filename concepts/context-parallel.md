---
title: Context Parallel (CP)
type: concept
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, 分布式, 长序列, attention]
sources: [[[2026-07-25-pytorch-cp-source]]]
---

Context Parallel（上下文并行，CP）是**沿序列维度**切分张量的并行方式：每个 rank 只持有序列的一段，attention 通过 rank 间交换 KV 来补全全局依赖。

它解决的是长序列训练时激活值显存爆炸的问题。和其他并行方式的切分轴不同：DP 切 batch、[[tensor-parallel]] 切权重、PP 切层，CP 切的是 sequence。逐项对照见 [[tp-vs-cp-sharding]]。

## 为什么 attention 特殊

除 attention 外的算子（LayerNorm、MLP、逐点激活）在序列维上是独立的，切开各算各的即可。只有 attention 需要 Q 的每个位置看到全部 KV，所以 CP 的全部复杂度都集中在 attention：要么把 KV 轮转一圈（[[ring-attention]]），要么直接 allgather 全量 KV（flex 路径）。

## PyTorch 中的三层结构

1. **切分**：把输入 buffer 沿 seq 维重排 + 切片 → [[cp-sequence-sharding]]
2. **负载均衡**：决定重排顺序，让因果掩码下各 rank 计算量相当 → [[cp-load-balancers]]
3. **注意力计算**：ring attention 轮转 KV 并增量合并 softmax → [[ring-attention]]

## API 现状（2.14.0a0）

**唯一公开的**是 `torch.distributed.tensor.experimental.context_parallel`——一个 context manager，猴子补丁掉 SDPA，切分传入的 buffers，退出时还原。文档里明确标了 prototype。

其余全部是 `_` 前缀的私有 API，但新代码实际都在用它们：

- `_context_parallel_shard(mesh, buffers, seq_dims, load_balancer)` — 新版切分入口
- `_ContextParallel(seq_dim, attention_type)` — `ParallelStyle` 子类，走 module wrapper 模式而非猴子补丁
- `context_parallel_unshard(mesh, buffers, seq_dims, load_balancer)` — 还原

两套入口对应两种 dispatch 模式（`_DispatchMode.MONKEY_PATCH` vs `MODULE_WRAPPER`），由全局变量 `_dispatch_mode` 切换。旧版 `context_parallel` 强制前者，`_context_parallel_shard` 强制后者。

## 全局配置

`_cp_options`（`_ContextParallelOptions` 的单例）：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `convert_to_f32` | `True` | softmax 合并时升到 fp32，避免累加误差 |
| `enable_load_balance` | `True` | 是否做重排 |
| `rotate_method` | `ALL_GATHER` | KV 轮转方式，见 [[allgather-vs-alltoall-kv-rotation]] |

注释里写 `convert_to_f32` "likely always True"，保留只为实验。

**这是可变全局状态，且被切分入口隐式修改**——见 [[cp-sequence-sharding]] 里的坑。源码自己的 TODO 就写着 "these global variables are going to bite us someday"。

## 约束

- 序列长度必须能被 `world_size * 2` 整除（启用负载均衡时）
- 负载均衡要求 `is_causal=True`，否则 `_templated_ring_attention` 直接抛 `RuntimeError`
- 多头注意力要求各 head 的 mask 一致（索引张量没有 head 维）
- 假定 batch 维是 dim 0
