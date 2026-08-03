---
title: Tensor Parallel (TP)
type: concept
created: 2026-07-26
updated: 2026-07-27
tags: [pytorch, 分布式, dtensor, 张量并行]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

Tensor Parallel（张量并行，TP）把**单个算子的权重矩阵**切开分到多卡，每卡算一部分再通过集合通信拼回来。切的是模型参数，不是数据。

和 [[context-parallel]] 的对照：CP 切 activation 的序列维、参数完整复制；TP 切参数、activation 按需重分布。两者正交，可叠加。详见 [[tp-vs-cp-sharding]]。

## 核心思路

以 Transformer 的 MLP 为例，`y = W2 @ act(W1 @ x)`：

这里写的是两矩阵普通 FFN；它的三矩阵门控变体、`2/3` 中间维度匹配方法与 T5 实验结果见 [[glu-variants-improve-transformer]]。

- `W1` 按**列**切（[[colwise-parallel]]）→ 每卡得到一段 hidden，激活函数逐元素所以无需通信
- `W2` 按**行**切（[[rowwise-parallel]]）→ 每卡的输出是全局结果的部分和，一次 allreduce 收尾

关键在于 colwise 的输出布局（`Shard(-1)`）恰好就是 rowwise 需要的输入布局（`Shard(-1)`），**中间那一层不需要任何通信**。整个 MLP 两层只付一次 allreduce。Attention 同理：QKV 投影 colwise、输出投影 rowwise，按 head 天然切分。

## PyTorch 的实现分层

1. **底层**：DTensor 的 placement 系统（`Shard` / `Replicate` / `Partial`）表达切分状态，`redistribute()` 自动插入集合通信 → [[dtensor-placement]]
2. **策略**：`ParallelStyle` 的子类声明"这个模块的参数怎么切、输入输出什么布局" → [[tp-module-sharding]]
3. **应用**：`parallelize_module(module, mesh, plan)` 按 FQN 把策略贴到子模块上
4. **编译期优化**：inductor 的 async TP pass 把通信和 matmul 融合 → [[micro-pipeline-tp]]

## 内置策略

| 策略 | 用途 |
|---|---|
| [[colwise-parallel]] | Linear/Embedding 列切，输出 `Shard(-1)` |
| [[rowwise-parallel]] | Linear/Embedding 行切，输出 allreduce 成 `Replicate` |
| [[sequence-parallel]] | LayerNorm/Dropout/RMSNorm，参数复制、激活按序列维切 |
| [[prepare-module-input-output]] | 不改参数，只在边界处调整张量布局 |
| [[loss-parallel]] | 词表维切分下直接算 cross entropy，不聚合 logits |

## 与 CP 的关键区别：API 成熟度

TP 的 API **全部是公开且稳定的**（`torch.distributed.tensor.parallel` 下 9 个导出符号，无 `_` 前缀）。CP 除 `context_parallel` 外全是私有 prototype。这个差别反映在文档、BC 保证和用法约定上。

## 约束

- `parallelize_module` **只接受 1-D DeviceMesh**。N-D mesh 必须先切片：`mesh["tp"]`
- Colwise/Rowwise 只支持 `nn.Linear` 和 `nn.Embedding`，其他模块 `NotImplementedError`
- SequenceParallel 假定权重是 ones 初始化（LayerNorm/RMSNorm 的默认），自定义初始化需要手动广播
