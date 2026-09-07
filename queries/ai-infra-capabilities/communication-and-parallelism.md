---
title: 大模型通信与并行切分
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, distributed, tp, pp, dp, cp, fsdp, zero, ep, communication]
sources:
  - "[[tensor-parallel]]"
  - "[[context-parallel]]"
  - "[[pipeline-parallel]]"
---

# 大模型通信与并行切分

并行优化的核心不是记住通信原语，而是把张量布局、集合通信、设备拓扑和计算重叠放进同一张时间线上。

## 五个集合通信原语

| 原语 | 输入/输出直觉 | 常见用途 | 主要代价 |
| --- | --- | --- | --- |
| AllReduce | 每个 rank 有一份分片，最后每个 rank 得到相同结果 | 梯度同步、部分结果求和 | 数据量和同步等待较大 |
| AllGather | 每个 rank 持有一段，最后每个 rank 得到完整张量 | 参数/激活拼接、KV 交换 | 瞬时显存和链路压力 |
| ReduceScatter | 先求和，再给每个 rank 一段结果 | ZeRO/FSDP 梯度或参数分片 | 依赖分片布局 |
| AllToAll | 每个 rank 向所有 rank 发送不同片段 | MoE token dispatch、部分 CP | 消息多、对拓扑敏感 |
| Broadcast | 一个 rank 的数据复制到所有 rank | 初始化、控制信息 | 根 rank 或链路可能成为瓶颈 |

### AllReduce、AllGather、ReduceScatter 的区别

设有 $P$ 个 rank，每个 rank 都有一份等长输入。三者最核心的区别是：是否做逐元素归约，以及每个 rank 最终拿到完整结果还是一个分片。

| 原语 | 是否逐元素归约 | 每个 rank 的输出 | 直观理解 |
| --- | --- | --- | --- |
| AllGather | 否 | 所有 rank 的输入片段拼成完整张量 | “把大家手里的不同片段都收集过来” |
| ReduceScatter | 是，通常为 sum | 全局归约结果的一个片段 | “先把相同位置相加，再把结果分给各 rank” |
| AllReduce | 是，通常为 sum | 每个 rank 都得到完整的全局归约结果 | “先归约，再让所有 rank 都拿到结果” |

用 4 个 rank 的向量举例。若 rank $i$ 持有 $X_i=[a_i,b_i,c_i,d_i]$：

- **AllGather**：每个 rank 最终得到 `[X_0, X_1, X_2, X_3]`，不做加法，输出规模约为单个输入的 4 倍。
- **ReduceScatter**：先计算 $S=[\sum_i a_i,\sum_i b_i,\sum_i c_i,\sum_i d_i]$，然后 rank 0 得到 $S$ 的第一个片段，rank 1 得到第二个片段，以此类推。
- **AllReduce**：每个 rank 最终都得到完整的 $S$；如果语义要求平均梯度，再额外除以 $P$。

在常见的 ring 实现中，AllReduce 通常可以理解为 ReduceScatter 阶段加 AllGather 阶段的组合；这是实现上的分解，不代表三者的 API 输入输出可以直接互换。实际通信量还取决于 local/global tensor 的定义、算法、分桶和拓扑。

### 什么时候用哪个

| 需求 | 更合适的原语 | 典型例子 |
| --- | --- | --- |
| 每个 rank 都要一份相同的完整和/平均结果 | AllReduce | DDP 梯度同步、某些 TP 部分结果归约 |
| 每个 rank 只有一片，后续计算需要完整张量 | AllGather | FSDP/ZeRO 参数使用前聚合、拼接分片激活或 KV |
| 只需要全局归约结果的一片，避免复制完整结果 | ReduceScatter | FSDP/ZeRO 梯度分片、分片优化器更新 |

一个实用判断方法：**要不要加法？**不要加法但要完整数据，选 AllGather；要加法且每个 rank 都要完整结果，选 AllReduce；要加法但每个 rank 只需负责一片，选 ReduceScatter。若只有一个指定 rank 需要归约结果，应考虑 `Reduce`，不要无谓使用 AllReduce。

在 [[dp-fsdp-zero-ep-parallelism]] 中，DP/DDP、FSDP 和 ZeRO 的显存差异，本质上就和这三种输出形态有关：DDP 保留完整梯度，FSDP/ZeRO 更倾向于保留 ReduceScatter 后的梯度分片，并在参数真正参与计算前按需 AllGather。

Ring、tree、分层算法是实现策略，不是新的语义。实际性能取决于消息大小、rank 拓扑、链路拥塞、融合/分桶和同步点。

## 并行维度与组合

- **TP**：切权重或 hidden/head 维；典型边界是 colwise/rowwise 配对，见 [[tensor-parallel]] 和 [[colwise-vs-rowwise-parallel]]。
- **PP**：按层切模型，以 micro-batch 隐藏通信和计算；气泡、调度和激活保存是关键，见 [[pipeline-parallel]]。
- **DP**：每个 rank 处理不同样本，梯度或参数需要同步。
- **FSDP/ZeRO**：进一步切分参数、梯度和优化器状态，以通信换显存；详见 [[dp-fsdp-zero-ep-parallelism]]。
- **CP/SP**：沿 sequence 维切激活或 attention 工作量，长序列时尤其依赖负载均衡，见 [[context-parallel]]、[[cp-load-balancers]]。
- **EP**：按 expert 切分，token dispatch 通常需要 AllToAll；热点 expert 会造成负载倾斜，详见 [[dp-fsdp-zero-ep-parallelism]]。

组合并行时要回答：哪个维度被切、哪个布局在模块边界可见、通信发生在何时、是否能和 GEMM/attention 重叠、失败后如何恢复。

## 拓扑和重叠

- 节点内：PCIe、NVLink 或 HCCS；节点间：RDMA、RoCE 或 InfiniBand。
- 先画物理拓扑，再决定 TP/PP/EP 的 rank 映射；把高频、大消息的通信放到带宽更高的局部链路。
- 用 bucket、stream、event 和异步 collective 组织“通信—计算”流水，但要明确依赖，避免为了重叠引入隐式同步。
- 记录 payload、启动次数、有效带宽、等待时间和 overlap ratio，而不是只看 collective 的平均耗时。

## Hang 与性能下降排查

1. 核对 world size、rank、local rank、通信组成员和每个 rank 的 collective 顺序。
2. 确认所有 rank 的 shape、dtype、device 和控制流一致；一个 rank 提前返回就可能让其他 rank 永久等待。
3. 区分初始化失败、网络/拓扑问题、设备 kernel 未完成和 host 线程阻塞。
4. 关闭 overlap、缩小 world size、改单机单卡，再逐层恢复变量。
5. 对比单次大通信和多次小通信，判断是带宽不足、启动开销还是分桶策略问题。

## 实践任务

- 用 2/4 卡测 AllReduce、AllGather、ReduceScatter、AllToAll 的消息大小—带宽曲线。
- 为一个 TP block 画计算和 collective 的时间线，分别实现同步与异步版本。
- 给 DeepSeek-V4 的 TP+PP+CP 组合记录 rank 映射、通信量、bubble 和异常恢复方式，形成一页设计说明。
