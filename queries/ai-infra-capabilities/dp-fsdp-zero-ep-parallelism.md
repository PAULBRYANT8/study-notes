---
title: DP、FSDP、ZeRO 与 EP 并行维度
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, distributed, data-parallel, fsdp, zero, expert-parallel, moe]
sources:
  - "[[communication-and-parallelism]]"
  - "[[tensor-parallel]]"
  - "[[context-parallel]]"
  - "[[dtensor-placement]]"
---

# DP、FSDP、ZeRO 与 EP 并行维度

DP 切 batch，FSDP/ZeRO 切数据并行副本中的模型状态，EP 切 MoE expert；它们和 TP/CP 的区别在于切分对象、通信时机和显存收益不同。
> [!IMPORTANT] [B, S, H]
>   - B：batch size，样本数量
>   - S：sequence length，序列长度
>   - H：hidden size，隐藏维度
>   - batch size 本质上是一次 forward/backward 中包含多少个独立样本或序列。
>   - Global Batch Size 指所有 DP rank 在一次参数更新前总共处理的样本数。
>   - Local Batch Size 指一个 DP rank 在一次计算中处理的样本数。
>   - Micro Batch Size 指一次 forward/backward 处理的最小 batch。当显存不足以一次处理较大的 local batch 时，会把它拆成多个 micro-batch，通过梯  度累积后再更新参数。

## 先建立统一坐标系

| 维度/策略 | 切分对象 | 一个 rank 主要持有什么 | 典型通信 | 主要解决的问题 |
| --- | --- | --- | --- | --- |
| DP/DDP | batch/sample | 完整模型 + 不同样本 | 反向后 AllReduce 梯度 | 扩大吞吐和有效 batch |
| FSDP | 参数、梯度、优化器状态 | 当前需要的参数分片及临时 full parameter | forward/backward AllGather、ReduceScatter | 降低单卡模型状态显存 |
| ZeRO | 参数、梯度、优化器状态，按 stage 逐级切分 | 由 ZeRO stage 决定 | ReduceScatter、AllGather、必要时参数广播 | 在 DP 语义下用通信换显存 |
| TP | hidden/head/权重矩阵维度 | 模型层的权重和激活分片 | AllReduce、AllGather 或 ReduceScatter | 单层计算和参数无法单卡容纳，或提高矩阵并行度 |
| CP/SP | sequence/token 维度 | 部分序列或 attention 工作量 | KV 轮转、AllGather、AllToAll 等 | 长序列显存和 attention 计算 |
| EP | expert 维度 | 一组本地 experts，或 expert 的 TP 分片 | Router 后 AllToAll，返回时再交换/合并 | MoE expert 容量和计算扩展 |

DP 是一种数据划分语义；FSDP 和 ZeRO 是在 DP 语义上进一步分片模型状态的实现/策略。EP 只有在模型包含 MoE experts 时才有意义，不能把它当成普通 Dense 模型的通用维度。

## 一、DP：数据并行

### 工作方式

每个 rank 保存一份完整模型副本，输入 batch 按 sample 维切开，各 rank 独立完成 forward 和 backward，然后同步梯度并各自更新参数：

```text
global batch
  → rank 0/1/... 各取一个 local batch
  → 各自 forward + backward
  → 梯度 AllReduce（或 ReduceScatter）
  → 每个 rank 用相同梯度更新完整模型
```

理想情况下，`global batch = local batch × DP world size × gradient accumulation steps`。实际还要考虑最后一个不完整 batch、梯度平均方式、loss mask 和不同 rank 的有效 token 数。

### 优点、代价和陷阱

- 优点：实现简单，模型代码改动少，适合模型能完整放入单卡的场景。
- 代价：每张卡都保存完整参数、梯度和优化器状态；通信集中在梯度同步。
- 计算重叠：可以在梯度 bucket ready 后异步 AllReduce，但 bucket 太大启动晚，太小则通信启动次数多。
- 正确性：所有 rank 必须有一致的参数更新顺序；随机种子、数据 sampler、dropout 和 loss normalization 不能悄悄产生差异。
- 性能：小 local batch 时通信/launch 开销占比高；global batch 变大又可能影响收敛和泛化，需要和学习率策略一起验证。

DP 适合先建立多卡基线。做 NPU 适配时，先用单卡结果校验数值，再比较 DP 扩展效率，避免把模型算子问题误判成通信问题。

## 二、FSDP：Fully Sharded Data Parallel

### 核心思想

FSDP 将 DP 副本中的参数、梯度和优化器状态分片保存；某个 FSDP module 在计算前临时 AllGather 完整参数，计算后释放或重新分片，反向完成后用 ReduceScatter 聚合并保留本 rank 的梯度分片。

```text
持久状态：每个 rank 只有 parameter/gradient/optimizer state 的一片
forward 前：AllGather → full parameter
forward 后：reshard（可选）
backward 后：ReduceScatter → 本 rank 的 gradient shard
```

常见策略可以按是否切参数理解：

- `FULL_SHARD`：参数、梯度、优化器状态都切，接近 ZeRO-3。
- `SHARD_GRAD_OP`：保留参数副本，切梯度和优化器状态，接近 ZeRO-2 的内存语义。
- `NO_SHARD`：不切状态，行为接近普通 DDP。
- HYBRID 方案：节点内做 shard，节点间做 replicate，利用拓扑降低跨节点通信。

### 需要掌握的实现点

- **FlatParameter**：多个原始参数可能被展平为更大的 buffer，影响参数视图、别名、state dict 和自定义算子边界。
- **auto wrap**：包裹粒度决定 AllGather 次数、峰值显存和通信/计算重叠；太细会频繁通信，太粗会临时占用大量显存。
- **mixed precision 与 offload**：参数、梯度、通信和 master weight 的 dtype/设备需要分别确认，不能只看 module 的 dtype。
- **activation checkpoint**：可以进一步减少激活显存，但会增加重计算；要测峰值显存和端到端吞吐。
- **state dict/checkpoint**：full、sharded、local state dict 的读取方和保存格式不同；大模型恢复必须设计 rank 变化和版本兼容。

FSDP 的主要收益是降低持久模型状态显存；临时 full parameter、激活、通信 buffer 和 allocator 碎片仍可能造成 OOM，因此不能只用“参数除以卡数”估算峰值。

## 三、ZeRO：Zero Redundancy Optimizer

### 三个 stage

| Stage | 切分内容 | 相比普通 DP 的主要节省 | 典型新增通信 |
| --- | --- | --- | --- |
| ZeRO-1 | optimizer states | 优化器状态不再每卡复制 | 梯度同步后更新分片状态 |
| ZeRO-2 | optimizer states + gradients | 梯度也不再每卡复制 | ReduceScatter 梯度、必要时参数同步 |
| ZeRO-3 | optimizer states + gradients + parameters | 参数也不再每卡复制 | 参数使用前 AllGather，梯度 ReduceScatter |

ZeRO 的 stage 是内存/通信语义，不等同于某一个具体代码库。DeepSpeed ZeRO、PyTorch FSDP 和其他实现可能在 bucket、prefetch、offload、checkpoint 和 overlap 上不同，比较时要按实际配置对齐。

### 如何估算收益

把训练状态分成四类：参数 $P$、梯度 $G$、优化器状态 $O$、激活 $A$。DP 主要复制 $P+G+O$；ZeRO/FSDP 逐项把持久状态近似除以 shard world size，但会增加通信 buffer、临时 full parameter 和重计算开销。激活 $A$ 通常要通过 activation checkpoint、sequence/ tensor/context parallel 或 micro-batch 另行处理。

### 常见陷阱

- 把 ZeRO-3/FSDP full shard 当成“永远没有完整参数”：forward、prefetch、保存 checkpoint 时仍可能出现临时聚合。
- 只测单步平均耗时，不测首次 all-gather、长序列峰值显存、checkpoint 和恢复时间。
- 不区分 optimizer state 的 dtype、master weight 和 offload 位置，导致显存账本错误。
- 只比较 stage 名称，不比较 shard group、bucket 大小、prefetch 距离、通信拓扑和容错语义。

## 四、EP：Expert Parallel

### 工作方式

EP 把 MoE 层的 experts 分布到不同 rank。Router 为每个 token 选择 top-k experts，token 按目标 expert 重排并通过 AllToAll 发送；本地 expert 计算后，再把结果交换回原 token 所在 rank 并按 routing weight 合并。

```text
hidden states
  → router / top-k
  → token dispatch + AllToAll
  → local experts（可再做 TP）
  → token combine + AllToAll
  → residual / next layer
```

### 关键问题

- **负载均衡**：expert 热点会让部分 rank 等待；需要 auxiliary loss、capacity factor、token dropping 或动态负载策略，并监控每个 expert 的 token 数。
- **容量与 padding**：静态 capacity 便于规整 kernel，但浪费 token；动态容量节省计算，却增加 shape 和调度复杂度。
- **通信布局**：AllToAll 的 token 排布、expert group、TP group 和 DP group 要明确；不规则 token 可能让通信和 kernel 都变慢。
- **共享 expert**：shared expert 是否复制、是否与 routed expert 共用 TP/EP group，会改变显存和通信边界。
- **训练/推理差异**：训练更关注 token drop、负载均衡损失和反向；推理还要关注请求动态性、decode 小 batch 和跨请求 cache。

EP 不是简单地把 FFN 权重切成几片；它切的是“专家路由后的计算归属”，所以 Router、AllToAll、容量管理和本地 expert kernel 必须一起设计。

## 五、与 TP/CP 的组合关系

常见的逻辑并行规划可以先写成 `world size ≈ TP × PP × DP × CP × EP`。这是 rank 组织的近似表达，不代表所有维度都必须同时使用；FSDP/ZeRO 通常是在 DP group 内改变参数、梯度和优化器状态的保存方式，而不是额外增加一个独立轴。

| 组合 | 适合场景 | 重点风险 |
| --- | --- | --- |
| TP + DP/FSDP | Dense 层太大，同时希望扩大全局 batch | TP 通信与 shard 通信争用链路；group 映射不合理 |
| TP + EP | MoE expert 内部还需要矩阵切分 | token dispatch、expert TP 和返回合并的顺序复杂 |
| CP + TP | 长序列且单层矩阵也很大 | sequence/head 布局、KV 交换和 TP collective 叠加 |
| PP + FSDP/ZeRO | 超大模型跨 stage 训练 | stage 间 bubble、参数 all-gather、checkpoint 恢复 |
| DP + EP | MoE 通过数据副本扩吞吐 | 每个 DP replica 的 expert 负载、AllToAll 和全局 batch 对齐 |

组合前先定义每个 process group 的成员，再画一层 forward/backward 时间线：参数何时可用、token 何时交换、梯度何时归约、哪个 collective 能和计算重叠。

## 六、统一对比：怎么选

| 约束 | 优先考虑 | 原因 |
| --- | --- | --- |
| 模型能完整放入单卡，想扩吞吐 | DP/DDP | 改动最小，通信模式简单 |
| 参数能放下，但梯度/优化器状态放不下 | ZeRO-1/2 或 FSDP 对应策略 | 先切持久状态，避免临时参数通信 |
| 单卡无法容纳完整模型状态 | ZeRO-3/FSDP FULL_SHARD + TP/PP | 参数和状态共同分片，再按层/矩阵切计算 |
| 模型含大量 MoE experts | EP，必要时组合 TP | 把 expert 计算分散到 rank，控制 token 路由 |
| 长序列 attention 显存或计算成为瓶颈 | CP/SP，必要时组合 TP | 按 sequence 维切工作量和 KV 处理 |

最终选择要同时看：单卡峰值显存、通信量和拓扑、计算/通信重叠、实现复杂度、checkpoint/恢复、数值一致性和目标吞吐。没有脱离模型形状、batch、序列长度和设备拓扑的“最优并行策略”。

## 七、面向 Ascend NPU 的实践清单

1. 用 2/4 卡建立 DP baseline，记录 local/global batch、梯度同步时间、有效带宽和扩展效率。
2. 对同一模型分别测 FSDP/ZeRO-2 语义与 full-shard 语义，记录参数 all-gather、ReduceScatter、峰值显存和 checkpoint 时间。
3. 用一个小型 MoE 测 EP AllToAll，记录每个 expert token 数、通信 payload、capacity/token drop 和尾部延迟。
4. 将 TP/CP/EP/FSDP 的 process group 和 rank 映射画成表，确认 HCCL 通信没有跨越不必要的慢链路。
5. 在 profiler 中区分 kernel、collective、host gap、layout conversion 和 allocator 峰值；不要用单个“设备利用率”替代完整证据。

相关基础：[[communication-and-parallelism]]、[[tensor-parallel]]、[[context-parallel]]、[[pipeline-parallel]]、[[dtensor-placement]]、[[cp-load-balancers]]。

[^1]: 
