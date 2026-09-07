---
title: AI Infra 七类能力学习地图
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, learning-path, npu, training, inference]
sources:
  - "[[npu-training-adaptation-learning-path]]"
---

# AI Infra 七类能力学习地图

这份目录把 `AI_Infra_7类能力学习清单.txt` 整理成一条面向 NPU 训练适配、推理引擎和 Infra 工程的可执行学习路线。

整理依据：`/home/zhangwei/下载/AI_Infra_7类能力学习清单.txt`。这里记录的是学习重点、实践入口和验收标准，不替代各专题的源码分析；已有专题优先复用并通过链接串起来。

## 七类能力总览

| 能力 | 要回答的核心问题 | 入口文档 |
| --- | --- | --- |
| C++ 工程 | 能否读懂并调试 PyTorch、后端和运行时的 C++ 调用链？ | [[cpp-engineering]] |
| 加速器体系结构 | 一个算子或模型为什么受计算、访存或布局限制？ | [[accelerator-architecture]] |
| 通信与并行 | TP/PP/DP/CP/EP 等切分怎样映射到集合通信和拓扑？ | [[communication-and-parallelism]] |
| Kernel 与性能 | 能否定位慢算子并设计、接入、验证融合算子？ | [[kernel-and-performance]] |
| PyTorch 内部与编译器 | Python API 如何落到 Dispatcher、Inductor 和 NPU 后端？ | [[pytorch-internals-and-compiler]] |
| 推理引擎 | 请求如何经过调度、批处理、KV Cache 和 Attention Backend？ | [[inference-engines]] |
| 分布式 Infra | 一个多机多卡任务如何被调度、运行、观测并恢复？ | [[distributed-infra]] |

## 建议阅读顺序

1. 先读 [[roadmap-and-projects]]，按自己的岗位方向选择主线。
2. 以 [[accelerator-architecture]]、[[communication-and-parallelism]] 建立硬件和分布式抽象。
3. 沿 [[pytorch-internals-and-compiler]] 读懂训练代码从 Python 到后端的路径。
4. 用 [[kernel-and-performance]] 把现有算子接入工作转化为可复用的方法论。
5. 再读 [[inference-engines]] 和 [[distributed-infra]]，补齐推理服务及集群视角。
6. 每个阶段都完成一个可复现项目，并把数据、日志、瓶颈和结论记录下来。

## 与现有工作之间的映射

- DeepSeek-V4 的 TP/PP/CP、负载均衡和布局问题：[[tensor-parallel]]、[[pipeline-parallel]]、[[context-parallel]]、[[cp-load-balancers]]。
- DP、FSDP、ZeRO、EP 与 TP/CP 的切分对象、通信时机和显存收益：[[dp-fsdp-zero-ep-parallelism]]。
- `torchtitan-npu` 的编译、算子和日志问题：[[PyTorch编译问题总结：Dynamo、Inductor与NPU Codegen]]、[[swiglu-group-接入复盘]]、[[inplace-partial-rotary-mul-接入复盘]]。
- 数值一致性和训练稳定性：[[deep-learning-training-numerics]]、[[floating-point-error-analysis]]。
- 调试基础：[[linux-debugging-for-npu-adaptation]]、[[cpp-reading-for-pytorch-backends]]、[[python-advanced-mechanisms-for-pytorch]]。

## 使用方法

每学完一个主题，至少留下四类证据：

- 一张调用链或数据流图；
- 一个可以重复运行的最小实验；
- 一份性能或正确性基线；
- 一段解释“现象—假设—验证—结论”的复盘。

这样形成的材料既能帮助后续排查 NPU 适配问题，也能直接转化为简历项目、技术分享或面试中的系统性案例。
