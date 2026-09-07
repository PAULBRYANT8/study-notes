---
title: AI Infra 能力提升路线与项目验收
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, roadmap, projects, career]
sources:
  - "[[npu-training-adaptation-learning-path]]"
  - "[[queries/ai-infra-capabilities/README]]"
---

# AI Infra 能力提升路线与项目验收

最有效的学习方式是围绕当前 NPU 适配工作建立连续项目，每个阶段同时产出代码、基线、故障复盘和可讲清楚的设计文档。

## 先做一次自评

给以下能力分别打 0～3 分：0=未接触，1=能运行/修改，2=能独立定位，3=能设计并指导他人。

- C++ 与 Linux 调试；
- NPU/GPU 体系结构与 profiler；
- TP/PP/DP/CP/EP 与集合通信；
- Ascend C/Triton/CUDA kernel；
- PyTorch Dispatcher、编译器和分布式 API；
- vLLM/SGLang/MindIE 等推理运行时；
- 容器、调度、网络、存储、观测与可靠性。

不要只看“是否用过”，要以能否提交最小复现、解释根因和给出回归证据作为评分依据。

## 12 个月四阶段

### 阶段一：0～3 个月，补底座并建立证据习惯

重点：C++17、gdb/core、CMake、Linux、Tensor/ATen/Dispatcher、基础体系结构和通信原语。

交付：

- 一个 C++/PyTorch 自定义算子扩展；
- 一份从 Python 到 NPU kernel 的调用链；
- 2/4 卡 collective benchmark；
- 一次 crash、hang 和数值不一致的分层复盘。

### 阶段二：4～6 个月，kernel 与编译器深入

重点：Ascend C tiling、double buffer、layout、torch.compile/Dynamo/Inductor、图断裂和 fallback。

交付：

- 一个融合算子及完整 benchmark；
- 一个 graph break 或 lowering 问题的最小修复；
- baseline、误差、显存、端到端收益和不适用 shape 的报告。

### 阶段三：7～9 个月，推理引擎与服务化

重点：scheduler、continuous batching、prefill/decode、KV Cache、attention backend、TP 服务和 SLO。

交付：

- 在 A5 上对同一模型完成至少一个推理引擎的适配/调优；
- 记录 TTFT、TPOT、P99、吞吐、显存、编译缓存和长稳结果；
- 完成 OOM、取消、设备异常和重启演练。

### 阶段四：10～12 个月，系统化 Infra 与方向选择

重点：Kubernetes/Ray、RDMA/拓扑、checkpoint、Prometheus/Grafana、故障恢复和成本模型。

交付：

- 一份多机多卡任务生命周期设计；
- 一套可查询的 metrics/log/tracing 面板或最小实现；
- 一次故障注入演练和 runbook；
- 根据兴趣选择“训练适配/Kernel 性能”“推理运行时”或“平台 Infra”主线。

## 项目选择建议

### 训练适配主线

继续深挖 `torchtitan-npu`：DeepSeek-V4 的 TP/PP/CP/EP 组合、数值一致性、compile 稳定性和算子融合。把每个 PR 转化为可复现 benchmark 和设计说明。

### 推理主线

选择 vLLM-Ascend、SGLang-Ascend 或 MindIE 之一作为主运行时，再比较另外两个的调度、KV Cache、模型适配和生产化边界，见 [[inference-engines]]。

### Infra 主线

围绕多机 NPU job 做 launcher、拓扑感知资源分配、checkpoint、观测和故障恢复，不必一开始实现完整调度器。

## 面试与简历验收标准

每个项目都准备 5 分钟版本：背景和约束、架构图、关键代码、基线与收益、失败案例、可推广边界。优先展示“定位过什么跨层问题、用什么证据证明、如何防止回归”，而不是只罗列工具名。
