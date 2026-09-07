---
title: Kernel 接入与性能工程
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, kernel, ascendc, triton, cuda, profiling, optimization]
sources:
  - "[[swiglu-group-接入复盘]]"
  - "[[inplace-partial-rotary-mul-接入复盘]]"
  - "[[accelerator-architecture]]"
---

# Kernel 接入与性能工程

Kernel 工程的基本闭环是“定义语义 → 选数据布局 → 设计 tiling/搬运 → 实现 → 正确性验证 → profiler 定位 → 端到端回归”。

## 需要掌握的两类工具

- **Ascend C**：理解 host 侧 tiling、device 侧 kernel、Vector/Cube 调度、片上 buffer、数据搬运和 double buffer；掌握编译、注册、打包和 Python 调用边界。
- **Triton/CUDA**：用通用 GPU kernel 语言建立 block、program、shared memory、warp 和融合的迁移能力。即使主战场是 NPU，也能借此理解通用 kernel 设计。

目标不是背 API，而是能根据算子形状和硬件约束解释选择。

## 算子设计检查表

1. **语义**：广播、边界、in-place、数值精度、异常输入和梯度定义是否明确。
2. **布局**：输入输出 stride、连续性、NZ/ND 或内部格式是否一致；是否需要 transpose/contiguous。
3. **切分**：根据 M/N/K、sequence、head、expert 和 dtype 选择 tile；尾块如何处理，是否需要 padding。
4. **搬运**：HBM 与片上 buffer 之间的读写是否合并，能否 double buffer；中间结果是否重复落盘。
5. **计算单元**：elementwise/归一化偏向 Vector，矩阵乘/投影偏向 Cube；混合算子要安排数据转换和同步。
6. **融合边界**：融合能否减少读写和 launch；不要只看单 kernel 时间，必须看上下游布局和端到端收益。

## 性能分析方法

- 先用 [[accelerator-architecture]] 的算术强度和 Roofline 做数量级判断。
- profiler 至少观察：设备 kernel 时间、HBM/片上带宽、利用率、occupancy、同步、layout conversion、host launch。
- 固定 warmup、重复次数、shape、dtype、频率和同步方式；报告 p50/p90/p99 或均值+方差。
- 对比 baseline 时保留正确性误差、显存峰值和端到端吞吐，避免“局部变快、全局变慢”。

## 当前工作可沉淀的案例

- [[SwigluGroup-融合算子接入介绍]] / [[swiglu-group-接入复盘]]：融合 FFN、分组权重、shape 和通信边界。
- [[inplace-partial-rotary-mul-接入介绍]] / [[inplace-partial-rotary-mul-接入复盘]]：in-place 语义、部分 rotary、别名和正确性验证。
- 对每个案例补一张表：输入规模、原始算子链、融合后数据流、kernel/通信耗时、误差、回归结果和适用 shape。

## 最小实践项目

1. 先实现一个 Vector 类 elementwise/归一化算子，完成非连续输入和尾块测试。
2. 再实现一个带 tiling 和 double buffer 的矩阵或 fused FFN 子图。
3. 将 kernel 接到 PyTorch dispatcher，加入 CPU/reference fallback、文档和 benchmark。
4. 选择小 batch、长序列、MoE 不均匀 token 三个压力点，写出“不适合使用该 kernel”的边界。
