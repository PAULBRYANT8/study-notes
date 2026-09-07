---
title: GPU 与 Ascend NPU 体系结构基础
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, npu, gpu, architecture, performance]
sources:
  - "[[npu-training-adaptation-learning-path]]"
  - "[[deepseek-v4-architecture-and-execution]]"
---

# GPU 与 Ascend NPU 体系结构基础

体系结构学习的结果，是能把“利用率低、shape 变慢、布局不匹配”翻译成计算、访存、并行度和同步上的具体假设。

## 统一抽象

| 通用概念 | GPU 中的常见叫法 | Ascend 中的学习抓手 | 需要关注的现象 |
| --- | --- | --- | --- |
| 标量控制 | CUDA core / scalar pipeline | Scalar | 分支、索引、控制流是否成为瓶颈 |
| 向量计算 | SIMD/SIMT 执行单元 | Vector | elementwise、归一化、激活的吞吐和对齐 |
| 矩阵计算 | Tensor Core | Cube | GEMM、attention 投影和量化矩阵乘 |
| 片上高速存储 | register/shared memory/cache | L1/L2、Unified Buffer 等片上 buffer | 重用、容量、bank/conflict、搬运次数 |
| 设备内存 | HBM/GDDR | HBM | 带宽、访问合并、跨卡访问和容量 |
| 通信互联 | NVLink/PCIe/RDMA | HCCS、PCIe、RDMA/RoCE 等 | 拓扑、跳数、拥塞、集合通信带宽 |

不同厂商的名词不能机械等同；表格用于建立问题映射，具体型号、指令和对齐要求必须以对应版本文档及 profiler 为准。

## 需要掌握的性能模型

- **计算密集 vs 访存密集**：先估算 FLOPs、读写字节数和并行工作量，再判断瓶颈。
- **Arithmetic Intensity**：`FLOPs / Bytes`。强度低的算子通常先受带宽限制，高强度 GEMM 才更可能受矩阵计算吞吐限制。
- **Roofline**：把算术强度与设备峰值带宽/峰值算力比较，得到理论上界；实测还会受 launch、同步、占用率和布局影响。
- **端到端账本**：算子耗时之外，记录 host launch、同步、通信、数据格式转换和 pipeline bubble。

## Ascend 适配中最容易忽略的维度

1. **布局**：ND/NZ 或其他内部格式的转换是否被隐藏在算子边界；transpose、contiguous 和 stride 变化是否触发额外搬运。
2. **对齐**：最后一维、tile 大小、batch/sequence 维是否满足 kernel 的向量化和矩阵化约束。
3. **shape**：静态 shape、动态 shape、小 batch、长序列和 MoE 不规则 token 分布可能落入完全不同的 kernel 路径。
4. **精度**：FP32、FP16、BF16、低比特量化不仅改变误差，也改变可用的 Cube/Vector 路径和带宽压力。
5. **同步**：隐式同步会让异步流水失效；必须区分 host 计时、设备 event 计时和端到端用户可见时延。

## 实验模板

对一个算子固定记录：输入 shape、dtype、layout、batch、warmup 次数、重复次数、设备数量、同步方式、显存峰值、设备耗时和吞吐。至少比较：

- baseline 算子 vs 融合算子；
- 连续布局 vs 非连续布局；
- 小 shape vs 目标 shape；
- 单卡 vs 多卡；
- eager vs 图/编译模式。

## 结合当前工作的练习

- 以 DeepSeek-V4 的 attention、MoE 和 `SwiGLU` 为例画出 HBM→片上 buffer→Cube/Vector 的数据流。
- 对 [[swiglu-group-接入复盘]] 和 [[inplace-partial-rotary-mul-接入复盘]] 补齐算术强度、搬运量、layout 和同步分析。
- 用 profiler 找出一次“理论上融合但端到端没有变快”的反例，说明收益被哪一层抵消。
