---
title: 分布式训练与推理 Infra 基础
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, distributed-systems, kubernetes, ray, observability, reliability]
sources:
  - "[[npu-training-adaptation-learning-path]]"
  - "[[communication-and-parallelism]]"
---

# 分布式训练与推理 Infra 基础

Infra 能力的边界，是能解释一个多机多卡任务从提交、调度、启动、通信、存储、观测到失败恢复的完整生命周期。

## 分层地图

### 主机与资源

- Linux process/thread、signals、cgroups、NUMA、CPU affinity、shared memory、file descriptor。
- Docker 镜像、驱动/固件/Toolkit 兼容矩阵、设备可见性、容器内权限和日志采集。
- Kubernetes device plugin、node label、taint/toleration、资源配额和拓扑感知调度；Ray 用于任务/actor 编排时要理解其资源声明和故障语义。

### 调度与执行

- 队列、优先级、配额、抢占、gang scheduling、backfill、碎片和拓扑约束。
- 多机启动：rank/world size/local rank、环境变量、时钟、SSH/launcher、网络接口和 HCCL/NCCL 初始化。
- 训练需要 checkpoint、断点续训和 elastic 语义；推理需要服务发现、路由、限流、扩缩容和优雅下线。

### 网络与存储

- TCP/IP、RDMA、RoCE/IB、拥塞控制、MTU、网卡/NUMA 亲和性和带宽测试。
- 本地 NVMe、对象存储、并行文件系统、缓存和 checkpoint 分片；关注吞吐、并发、元数据压力和一致性。

### 可观测性与可靠性

- 日志：包含 job、rank、request、设备、版本和时间戳，可聚合检索。
- Metrics：吞吐、利用率、通信时间、队列长度、TTFT/TPOT、OOM、重试和 checkpoint 时间。
- Tracing：把调度、数据加载、通信、kernel 和服务请求放在同一条时间线上。
- Prometheus/Grafana 只是一种实现；更重要的是定义 SLI/SLO、告警阈值、runbook 和故障演练。

## 1000 卡任务的思考框架

```text
提交/排队 → 资源与拓扑分配 → 镜像/环境准备 → 多机启动
→ 通信组建立 → 数据/权重/checkpoint 读取 → 训练或服务
→ 指标与日志 → 失败重试/恢复 → 释放资源与产物归档
```

对每一步写出：输入、输出、依赖、超时、幂等性、可观测信号和失败后的回滚方式。这个框架能把“会调模型”提升到“能运营系统”。

## 实践项目

- 用 Docker 启动一个多进程 NPU demo，故意注入 rank 配置错误、网络端口冲突和设备不可见，写 runbook。
- 用 Prometheus/Grafana 或轻量替代品采集训练吞吐、通信等待、设备利用率和 OOM。
- 设计 checkpoint 分片上传、校验、断点恢复和版本兼容测试。
- 做一次节点下线或进程杀死演练，验证任务能否按预期失败、重试或恢复，而不是无限 hang。
