---
title: LLM 推理引擎与服务化
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [ai-infra, inference, vllm, sglang, mindie, serving, kv-cache]
sources:
  - "[[npu-training-adaptation-learning-path]]"
  - "[[deepseek-v4-architecture-and-execution]]"
---

# LLM 推理引擎与服务化

推理工程要把单次模型执行改造成面向请求的系统：调度、动态批处理、KV Cache、并行、kernel 和服务可靠性共同决定用户可见性能。

## 一条完整执行链

```text
HTTP/RPC 请求
  → tokenizer 与请求队列
  → scheduler / continuous batching
  → prefill 或 decode 调度
  → KV Cache manager
  → model runner 与 TP/PP/DP/EP
  → attention backend / custom op
  → 设备 kernel 与通信
  → detokenizer、流式返回和 metrics
```

训练代码通常以固定 batch 和完整反向图为主；推理服务必须处理不同到达时间、不同 prompt 长度、输出长度、取消和超时。

## 必学概念

- **Prefill / Decode**：前者计算密集、一次处理大量 prompt token；后者逐 token 生成、KV 访问和调度开销突出。
- **Continuous batching**：每轮把已完成请求移出、把新请求插入，提升设备利用率，同时要控制长请求对短请求的影响。
- **KV Cache**：理解 block/page、生命周期、复用、显存碎片、prefix cache 和跨请求共享；缓存布局必须和 attention kernel 对齐。
- **调度策略**：FIFO、token budget、优先级、最大并发、抢占和 chunked prefill 的取舍。
- **PD 分离**：prefill/decode disaggregation 可以分别扩展资源，但引入 KV 传输、路由、流量控制和故障恢复成本。
- **规格解码与投机执行**：用 draft model 或其他预测路径换取 decode 步数，必须以端到端接受率和成本评估。

## vLLM-Ascend、SGLang-Ascend 与 MindIE 的学习角度

三者都要回答“模型图如何落到 Ascend 设备、请求如何被调度、KV 如何管理、kernel 如何执行”，差异主要在运行时抽象、调度/缓存策略、模型适配范围、插件边界和生产化组件。学习时不要只比较启动命令，建议固定同一模型、同一 dtype、同一并发和同一 prompt 集合，记录：

- TTFT（首 token 延迟）、TPOT/ITL（生成间隔）、端到端延迟和 P99；
- prefill/decode 吞吐、并发上限、KV Cache 命中率、显存峰值；
- graph break/fallback、通信比例、首次编译时间和长稳运行错误；
- 模型改造量：权重格式、custom op、attention backend、量化和多卡启动方式。

## 与当前训练适配的连接

- `torchtitan-npu` 中的 TP/PP/CP、dtype、layout 和自定义算子，是推理 engine 的模型执行底座。
- 训练侧关注吞吐、收敛和 checkpoint；推理侧还要关注请求级公平性、流式体验、缓存复用和服务 SLO。
- 训练中验证过的 kernel 不一定适合 decode：小 batch、动态 shape、频繁 launch 和 KV 访问会改变瓶颈。

## 实践项目

1. 在同一台 A5 上部署一个小模型，画出请求从入队到返回的时序图。
2. 采集不同 prompt/output 长度和并发下的 TTFT、TPOT、P99、显存和设备利用率。
3. 替换一个 attention 或 FFN backend，证明局部 kernel 优化是否改善端到端指标。
4. 设计一次 OOM、请求取消、设备异常和节点重启演练，记录缓存清理和请求恢复行为。

Agent 系统可以把推理 engine 当作工具或执行层；Agent 的规划、记忆、工具调用和评测属于更上层产品逻辑，不应替代对推理运行时的学习。
