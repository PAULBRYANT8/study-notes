---
title: PyTorch 内部机制与 NPU 编译链
type: query
created: 2026-08-31
updated: 2026-08-31
tags: [pytorch, dispatcher, aten, dynamo, inductor, torch-compile, npu]
sources:
  - "[[PyTorch编译问题总结：Dynamo、Inductor与NPU Codegen]]"
  - "[[python-advanced-mechanisms-for-pytorch]]"
  - "[[dtensor-placement]]"
---

# PyTorch 内部机制与 NPU 编译链

理解 PyTorch 的关键，是把 Python 模块、算子 schema、DispatchKey、Autograd、分布式布局和编译后端串成一条可追踪路径。

## eager 执行路径

```text
Python Module/torch.ops
  → Tensor 与 ATen operator schema
  → Dispatcher / DispatchKey
  → Autograd 或 Functionalization
  → torch_npu / CPU / 其他 backend kernel
  → runtime、stream、设备执行
```

重点学习：

- Tensor 的 storage、stride、view、alias 和 device/dtype；
- ATen schema、operator registration、backend kernel 和错误处理；
- DispatchKey 如何选择实现，PrivateUse1/自定义 backend 适配时边界在哪里；
- Autograd、AOTAutograd、forward/backward graph 与自定义算子的梯度注册；
- `torch.distributed`、ProcessGroup、DTensor、DeviceMesh、FSDP 如何表达布局和通信，见 [[dtensor-placement]]。

## `torch.compile` 路径

```text
Python 程序
  → TorchDynamo 捕获 FX graph
  → decomposition / functionalization
  → AOTAutograd（需要时）
  → Inductor lowering 与调度
  → NPU codegen / backend kernel
  → runtime 编译、缓存和执行
```

每次遇到 graph break、fallback、编译失败或性能回退，都记录：触发的 Python 语句、graph 输入约束、decomposition、lowering、生成的 kernel、设备执行和 fallback 代价。

## NPU 适配的源码阅读任务

1. 找一个现有 `torch_npu` 算子，分别定位 schema、注册、Python 包装和设备 kernel。
2. 对一个无法编译的 DeepSeek-V4 子图，保存 Dynamo/Inductor 日志，标注 graph break 和 unsupported op。
3. 比较 eager、单算子、图模式和自定义 lowering 的正确性、首次编译时间、缓存命中率和稳定吞吐。
4. 把 [[PyTorch编译问题总结：Dynamo、Inductor与NPU Codegen]] 中的案例整理成“症状—层级—证据—修复—回归”表。

## 验收标准

- 能画出一个自定义算子从 Python 到设备 kernel 的调用链。
- 能区分 dispatcher 未命中、Autograd 不支持、Dynamo graph break、Inductor lowering 缺失和设备 kernel 错误。
- 能解释一个 fallback 为什么影响性能，并给出最小复现和回归测试。
