---
title: PyTorch 的 TP 是怎么切分模块的？
type: query
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, 分片]
sources:
  - "[[2026-07-26-pytorch-tp-source]]"
  - 本地仓库 commit a07d9d50489 (2026-07-25), version.txt 2.14.0a0
---

**结论：TP 是声明式的——`parallelize_module` 按 FQN 把 `ParallelStyle` 挂到子模块上，style 负责把参数换成 DTensor 并注册改布局的 hook。通信不是写出来的，是 DTensor 在布局不匹配时自动推导出来的。**

## 一句话流程

```python
parallelize_module(model, tp_mesh, {"layers.*.w1": ColwiseParallel(), ...})
  └─► style._apply(submodule, mesh)
        └─► distribute_module(module, mesh, partition_fn, input_fn, output_fn)
              ├─ partition_fn: distribute_tensor(param, mesh, [Shard(d)]) → nn.Parameter
              ├─ input_fn:  from_local(声明) + redistribute(到 desired)   ← 通信在这
              └─ output_fn: redistribute(到 output_layouts)               ← 通信在这
```

想知道一份 plan 会产生多少通信，把每层的 `output_layouts` 和下一层的 `desired_input_layouts` 对上，不一致的地方查 [[dtensor-placement]] 的映射表。详见 [[tp-module-sharding]]。

## 核心切法

| | weight | 输入要求 | 输出 |
|---|---|---|---|
| [[colwise-parallel]] | `Shard(0)`（Linear） | `Replicate` | `Shard(-1)`，无需归约 |
| [[rowwise-parallel]] | `Shard(1)`（Linear） | `Shard(-1)` | `Partial` → allreduce/reduce_scatter |

存储视角和数学视角是错位的：`nn.Linear.weight` 形状 `(out, in)` 参与的是 `weight^T`，所以存储切 dim 0 = 数学切列。

**配对使用时中间零通信**，整个 MLP/Attention 只付一次归约——这是 TP 的全部要点。见 [[colwise-vs-rowwise-parallel]]。

## 分层回答

| 层 | 做什么 | 关键代码 |
|---|---|---|
| 底层 | placement 表达布局，redistribute 推导通信 | `placement_types.py`、`_redistribute.py:191-195` |
| 策略 | 声明参数切法 + 输入输出布局 | `parallel/style.py` |
| 应用 | FQN 匹配 + 递归下发 | `parallel/api.py:14-142` |
| 编译 | 通信与 matmul 融合流水 | [[micro-pipeline-tp]] |

## 值得记住的坑

1. **plan 的 key 匹配不上只 warning 不报错**。改了模型结构后 plan 失效，训练照跑但没并行。
2. **`input_layouts` 是断言不是转换**，`from_local(run_check=False)` 不校验。声明错了静默算错。已经是 DTensor 的输入更是**完全忽略你的声明**（源码里那行 assert 被注释掉了，TODO 说等修好 compile 路径再开）。
3. **[[sequence-parallel]] 用 `from_local` 复制参数**，依赖 ones 初始化假设。自定义 init 会导致各卡权重不一致却被标为 Replicate。
4. **`RowwiseParallel._apply` 会写实例属性**，同一实例复用到 Linear 和 Embedding 上会互相覆盖。
5. **`parallelize_module` 只吃 1-D mesh**，N-D 必须 `mesh["tp"]` 切片。
6. **async TP 默认关闭**（`_micro_pipeline_tp = False`），且需要 Symmetric Memory 可用，否则静默跳过融合。

## 与 CP 的关系

切的东西不同（参数 vs 激活序列维），实现风格也相反（全程持有 DTensor vs 用完即弃）。TP 的切分对象是齐次的所以不需要负载均衡，CP 的不是。详见 [[tp-vs-cp-sharding]]。

## 状态

TP 的 9 个导出符号全部是公开稳定 API，和 [[context-parallel]] 的 prototype 状态形成对比。唯一带下划线的是 inductor 侧的 `_micro_pipeline_tp` 配置。

## 相关

[[tensor-parallel]] · [[tp-module-sharding]] · [[dtensor-placement]] · [[colwise-vs-rowwise-parallel]] · [[loss-parallel]] · [[prepare-module-input-output]]
