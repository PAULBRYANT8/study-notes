---
title: ColwiseParallel
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, linear, embedding]
sources: [[[2026-07-26-pytorch-tp-source]]]
---

`ColwiseParallel` 把 `nn.Linear` / `nn.Embedding` 的权重按**输出维度**（列）切开，输入要求完整复制，输出天然按最后一维分片。

位置：`torch/distributed/tensor/parallel/style.py:45-183`。

## 切法

| 模块 | 参数 | placement |
|---|---|---|
| `nn.Linear` | weight **和 bias** | `Shard(0)` |
| `nn.Embedding` | weight | `Shard(1)` |

Linear 的注释解释了这个看似矛盾的写法：

> colwise shard weight/bias to Shard(0), weight be Shard(0) means Colwise as Linear is `input * weight^T + bias`, where weight would become Shard(1)

`nn.Linear.weight` 的存储形状是 `(out_features, in_features)`，数学上参与的是 `weight^T`。所以**存储上切 dim 0 = 数学上切列**。名字"colwise"说的是数学视角，代码写的是存储视角。

bias 形状是 `(out_features,)`，跟着输出维一起切，不需要额外处理。这一点和 [[rowwise-parallel]] 正好相反。

## 布局

| | 默认值 | 说明 |
|---|---|---|
| `input_layouts` | `Replicate()` | 声明进来的张量是完整的 |
| `desired_input_layouts` | `Replicate()` | **写死**，不可配置 |
| `output_layouts` | `Shard(-1)` | 每卡持有输出的一段 |
| `use_local_output` | `True` | 默认返回普通 Tensor |

`desired_input_layouts` 恒为 `Replicate()`——colwise 的数学要求就是每卡都要看到完整输入。所以如果 `input_layouts` 声明成 `Shard(...)`，就会在 forward 前触发一次 `all_gather`。

## 为什么它和 Rowwise 配对

colwise 输出是 `Shard(-1)`，而 [[rowwise-parallel]] 的 `desired_input_layouts` 也是 `Shard(-1)`。**两者衔接处零通信**，这是 TP 在 MLP / Attention 上只需一次归约的根本原因。

docstring 里专门提醒：输出默认分片，如果下游算子对形状敏感（比如不是紧接着 RowwiseParallel），要么改 `output_layouts`，要么调整算子去适应分片后的尺寸。

## 约束

- 只支持 `nn.Linear` 和 `nn.Embedding`，其他类型抛 `NotImplementedError`
- `use_local_output=True` 意味着输出脱离 DTensor 追踪，下游要自己保证布局假设正确

## 相关

- 配对使用：[[rowwise-parallel]]，对比见 [[colwise-vs-rowwise-parallel]]
- 上层机制：[[tp-module-sharding]]
