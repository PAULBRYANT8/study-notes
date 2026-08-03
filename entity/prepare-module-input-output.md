---
title: PrepareModuleInput / Output / InputOutput
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, 布局, hook]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

这三个 `ParallelStyle` **不切任何参数**，只在模块边界处把张量转成 DTensor 并调整布局。用于把 [[colwise-parallel]] / [[rowwise-parallel]] / [[sequence-parallel]] 拼接起来的地方。

位置：`style.py:442-604`（Input）、`607-714`（Output）、`717-823`（InputOutput）。

## 为什么需要它们

Colwise/Rowwise 只管自己那一层的输入输出。但真实模型里，模块的边界往往在**上层容器**上——比如整个 `TransformerBlock` 收到 `Shard(1)` 的输入，需要在进 attention 之前 all_gather 成 `Replicate`。这个位置没有权重可切，只能靠 hook 改布局。

## PrepareModuleInput

| 参数 | 说明 |
|---|---|
| `input_layouts` | 声明每个位置参数当前的布局，非张量或不需要转换的位置填 `None` |
| `desired_input_layouts` | 目标布局，长度必须与上面一致 |
| `input_kwarg_layouts` | 关键字参数版本，dict |
| `desired_input_kwarg_layouts` | 同上 |
| `use_local_output` | 默认 `False` |

```python
PrepareModuleInput(
    input_layouts=(Shard(1), None, None),
    desired_input_layouts=(Replicate(), None, None),
)
```

`None` 占位是必要的——`inputs` 和 `input_layouts` 长度必须严格相等，否则 `ValueError`。

实现上注册 `forward_pre_hook`；传了 `input_kwarg_layouts` 才走 `with_kwargs=True` 的版本。

## PrepareModuleOutput

结构对称，用 `output_layouts` / `desired_output_layouts`，注册的是 `forward_hook`。

## PrepareModuleInputOutput

两者的组合，省掉在 plan 里为同一个模块写两条目。内部就是依次 `_apply` 两个 style。

## 值得注意的点

**`input_layouts` 是断言，不是转换。** 走的是 `DTensor.from_local(..., run_check=False)`——只贴标签，不校验实际数据是否真的那样分布。声明错了不会报错，只会算错。真正产生通信的是 `input_layouts != desired_layout` 时的 `redistribute`。

**已经是 DTensor 的输入不校验布局。** 源码里那行断言被注释掉了：

```python
if isinstance(input, DTensor):
    # TODO: re-enable the check once we fix the compile path
    # assert inp.placements[0] == input_layout
    dt_inp = input
```

也就是说传进来的 DTensor 布局和你声明的 `input_layouts` 不一致时，**声明会被静默忽略**，直接拿实际布局去和 `desired_layout` 比。这在混用编译路径时容易造成困惑。

**`use_local_output` 默认 `False`**，和 Colwise/Rowwise 的 `True` 相反。这里默认保持 DTensor 是对的——这些 style 的存在意义就是维持布局链路。

## 相关

- 上层机制：[[tp-module-sharding]]
- 布局与通信的对应：[[dtensor-placement]]
