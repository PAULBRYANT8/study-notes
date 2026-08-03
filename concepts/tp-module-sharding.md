---
title: TP 模块切分机制
type: concept
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, dtensor, 模块]
sources: ["[[2026-07-26-pytorch-tp-source]]"]
---

TP 的切分是**声明式**的：用 `parallelize_module(module, mesh, plan)` 把 `ParallelStyle` 按模块 FQN 贴上去，每个 style 负责三件事——切参数、改输入布局、改输出布局。

对照 [[cp-sequence-sharding]]：CP 是命令式的，你手动把 buffer 传进 `_context_parallel_shard` 拿回本地张量；TP 是把规则挂到模块上，运行时由 forward hook 自动生效。

## ParallelStyle 契约

```python
class ParallelStyle(ABC):
    src_data_rank: int | None = 0

    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
```

只有一个抽象方法，刻意留得极宽松——[[context-parallel]] 的 `_ContextParallel` 也是 `ParallelStyle` 的子类，复用的正是这个口子。

内置的几个 style 都通过 `distribute_module()` 实现，它接收三个回调：

```python
distribute_module(
    module,
    device_mesh,
    partition_fn,       # 怎么切参数
    input_fn,           # forward 前调整输入布局
    output_fn,          # forward 后调整输出布局
)
```

`PrepareModuleInput/Output` 例外——它们不碰参数，直接注册 `forward_pre_hook` / `forward_hook`。

## 三段式布局协议

每个 style 内部维护三组 placement：

| 字段 | 含义 |
|---|---|
| `input_layouts` | **声明**进来的张量当前是什么布局（用于 `from_local` 打标签，不产生通信） |
| `desired_input_layouts` | 算子实际需要的布局，与 `input_layouts` 不同则 `redistribute`（**产生通信**） |
| `output_layouts` | 期望输出成什么布局，与算子自然产出不同则 `redistribute` |

`input_layouts` 是断言性质的：`DTensor.from_local(..., run_check=False)` 直接相信你的声明，不校验。**声明错了不报错，只会静默算错。**

`desired_input_layouts` 通常是策略写死的（colwise 恒为 `Replicate()`），只有 [[rowwise-parallel]] 会根据模块类型在 `_apply` 里动态设置。

## 参数切分

统一走 `distribute_tensor(param, mesh, [placement], src_data_rank=...)`，包成 `nn.Parameter` 后 `register_parameter` 换掉原参数。**参数从此是 DTensor**，优化器、checkpoint 都要能处理。

`SequenceParallel` 是唯一的例外：它用 `DTensor.from_local(param, mesh, [Replicate()])` 而不是 `distribute_tensor`，理由写在注释里——LayerNorm/RMSNorm 默认 ones 初始化，各卡本来就一样，不必广播。代价是**自定义初始化会静默失效**，各卡权重不一致，docstring 里专门警告了。

## src_data_rank

`parallelize_module(..., src_data_rank=0)`（默认）：以 rank 0 的参数为准，scatter/broadcast 给其他 rank，保持单卡语义——各卡随机初始化不同也没关系。

传 `None`：各 rank 直接用自己的本地数据切，不通信。适合已经保证各卡一致（比如从 checkpoint 加载）的场景，省一次启动开销。

这个参数会被 `parallelize_module` 写进 style 实例的 `src_data_rank` 属性再 `_apply`。

## FQN 匹配

plan 的 key 是模块路径，**支持 fnmatch 通配符**：

```python
parallelize_module(model, tp_mesh, {
    "layers.*.attn.wq": ColwiseParallel(),
    "layers.*.attn.wo": RowwiseParallel(),
    "layers.*.ffn_norm": SequenceParallel(),
})
```

匹配逻辑按 `.` 逐级递归，每级用 `fnmatch(child_name, token)`。空字符串 key `""` 表示应用到当前模块自身。

**匹配不上只会 warning 然后跳过，不报错。** 改了模型结构后 plan 失效，训练照常跑但没并行——这是最容易踩的坑，`warnings.warn` 很容易被淹没在日志里。

## 组合出一个 Transformer 层

典型 plan 的通信开销可以直接从布局衔接读出来：

```
输入 Shard(1)  ──SequenceParallel──►  Shard(1)
               ──PrepareModuleInput(desired=Replicate)──►  Replicate    [all_gather]
               ──ColwiseParallel(wq/wk/wv)──►  Shard(-1)                [无通信]
               ──attention──►  Shard(-1)
               ──RowwiseParallel(wo, output=Shard(1))──►  Shard(1)      [reduce_scatter]
```

衔接处布局一致就是零通信，不一致就查 [[dtensor-placement]] 的映射表。

## 相关

- 两种基本切法的对比：[[colwise-vs-rowwise-parallel]]
- 编译期把这些通信和 matmul 融合：[[micro-pipeline-tp]]
