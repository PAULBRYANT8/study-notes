---
title: PyTorch Tensor Parallel 源码存档（切分部分）
type: raw
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, 分布式, 源码]
sources:
  - 本地仓库 e:\浏览器下载\工作\pytorch
  - commit a07d9d50489 (2026-07-25), version.txt = 2.14.0a0
---

从本地 PyTorch 仓库读取。主体位于 `torch/distributed/tensor/parallel/`，底层依赖 `torch/distributed/tensor/` 的 DTensor placement 机制，编译期优化在 `torch/_inductor/fx_passes/micro_pipeline_tp.py`。

## 模块清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `parallel/style.py` | 823 | `ParallelStyle` 及 6 个内置策略 |
| `parallel/loss.py` | 614 | `loss_parallel` |
| `parallel/fsdp.py` | 400 | TP + FSDP 组合 |
| `parallel/api.py` | 142 | `parallelize_module` |
| `parallel/input_reshard.py` | 107 | 输入重分片 |
| `parallel/ddp.py` | 105 | TP + DDP 组合 |
| `parallel/_data_parallel_utils.py` | 51 | |
| `_inductor/fx_passes/micro_pipeline_tp.py` | 1326 | async TP 融合 pass |

公开导出（`parallel/__init__.py`）：`parallelize_module`、`loss_parallel`、`ParallelStyle`、`ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`、`PrepareModuleInput`、`PrepareModuleOutput`、`PrepareModuleInputOutput`。

**与 CP 不同，TP 的 API 全部是公开的、非 prototype 的。**

## 关键位置锚点

- `ParallelStyle` 基类：`parallel/style.py:31-42`
- `ColwiseParallel`：`parallel/style.py:45-183`
- `RowwiseParallel`：`parallel/style.py:186-336`
- `SequenceParallel`：`parallel/style.py:339-439`
- `PrepareModuleInput`：`parallel/style.py:442-604`
- `PrepareModuleOutput`：`parallel/style.py:607-714`
- `PrepareModuleInputOutput`：`parallel/style.py:717-823`
- `parallelize_module`：`parallel/api.py:14-142`
- `loss_parallel`：`parallel/loss.py:31-`
- `Shard` / `_StridedShard` / `Replicate` / `Partial` / `_MaskPartial`：`placement_types.py:162 / 799 / 1673 / 1738 / 1868`
- 集合通信映射：`_redistribute.py:191-195`
- `micro_pipeline_tp_pass`：`micro_pipeline_tp.py:1299-1326`
- 调用点：`_inductor/fx_passes/post_grad.py:270`
- 配置开关：`_inductor/config.py:1231`，`_micro_pipeline_tp: bool = False`

## 关键片段（逐字）

### Colwise 的参数切分（`style.py:121-143`）

```python
def _partition_linear_fn(self, name, module, device_mesh):
    # colwise shard weight/bias to Shard(0), weight be Shard(0)
    # means Colwise as Linear is input * weight^T + bias, where
    # weight would become Shard(1)
    for name, param in module.named_parameters():
        dist_param = nn.Parameter(
            distribute_tensor(
                param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank
            ),
            requires_grad=param.requires_grad,
        )
        module.register_parameter(name, dist_param)

def _partition_embedding_fn(self, name, module, device_mesh):
    # colwise shard embedding.weight is straight forward as Shard(1)
    ...
    distribute_tensor(param, device_mesh, [Shard(1)], ...)
```

### Rowwise 的参数切分（`style.py:250-279`）

```python
def _partition_linear_fn(self, name, module, device_mesh):
    # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
    # means Rowwise as nn.Linear is input * weight^T + bias, where
    # weight would become Shard(0)
    module.register_parameter(
        "weight",
        nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)], ...)),
    )
    if getattr(module, "bias", None) is not None:
        module.register_parameter(
            "bias",
            nn.Parameter(distribute_tensor(module.bias, device_mesh, [Replicate()], ...)),
        )
```

### Rowwise 的输出归约（`style.py:292-300`）

```python
@staticmethod
def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
    # Rowwise sharding produces partial output, depending on output layouts:
    # 1. to replicate -> allreduce
    # 2. to shard -> reduce_scatter
    if outputs.placements != output_layouts:
        outputs = outputs.redistribute(placements=output_layouts, async_op=True)
    return outputs.to_local() if use_local_output else outputs
```

### Rowwise 的 desired_input 因模块类型而异（`style.py:302-314`）

```python
def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    if isinstance(module, nn.Linear):
        partition_fn = self._partition_linear_fn
        # rowwise linear runtime sharding requires input tensor shard on last dim
        self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
    elif isinstance(module, nn.Embedding):
        partition_fn = self._partition_embedding_fn
        # rowwise embedding runtime sharding requires input tensor replicated
        self.desired_input_layouts = (Replicate(),)
    else:
        raise NotImplementedError(
            "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
        )
```

### SequenceParallel 用 from_local 复制参数（`style.py:389-398`）

```python
def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
    for p_name, param in module.named_parameters():
        # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
        # us to simply just use from_local
        replicated_param = torch.nn.Parameter(
            DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
        )
        module.register_parameter(p_name, replicated_param)
```

### FQN 匹配支持通配符（`api.py:99-114`）

```python
matched_children = list(
    filter(
        # `t[0]` is child name
        lambda t: fnmatch(t[0], token),
        module.named_children(),
    )
)
if not matched_children:
    # No match at this level. Log a warning and process next plan entry.
    warnings.warn(
        f"Parallelize plan key '{module_path}' could not be resolved: "
        f"no submodule matching token '{token}' in module {module}, "
        f"skipping this plan entry.",
        stacklevel=2,
    )
    continue
```

### Partial 的线性归约算子（`placement_types.py:1739-1742`）

```python
class Partial(torch._C._distributed.Partial):
    # reduce_ops that distribute over addition, enabling per-input linearity
    # for bilinear ops like mm: reduce_op(A_i @ B) = reduce_op(A_i) @ B
    LINEAR_REDUCE_OPS: tuple[str, ...] = ("sum", "avg")
    ALL_REDUCE_OPS: tuple[str, ...] = ("sum", "avg", "min", "max", "product")
```

### _MaskPartial 用于 rowwise embedding（`placement_types.py:1868-1876`）

```
A partial mask placement devised for rowwise sharded embedding op, where we need
to mask and adjust the indices to the local embedding shard, embedding masking
is a special type of the Partial placement
```

### async TP pass 入口（`micro_pipeline_tp.py:1299-1326`）

```python
def micro_pipeline_tp_pass(graph: torch.fx.Graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

    # When a collective can be hidden through either simple overlapping or
    # micro-pipeline TP, we prefer simple overlapping to avoid the overhead
    # associated with decomposition. If reorder_for_compute_comm_overlap is
    # enabled, we identify collectives that can be hidden through simple
    # overlapping and exclude them from micro-pipeline TP candidates.
    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        all_gathers = [x for x in all_gathers if x.ag_node not in unexposed_collectives]
        reduce_scatters = [
            x for x in reduce_scatters
            if x.reduce_scatter_node not in unexposed_collectives
        ]

    if not all_gathers and not reduce_scatters:
        log.warning(
            "async TP found no matching all-gather/reduce-scatter patterns for fusion"
        )

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)

    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)
```

融合目标算子：`torch.ops.symm_mem.fused_all_gather_matmul`、`fused_all_gather_scaled_matmul`、`fused_matmul_reduce_scatter`、`fused_scaled_matmul_reduce_scatter`。
