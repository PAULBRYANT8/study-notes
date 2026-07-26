---
title: PyTorch Context Parallel 源码存档（切分部分）
type: raw
created: 2026-07-25
updated: 2026-07-25
tags: [pytorch, context-parallel, 分布式, 源码]
sources:
  - 本地仓库 e:\浏览器下载\工作\pytorch
  - commit a07d9d50489 (2026-07-25), version.txt = 2.14.0a0
---

从本地 PyTorch 仓库读取。全部代码位于 `torch/distributed/tensor/experimental/_context_parallel/`。

## 模块清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `_attention.py` | 1728 | ring attention、rotater、切分入口、dispatcher、flex 支持 |
| `_load_balancer.py` | 483 | 三种负载均衡策略 + 默认策略工厂 |
| `_sharding_rules.py` | 338 | DTensor 分片规则 |
| `_cp_custom_ops.py` | 88 | `flex_cp_allgather` 自定义算子 |
| `__init__.py` | 46 | 导出 |

`torch/distributed/tensor/experimental/_attention.py`（44 行）现在只是 BC stub，全部转发到 `_context_parallel/`。

## 关键位置锚点

- `_LoadBalancer` 抽象基类与 sharding contract：`_load_balancer.py:12-84`
- `_HeadTailLoadBalancer`：`_load_balancer.py:87-173`
- `_PerDocumentHeadTailLoadBalancer`：`_load_balancer.py:176-297`
- `_PTRRLoadBalancer` / `ptrr_scheduling`：`_load_balancer.py:300-472`
- `_create_default_load_balancer`：`_load_balancer.py:475-483`
- `_ContextParallelOptions` / `_cp_options`：`_attention.py:70-80`
- `_is_causal_behavior`：`_attention.py:83-100`
- `_AllToAllRotater` / `_AllGatherRotater`：`_attention.py:253-314`
- `_templated_ring_attention`：`_attention.py:317-488`
- `_context_parallel_buffers`（实际执行切分）：`_attention.py:1074-1181`
- `_create_cp_block_mask`：`_attention.py:1184-1325`
- `_ContextParallel`（ParallelStyle）：`_attention.py:1327-1425`
- `_context_parallel_shard`（新版入口）：`_attention.py:1433-1503`
- `context_parallel`（旧版 context manager）：`_attention.py:1523-1616`
- `context_parallel_unshard`：`_attention.py:1619-1700`

测试：`test/distributed/tensor/test_attention.py`

## 关键片段（逐字）

### sharding contract（`_load_balancer.py:76-83`）

```
Sharding contract:
    After rearranging, Context Parallel shards `Q[rearrange_idx]` with
    `Shard(seq_dim)`, which follows `torch.chunk` semantics: the sequence is
    split into `world_size` contiguous, equal-sized chunks and chunk `r` is
    assigned to rank `r`. Implementations must therefore lay out
    `rearrange_idx` so that each such contiguous chunk is well-balanced --
    i.e. group each rank's positions together (rank-major), not interleaved
    by any other axis such as document.
```

### head-tail 索引生成（`_load_balancer.py:155-173`）

```python
seq_length = self.seq_length
world_size = self.world_size
if seq_length % (world_size * 2) != 0:
    raise AssertionError
chunk_size = seq_length // (world_size * 2)

# Split sequence into 2*world_size chunks, then pair chunk r with
# chunk (2*world_size - 1 - r) for each rank.
indices = torch.arange(seq_length, dtype=torch.int, device=self.device)
chunks = indices.view(world_size * 2, chunk_size)
head_idx = torch.arange(world_size, device=self.device)
tail_idx = 2 * world_size - 1 - head_idx
paired = torch.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
all_indices_tensor = paired.reshape(-1)

if restore:
    all_indices_tensor = torch.argsort(all_indices_tensor)

return all_indices_tensor.unsqueeze(0)  # add batch dim
```

### 实际切分（`_attention.py:1161-1165`）

```python
# use DTensor to shard the buffer on sequence dimension,
# retain the local tensor
sharded_buffer = distribute_tensor(
    buffer, mesh, [Shard(seq_dim)], src_data_rank=None
).to_local()
```

### 因果块跳过判定（`_attention.py:83-100`）

```python
def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP
```

### 全局状态副作用（`_attention.py:1466-1475`）

```python
# TODO: these global variables are going to bite us someday.
# We will have to remove them soon.
# For the new API, we only support the module wrapper mode.
global _dispatch_mode
_dispatch_mode = _DispatchMode.MODULE_WRAPPER
global _cp_options
if load_balancer is not None:
    _cp_options.enable_load_balance = True
else:
    _cp_options.enable_load_balance = False
```

### per-document rank-major 布局的理由（`_load_balancer.py:268-274`）

```
# For each document, split it into 2 * world_size equal chunks and pair
# chunk r with chunk (2 * world_size - 1 - r), so row r holds rank r's
# head+tail chunks (the same strategy as _HeadTailLoadBalancer). Stacking
# the per-document rows rank-wise (dim=1) and flattening lays the indices
# out rank-major, so a contiguous per-rank shard gives each rank a
# head+tail slice of every document. A document-major layout would instead
# let the cut fall mid-document, imbalancing work for mixed-length docs.
```

## 模块近期提交

```
1544350488a Fix typos in comments and docstrings across torch modules (#190477)
53db8cca157 [CP] Balance per-document head-tail load balancer with rank-major layout (#189902)
f9675d4d180 [distributed] Fix max_seqlen mismatch in ring attention backward (#185493)
```
