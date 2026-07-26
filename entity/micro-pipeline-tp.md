---
title: micro_pipeline_tp（async TP）
type: entity
created: 2026-07-26
updated: 2026-07-26
tags: [pytorch, tensor-parallel, inductor, 通信重叠, symm-mem]
sources: [[[2026-07-26-pytorch-tp-source]]]
---

`micro_pipeline_tp_pass` 是 inductor 的一个 post-grad FX pass，把 TP 产生的 **集合通信 + matmul** 融合成一个算子，让通信按 micro-batch 切成小块与计算流水重叠。俗称 async TP。

位置：`torch/_inductor/fx_passes/micro_pipeline_tp.py`（1326 行），调用点 `_inductor/fx_passes/post_grad.py:270`。

## 解决什么问题

[[tensor-parallel]] 的通信在关键路径上：`all_gather` 必须全部完成才能开始 matmul，`matmul` 必须全部完成才能 `reduce_scatter`。这段时间 GPU 的计算单元是闲的。

async TP 把它拆成流水：先 all_gather 第一块 → 边算第一块的 matmul 边 gather 第二块 → 以此类推。通信被计算掩盖掉。

## 匹配的两个模式

| 模式 | 融合成 |
|---|---|
| `all_gather` → `matmul` | `torch.ops.symm_mem.fused_all_gather_matmul` |
| `matmul` → `reduce_scatter` | `torch.ops.symm_mem.fused_matmul_reduce_scatter` |

fp8 有对应的 scaled 变体：`fused_all_gather_scaled_matmul`、`fused_scaled_matmul_reduce_scatter`。

对应到 TP plan 里就是：`all_gather → matmul` 是 [[colwise-parallel]] 前面那次 gather，`matmul → reduce_scatter` 是 [[rowwise-parallel]] 输出改 `Shard` 时那次 scatter（[[sequence-parallel]] 的典型配置）。

## 入口逻辑

```python
def micro_pipeline_tp_pass(graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        # 已经能被简单重叠掩盖的集合通信，排除掉
        ...

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)
    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)
```

**优先级设计值得注意**：如果 `reorder_for_compute_comm_overlap` 已经开启，那些靠简单重排就能藏起来的集合通信会被**排除**在 async TP 之外。注释说明理由——简单重叠没有分解开销，能用就优先用，async TP 是兜底手段。

找不到任何可融合模式时会 `log.warning`，这是排查"开了 async TP 但没提速"的第一个信号。

## 启用条件

**默认关闭**：`torch._inductor.config._micro_pipeline_tp = False`（`config.py:1231`）。下划线前缀说明还是内部配置。

运行时还要求：

- **Symmetric Memory 对该进程组可用**（`is_symm_mem_enabled_for_group`），否则跳过融合。这是硬约束——融合算子依赖 symm mem 做跨卡直访
- 输入张量需要 restride（`restride_A_shard_for_fused_all_gather_matmul` / `restride_A_for_fused_matmul_reduce_scatter`），pass 会自动插入
- `fused_matmul_reduce_scatter` **不返回 matmul 的中间结果**，所以 matmul 结果有多个 consumer 时会放弃融合并打日志

## 相关

- 被优化的对象：[[tp-module-sharding]] 里布局衔接处产生的通信
- 通信从哪来：[[dtensor-placement]]
