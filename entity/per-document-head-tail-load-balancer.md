---
title: _PerDocumentHeadTailLoadBalancer
type: entity
created: 2026-07-25
updated: 2026-11-06
tags: [pytorch, context-parallel, 负载均衡, 文档掩码]
sources: ["[[2026-07-25-pytorch-cp-source]]"]
---

`_PerDocumentHeadTailLoadBalancer` 把 [[head-tail-load-balancer]] 的头尾配对**在每篇文档内部**各做一遍，用于多文档打包（document packing）场景。

位置：`_load_balancer.py:176-297`。

## 适用场景

训练时把多篇短文档拼进一条序列，掩码是分块对角的 document causal mask——每篇文档内部因果，文档之间互不可见：

```
[1, 0, 0, 0 | 0 0 0 0 ...]   doc 0 (len 4)
[1, 1, 0, 0 | ...        ]
[1, 1, 1, 0 | ...        ]
[1, 1, 1, 1 | ...        ]
[0 0 0 0 | 1, 0, ...     ]   doc 1 (len 8)
...
```

全局的头尾配对在这里失效：序列整体的"头"和"尾"分属不同文档，配起来的两块计算量不再互补。

## 算法

对每篇文档独立切成 `2 * world_size` 块并配对，得到 `(world_size, 2 * chunk_length)`——第 r 行是 rank r 在这篇文档里的头块+尾块。然后：

```python
indices_tensor = torch.cat(per_doc_rank_chunks, dim=1).reshape(-1)
```

**沿 dim=1 拼接**是关键：这样所有文档的 rank r 那一行拼在一起，flatten 后是 rank-major 布局，连续切分时 rank r 恰好拿到**每篇文档的头尾片段**。

构造函数接收 `seq_length_per_doc: list[list[int]]`，外层是 batch，内层是该样本的各文档长度。返回 `(B, seq_len)` 或 `(1, seq_len)`。

## rank-major 是修出来的

commit `53db8cca157`（[CP] Balance per-document head-tail load balancer with rank-major layout，#189902）修的就是这里。原先按 document-major 排布，切割线会落在文档中间，文档长度不齐时依然失衡。源码里现在留了解释性注释：

> A document-major layout would instead let the cut fall mid-document, imbalancing work for mixed-length docs.

这是 [[cp-sequence-sharding]] 里那条 sharding contract 的直接体现——切分逻辑写死为连续等分，均衡器必须自己保证连续块是均衡的。

## 约束

- **每篇文档**的长度都要能被 `2 * world_size` 整除，不只是总长度。短文档在这里很容易踩线
- 同样只对因果掩码有意义
- 多头场景要求各 head 掩码一致

## 混合长度示例与掩码边界

**推导**：两个 rank 打包长度分别为 4 和 8 的文档，全局位置分别为 `[0,1,2,3]` 和 `[4,5,6,7,8,9,10,11]`。每篇文档独立头尾配对后，rank-major 排列为 `[0,3,4,5,10,11, 1,2,6,7,8,9]`。连续切成两份后，每个 rank 从两篇文档分别取得 2、4 个位置；按各文档内的标准因果关系计算，每个 rank 的可见 QK 对数均为 $5+18=23$。

若把同样的逐文档结果按 document-major 排为 `[0,3,1,2,4,5,10,11,6,7,8,9]`，再每 6 个位置切一份，两个 rank 的可见对数分别为 13 和 33。该排列是解释 sharding contract 的示意，不声称复现了历史 bug 的全部代码。依据见 [[2026-07-25-pytorch-cp-source]] 中 per-document rank-major 注释。

**语义边界**：重排只决定哪些位置归哪个 rank，不会自行创建文档隔离掩码。如何在所选 attention 后端传递并执行 document-causal mask，需核对该版本完整实现与测试；原始摘录未展示此路径。

## 相关

- 基础版本：[[head-tail-load-balancer]]
- 对比：[[cp-load-balancers]]
- 同场景的另一选择：[[ptrr-load-balancer]]（flex_attention 时更通用）
