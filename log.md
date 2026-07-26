# Log

时间正序，新记录**追加在末尾**（这样 diff 只有增量，不会整体位移）。

格式：`- YYYY-MM-DD 动作 —— [[文件]]、[[文件]]`

**提到的每个文件都必须写成 `[[链接]]`**，不能只写文件名。这样从任何一条日志都能直接跳到当时改动的内容，日志本身就是一条可导航的时间线。

---

- 2026-07-25 初始化知识库结构 —— [[SCHEMA]]、[[index]]、[[CLAUDE]]，以及 [[raw/README]]、[[entity/README]]、[[concepts/README]]、[[comparisons/README]]、[[queries/README]]
- 2026-07-25 log 格式改为强制 `[[链接]]`，SCHEMA 补同名文件的目录限定规则 —— [[SCHEMA]]、[[log]]
- 2026-07-25 录入 PyTorch CP 切分分析（源码存档 + 8 篇笔记）—— [[2026-07-25-pytorch-cp-source]]、[[context-parallel]]、[[cp-sequence-sharding]]、[[ring-attention]]、[[head-tail-load-balancer]]、[[per-document-head-tail-load-balancer]]、[[ptrr-load-balancer]]、[[cp-load-balancers]]、[[allgather-vs-alltoall-kv-rotation]]、[[how-does-pytorch-cp-shard-sequences]]、[[index]]
- 2026-07-26 录入 PyTorch TP 切分分析（源码存档 + 12 篇笔记）—— [[2026-07-26-pytorch-tp-source]]、[[tensor-parallel]]、[[tp-module-sharding]]、[[dtensor-placement]]、[[colwise-parallel]]、[[rowwise-parallel]]、[[sequence-parallel]]、[[prepare-module-input-output]]、[[loss-parallel]]、[[micro-pipeline-tp]]、[[colwise-vs-rowwise-parallel]]、[[tp-vs-cp-sharding]]、[[how-does-pytorch-tp-shard-modules]]、[[index]]
- 2026-07-26 回补 CP 笔记到 TP 的交叉链接 —— [[context-parallel]]、[[cp-sequence-sharding]]、[[how-does-pytorch-cp-shard-sequences]]
- 2026-07-26 写入 NPU 训练适配方向的学习路线，并把其中提到的 9 个未写主题登记进待写 —— [[npu-training-adaptation-learning-path]]、[[index]]
- 2026-07-26 补全 NPU 训练适配的五项前置基础，并回链学习路线 —— [[linux-debugging-for-npu-adaptation]]、[[cpp-reading-for-pytorch-backends]]、[[python-advanced-mechanisms-for-pytorch]]、[[deep-learning-training-numerics]]、[[floating-point-error-analysis]]、[[npu-training-adaptation-learning-path]]、[[index]]、[[log]]
