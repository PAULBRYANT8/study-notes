# Index

本库的内容目录。只列条目和一句话钩子，不写内容。约定见 [[SCHEMA]]。

## Entities

### PyTorch TP

- [[colwise-parallel]] — Linear 权重按输出维切，输出 `Shard(-1)` 无需归约
- [[rowwise-parallel]] — 按输入维切，输出 `Partial`；bias 必须复制否则加 N 次
- [[sequence-parallel]] — norm/dropout 层参数复制、激活按序列维切，省冗余显存
- [[prepare-module-input-output]] — 不切参数，只在模块边界调布局
- [[loss-parallel]] — 词表维分片下直接算 cross entropy，不聚合 logits
- [[micro-pipeline-tp]] — inductor 把通信和 matmul 融合成流水，默认关闭

### PyTorch CP

- [[head-tail-load-balancer]] — CP 默认均衡策略，头尾对称配对抵消因果掩码的负载倾斜
- [[per-document-head-tail-load-balancer]] — 多文档打包场景的变体，rank-major 布局是修出来的
- [[ptrr-load-balancer]] — 从 BlockMask 测量实际负载再贪心调度，唯一能处理非因果掩码的

## Concepts

- [[linux-debugging-for-npu-adaptation]] — 用 gdb/core、pdb、strace、perf 和 dmesg 按层定位训练崩溃、hang 与 host 性能问题
- [[cpp-reading-for-pytorch-backends]] — 围绕模板、RAII、智能指针、宏、注册和动态链接读懂 ATen/torch_npu C++ 调用链
- [[python-advanced-mechanisms-for-pytorch]] — 装饰器、context、hook/override 协议及 Python 跨入 C/C++ 扩展的完整路径
- [[deep-learning-training-numerics]] — 反向传播、fp16/bf16、AMP、loss scaling 和参数/梯度/优化器/激活显存账本
- [[floating-point-error-analysis]] — 非结合律、误差累积、条件数、稳定算法，以及 rtol/atol 何时失效、如何组合
- [[dtensor-placement]] — Shard/Replicate/Partial 与集合通信的映射表，TP 和 CP 共同的底座
- [[tensor-parallel]] — 切权重矩阵，colwise 接 rowwise 中间零通信
- [[tp-module-sharding]] — 声明式：style 挂到模块上，通信由布局不匹配自动推导
- [[context-parallel]] — 沿序列维切分的并行方式，长序列训练的显存解法
- [[cp-sequence-sharding]] — 先重排后等分；切分是死的，灵活性全在重排索引里
- [[ring-attention]] — Q 固定、KV 轮转一圈，用 logsumexp 增量合并

## Comparisons

- [[tp-vs-cp-sharding]] — 切参数 vs 切序列；为什么只有 CP 需要负载均衡
- [[colwise-vs-rowwise-parallel]] — 必须配对使用，三处不对称容易踩坑
- [[cp-load-balancers]] — 三种 CP 均衡策略怎么选，整除条件的严格程度差很多
- [[allgather-vs-alltoall-kv-rotation]] — 用显存换通信次数，默认 allgather

## Queries

- [[how-does-pytorch-tp-shard-modules]] — PyTorch 的 TP 是怎么切分模块的？
- [[how-does-pytorch-cp-shard-sequences]] — PyTorch 的 CP 是怎么切分序列的？
- [[npu-training-adaptation-learning-path]] — NPU 训练适配要学什么、怎么学；护城河是跨层定位，不是会用框架

## Raw

- [[2026-07-26-pytorch-tp-source]] — PyTorch TP 源码锚点与关键片段
- [[2026-07-25-pytorch-cp-source]] — PyTorch CP 源码锚点与关键片段，commit a07d9d50489

## 待写

悬空的 `[[链接]]` 指向的选题，攒到这里：

- [[llm-wiki]] — 本库依据的知识库模式本身
- [[cann-stack]] — CANN 的分层：runtime / AscendCL / aclnn / GE / HCCL
- [[ascend-davinci-architecture]] — Cube/Vector/Scalar 与内存层级，NPU 利用率问题的根
- [[torch-dispatcher]] — DispatchKey 与 PrivateUse1，后端接入的落脚点
- [[hccl-vs-nccl]] — 两套集合通信库的差异，NPU 适配的必修差异点
- [[npu-precision-debugging]] — GPU 对 NPU 不对时的分层二分法与归因表
- [[ascend-c]] — 自定义算子开发：tiling 与 double buffer
- [[torchair]] — Ascend 上的图模式后端，与单算子模式的取舍
- [[pipeline-parallel]] — 1F1B、interleaved 与气泡率
- [[zero-and-fsdp]] — 三个阶段各省多少显存、各加多少通信
