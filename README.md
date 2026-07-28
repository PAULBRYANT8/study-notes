# Study Notes

面向 **NPU 训练适配、PyTorch 内部机制与分布式并行** 的个人学习笔记库。

这里不追求堆积零散资料，而是把问题、概念、源码锚点和横向对比组织成可以持续生长的知识网络。目前重点覆盖：

- NPU 训练适配所需的 Linux 调试、C++ 阅读、Python 机制和数值基础
- PyTorch Tensor Parallel（TP）与 Context Parallel（CP）
- DTensor placement、模块切分、集合通信与负载均衡
- 训练精度、浮点误差、混合精度和显存估算

> 本仓库是个人学习记录，不是 PyTorch、华为昇腾或其他项目的官方文档。内容会随着学习和源码版本持续修订。

## 从哪里开始

| 入口 | 用途 |
|---|---|
| [index.md](index.md) | 浏览全部主题及一句话摘要 |
| [NPU 训练适配学习路径](queries/npu-training-adaptation-learning-path.md) | 从岗位能力出发查看学习顺序 |
| [SCHEMA.md](SCHEMA.md) | 了解目录、命名、双链和资料录入约定 |
| [log.md](log.md) | 查看知识库的更新记录 |

如果目标是准备 NPU 训练适配岗位，建议先读学习路径，再按下面的顺序进入五个基础专题：

1. [Linux 与调试：训练崩溃和性能问题定位](concepts/linux-debugging-for-npu-adaptation.md)
2. [面向 PyTorch 后端的 C++ 阅读基础](concepts/cpp-reading-for-pytorch-backends.md)
3. [PyTorch 相关的 Python 进阶机制](concepts/python-advanced-mechanisms-for-pytorch.md)
4. [深度学习训练中的数值基础](concepts/deep-learning-training-numerics.md)
5. [浮点误差与数值分析常识](concepts/floating-point-error-analysis.md)

## 推荐学习路径

### NPU 训练适配

```text
Linux 故障定位
    ↓
读懂 Python → C/C++ 扩展调用链
    ↓
读懂 ATen / torch_npu C++ 层
    ↓
理解混合精度、loss scale 与显存账本
    ↓
建立跨层定位精度和性能问题的能力
```

对应入口：

- [NPU 训练适配学习路径](queries/npu-training-adaptation-learning-path.md)
- [Linux 调试](concepts/linux-debugging-for-npu-adaptation.md)
- [C++ 阅读](concepts/cpp-reading-for-pytorch-backends.md)
- [Python 进阶](concepts/python-advanced-mechanisms-for-pytorch.md)
- [训练数值基础](concepts/deep-learning-training-numerics.md)
- [浮点误差分析](concepts/floating-point-error-analysis.md)

### PyTorch 分布式并行

建议先理解 DTensor 的布局语义，再分别学习 TP 和 CP：

```text
DTensor Placement
    ├── Tensor Parallel → 模块切分 → Colwise / Rowwise / Sequence Parallel
    └── Context Parallel → 序列切分 → Ring Attention → 负载均衡
```

关键入口：

- [DTensor Placement](concepts/dtensor-placement.md)
- [Tensor Parallel](concepts/tensor-parallel.md)
- [TP 模块切分](concepts/tp-module-sharding.md)
- [Context Parallel](concepts/context-parallel.md)
- [CP 序列切分](concepts/cp-sequence-sharding.md)
- [Ring Attention](concepts/ring-attention.md)
- [TP 与 CP 切分对比](comparisons/tp-vs-cp-sharding.md)
- [CP 负载均衡策略对比](comparisons/cp-load-balancers.md)

## 目录结构

| 目录或文件 | 内容 |
|---|---|
| `raw/` | 原始资料和源码锚点，只追加、不改写 |
| `entity/` | 可以明确指认的实现、工具、模块或策略 |
| `concepts/` | 原理、方法、机制和抽象概念 |
| `comparisons/` | 两个或多个方案的横向比较 |
| `queries/` | 从具体问题出发形成的结论和学习路径 |
| `index.md` | 全库导航，不承载正文 |
| `log.md` | 知识库更新记录 |
| `SCHEMA.md` | 知识库维护规范 |

## 使用方式

### 在 GitHub 上阅读

从 [index.md](index.md) 或上面的学习路径进入。README 使用普通 Markdown 链接，适合直接在 GitHub 中浏览；笔记正文中的 `[[双链]]` 主要供 Obsidian 使用。

### 在 Obsidian 中阅读

仓库保留了 `.obsidian/` 工作区设置，可以直接克隆并将根目录作为 Obsidian Vault 打开：

```bash
git clone https://github.com/PAULBRYANT8/study-notes.git
```

打开后可以通过双链、反向链接和关系图查看主题之间的联系。

## 笔记约定

- 一个文件只解决一个主要问题，文件名使用 `kebab-case`
- 第一段优先给出定义或结论，再展开证据和推导
- `raw/` 保存原始资料，其余目录保存提炼后的知识
- 新笔记需要回链相关旧笔记，并同步更新 `index.md` 和 `log.md`
- 不确定的信息明确标注，不为了完整而补写未经核实的内容
- 外部资料尽量在对应知识点附近给出链接，便于继续核对

完整规则见 [SCHEMA.md](SCHEMA.md)。
