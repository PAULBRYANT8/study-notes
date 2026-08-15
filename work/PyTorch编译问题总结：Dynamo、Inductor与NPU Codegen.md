# PyTorch 编译问题总结：Dynamo、Inductor 与 NPU Codegen

Dynamo backend 负责把 FX 图交给某个编译器，Inductor 内部的 NPU Codegen 负责把 Inductor IR 生成 NPU Kernel；`decomposition` 和 `component` 分别属于图变换层和训练编排层。

> 这份文档用于记录工作中遇到的 PyTorch 编译相关问题。
>
> 本文基于 `torchtitan-npu` 本地 `master=2f056d6`，以及关键迁移提交
> `4cbf637 [fix] Use bundled AscendC backend for torch.compile` 整理。

相关工作记录：[[swiglu-group-接入复盘]]

## 一、当前 master 与旧 `inductor_npu_ext` 的 torch.compile 方式

### 1. 完整调用链

当前 NPU 编译链是分层的：

```text
Python 模型/函数
    ↓
torch.compile(backend="inductor")
    ↓
Dynamo：捕获 Python，生成 FX Graph 和 guards
    ↓
AOTAutograd：处理 forward/backward，并应用 decomposition
    ↓
Inductor：图优化、lowering、调度、融合
    ↓
npu_backend="ascendc"
    ↓
AscendC / AutoFuse：生成 NPU Kernel 和 runtime wrapper
```

需要区分两个 backend：

```python
torch.compile(
    fn,
    backend="inductor",
    options={"npu_backend": "ascendc"},
)
```

- `backend="inductor"`：Dynamo 层的编译后端，表示把 Dynamo 生成的 FX 图交给 Inductor。
- `npu_backend="ascendc"`：Inductor 内部的设备代码生成后端，表示 NPU 节点使用 AscendC Codegen。

因此当前 master 仍然是：

```text
Dynamo → Inductor → AscendC NPU Codegen
```

不是 `Dynamo → AscendC`，也不是通常意义上的：

```python
torch.compile(fn, backend="ascendc")
```

### 2. Dynamo 编译后端

`torch.compile(..., backend="inductor")` 中的 `backend` 属于 Dynamo 层。

Dynamo 负责：

- 捕获 Python 代码中的 Tensor 运算；
- 将 Python 代码转换为 FX `GraphModule`；
- 处理 graph break；
- 记录 shape、dtype、设备和常量等 guards；
- 在运行条件变化时触发重新编译；
- 调用指定的编译后端。

从接口角度看，Dynamo backend 大致接收：

```python
backend(gm: torch.fx.GraphModule, example_inputs) -> callable
```

它面对的是“整张 FX 图”，最终返回一个可调用的编译函数。`inductor`、`eager`、
`aot_eager`、`cudagraphs` 和自定义 callable 都属于这一层的 backend。

### 3. Inductor 内部的 NPU Codegen 后端

Inductor 接收 FX 图后，会继续做：

```text
FX Graph
   ├─ decomposition
   ├─ lowering 到 Inductor IR
   ├─ 图优化
   ├─ scheduler 调度
   └─ fusion 融合
```

然后根据设备选择代码生成器：

- CPU：C++ Codegen；
- CUDA：Triton/CUDA Codegen；
- NPU：AscendC Codegen。

NPU Codegen 负责 NPU 节点的 lowering、调度、融合、AscendC Kernel、host/runtime wrapper，
以及部分 NPU 特有的 layout、dtype、shape 和 SoC 约束。它看到的不是原始 Python 函数，
而是 Inductor IR、调度节点和设备信息。

### 4. 最简单的例子

```python
import torch

def foo(x):
    y = torch.relu(x)
    return y + 1

compiled_foo = torch.compile(
    foo,
    backend="inductor",
    options={"npu_backend": "ascendc"},
)
```

#### 第一步：Dynamo 生成 FX 图

Dynamo 可能捕获成：

```text
输入 x
  ↓
aten.relu(x)
  ↓
aten.add(relu_result, 1)
  ↓
输出
```

Dynamo 此时只回答：

```text
这张 FX 图交给哪个编译器？
```

因为配置了 `backend="inductor"`，所以它调用 Inductor backend。

#### 第二步：Inductor 生成 NPU Kernel

Inductor lowering 后可能得到：

```text
tmp0 = relu(x)
tmp1 = tmp0 + 1
```

因为配置了 `npu_backend="ascendc"`，Inductor 才会使用 NPU AscendC Codegen。它可能把
`relu + add` 融合成一个 Kernel，概念上类似：

```cpp
for (int i = 0; i < numel; ++i) {
    output[i] = max(input[i], 0) + 1;
}
```

这个例子的完整路径是：

```text
foo 的 Python 代码
    ↓
Dynamo 捕获 FX Graph
    ↓  backend="inductor"
Inductor 编译整张图
    ↓  npu_backend="ascendc"
AscendC 生成 NPU Kernel
```

### 5. 三种配置的对比

```python
# Inductor + AscendC
torch.compile(foo, backend="inductor",
              options={"npu_backend": "ascendc"})
```

```text
Dynamo → Inductor → NPU Scheduling → AscendC Codegen → NPU Kernel
```

```python
# Dynamo + eager
torch.compile(foo, backend="eager")
```

```text
Dynamo → eager 执行 FX 图
```

不会进入 Inductor，也不会生成 AscendC Kernel。

如果在 CPU 上使用 `backend="inductor"`，路径可能是：

```text
Dynamo → Inductor → CPU Scheduling → C++ Codegen
```

所以 `backend="inductor"` 只表示使用 Inductor，不等于使用 AscendC。

### 6. 与旧 `inductor_npu_ext` 的关系

旧方式通常是：

```python
import inductor_npu_ext
compiled_foo = torch.compile(foo, backend="inductor")
```

旧扩展 import 时，会向 Inductor 注册 NPU Scheduling、NPU Wrapper Codegen、NPU lowering、
decomposition 和 scheduler/fusion 规则，核心逻辑类似：

```python
register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)
```

所以旧方式中 Dynamo 的 backend 仍然是 `inductor`；`inductor_npu_ext` 只是把 NPU Codegen
注册到了 Inductor 内部。

当前 master 的 `4cbf637` 将其迁移到 `torch_npu` 内置的 `torch_npu/_inductor/ascendc`，
通过 `inductor_config.npu_backend = "ascendc"` 或 `options={"npu_backend": "ascendc"}` 选择。
变化的是 NPU Codegen 的归属、安装和激活方式，整体调用链仍是：

```text
Dynamo → Inductor → NPU Codegen
```

## 二、decomposition 是什么

### 1. 定义

`decomposition` 是“算子分解”或“算子展开”：

> 将一个高层、复杂或当前后端不直接支持的算子，改写成多个语义等价的基础算子。

例如：

```text
aten.silu(x)
```

可以概念性地展开为：

```text
neg_x = -x
exp_x = exp(neg_x)
denom = 1 + exp_x
y = x / denom
```

`native_layer_norm` 可以展开为 `mean、variance、sub、rsqrt、mul、add`；泛化的 `matmul`
可以按输入 rank 规范化为 `mm` 或 `bmm`。具体规则由 PyTorch/Inductor/NPU backend 的
decomposition table 决定。

### 2. 它在编译链中的位置

```text
Dynamo 生成 FX 图
    ↓
AOTAutograd 创建 forward/backward 图
    ↓
应用 decomposition table
    ↓
Inductor lowering 到 IR
    ↓
调度、融合和 Codegen
```

decomposition 不是最终 Kernel 生成，而是对图进行改写。

### 3. 与其他术语的区别

| 概念 | 作用 |
|---|---|
| decomposition | 一个高层算子改写成多个等价算子 |
| lowering | 将算子映射到 Inductor IR 或具体 Kernel 实现 |
| fusion | 将多个 IR 节点合并成一个 Kernel |
| fallback | 后端无法处理时，保留原算子交给 eager/extern kernel |

例如：

```text
silu
  ↓ decomposition
neg + exp + add + div
  ↓ lowering
Inductor IR
  ↓ fusion
一个 AscendC 融合 Kernel
```

分解成多个算子并不意味着运行时一定执行多个 Kernel，后续 Inductor 仍可能将它们融合。

### 4. 为什么需要，代价是什么

常见原因：

1. NPU backend 没有某个高层算子的直接 lowering，但支持它依赖的基础算子；
2. 展开复合算子后，可以暴露更多 pointwise 融合机会；
3. 将泛化算子规范化为后端已经支持的标准算子。

代价包括图节点、编译时间和中间 tensor 增加，可能失去硬件原生 fused operator，或者带来
dtype、精度、layout 和动态 shape 方面的变化。因此 NPU backend 可能按 shape、dtype、layout
和 SoC 条件决定是否分解。

旧 `inductor_npu_ext` 曾经对白名单 decomposition 做定制，例如 `_to_copy`、`silu`、`sgn`、
`floor_divide`、`matmul`、`clamp`、`cat`、`zeros/ones/full`，并对 `_softmax`、
`native_layer_norm` 做带 shape 条件的 decomposition。当前 master 主要由 `torch_npu` 内置
AscendC backend 管理这些 NPU-specific 规则。

模型代码中把 SwiGLU 写成：

```python
gate, up = h.chunk(2, dim=-1)
return torch.nn.functional.silu(gate) * up
```

这是模型算法层面的拆分；Inductor decomposition 则是编译器在 ATen/FX 图上进行的算子替换，
两者属于不同层次。

## 三、lowering 是什么

### 1. 定义：把 FX/ATen 节点翻译成 Inductor IR

`lowering` 可以理解为“降低抽象层次”。Dynamo 捕获到的 FX 图仍然接近 ATen 算子图，
例如 `aten.add.Tensor`、`aten.mm`、`aten.native_layer_norm`。Inductor 需要把这些节点翻译
成自己能够调度和优化的中间表示（Inductor IR），这一步就是 lowering。

从实现角度看，Inductor 的 `GraphLowering` 会遍历 FX 图的节点，并为每个节点查找已注册的
lowering 函数。lowering 的输入通常不是普通的 `torch.Tensor`，而是描述值、形状、布局和
设备信息的 `TensorBox`；输出也通常是 `TensorBox`、`StorageBox` 或其他 IR 节点。也就是说，
它并不是马上执行算子，而是在“构建一份未来如何执行的计划”。

```text
FX/ATen 节点
    ↓ 查找对应 lowering
规范化输入：dtype、device、shape、layout、stride
    ↓
创建 Inductor IR：Pointwise / Reduction / ExternKernel / 模板节点
    ↓
Scheduler 分析依赖、布局和并行度
    ↓
Fusion、代码生成和 runtime wrapper
```

官方源码中的核心关系可以概括为：Inductor IR 是执行 lowering 代码产生的；每个 lowering
通常注册到某个 ATen 算子，并接收符合 ATen schema 的输入。

### 2. lowering 和 decomposition、Codegen 的边界

这几个词处在不同阶段，不能混用：

| 阶段 | 输入 | 输出 | 主要问题 |
|---|---|---|---|
| decomposition | 一个高层/复合算子 | 多个语义等价的 ATen 算子 | 这个算子能否改写成后端认识的基础算子？ |
| lowering | 一个 ATen/FX 节点 | Inductor IR 或 extern/template 节点 | 这个节点在 Inductor 中如何表达和调度？ |
| scheduling/fusion | 多个 IR 节点 | 可执行的调度组 | 哪些节点放在同一个 kernel，采用什么布局和并行策略？ |
| Codegen | 调度组和设备信息 | AscendC kernel、host wrapper、launch 逻辑 | 如何生成目标 NPU 可执行代码？ |

例如 `silu` 的处理可能是：

```text
aten.silu(x)
    ↓ decomposition
neg(x) + exp(x) + add(1) + reciprocal/mul
    ↓ lowering
多个 Pointwise IR 节点
    ↓ scheduler/fusion
一个融合的计算组
    ↓ AscendC Codegen
一个 NPU Kernel
```

如果后端本身有高效的 `silu` 原生实现，也可能不做 decomposition，而是直接把
`aten.silu` lowering 成一个 extern/template 节点。前一种方案通常更容易复用通用融合能力，
后一种方案可能保留硬件厂商提供的融合指令或更好的数值实现。

### 3. 常见 lowering 类型

#### 3.1 Pointwise lowering

加法、乘法、比较、`relu`、`exp` 等逐元素操作通常可以 lowering 成 Pointwise IR。Pointwise
IR 记录索引映射和表达式，后续 scheduler 可以将多个逐元素节点合并，减少 kernel launch
和中间 tensor。

#### 3.2 Reduction lowering

`sum`、`mean`、`amax`、`softmax` 中的归约部分需要表达归约维度、初始值、累积 dtype 和
输出布局。NPU backend 还要考虑归约轴是否连续、是否对齐、是否支持动态长度，以及是否需要
临时 workspace。

#### 3.3 矩阵/模板/Extern lowering

`mm`、`bmm`、`matmul`、卷积、attention 等算子可能被 lowering 成矩阵乘模板、专用模板或
`ExternKernel`。这类节点通常由 NPU 厂商库、AscendC 模板或 AutoFuse 负责具体实现，
Inductor 主要负责参数、布局、依赖和调用包装。

#### 3.4 Layout、copy 和 device-specific lowering

`to`、`_to_copy`、reshape、view、contiguous 等操作不一定产生真正的计算 kernel，但会影响
存储布局、stride、dtype 和设备转换。NPU lowering 需要判断转换是否可以消除、是否能与前后
算子融合，以及是否会引入额外的搬运 kernel。

### 4. 为什么 lowering 很容易成为 NPU 编译问题的根源

一个算子“语义上支持”并不等于“可以成功 lowering”。lowering 还必须满足以下约束：

- **dtype**：例如 `float16`、`bfloat16`、`float32`、`int8` 的输入和累积精度组合是否被支持；
- **shape**：维度是否静态、是否为 0/1、是否满足硬件 tile 对齐，动态符号能否传递到模板；
- **layout/stride**：输入是否 contiguous，转置视图是否支持，输出 layout 是否能被下游消费；
- **device**：所有输入和常量是否位于 NPU，是否混入 CPU scalar 或 CPU tensor；
- **alias/mutation**：原地写、视图、存储别名是否能安全表示；
- **版本和 SoC**：某个 AscendC 指令、模板或 runtime API 是否只在特定 CANN/芯片版本存在。

因此日志中出现 `LoweringException`、`NotImplementedError` 或模板选择失败时，问题通常发生
在“ATen 节点 → Inductor IR/Extern 节点”的阶段，不能直接归因于最终 AscendC kernel 的运行时
错误。反过来，lowering 成功但 kernel 编译失败，则应继续检查 AscendC 编译器、头文件、CANN
版本和生成的 wrapper。

### 5. lowering 问题的定位方法

遇到某个算子编译失败时，可以按下面的顺序缩小范围：

1. 从错误栈中确认具体 `target`，例如 `aten.foo.default`，不要只看最外层的
   `torch.compile` 异常；
2. 记录该节点的输入 shape、dtype、device、layout/stride 以及动态 shape 约束；
3. 分别测试 eager、`backend="eager"`、普通 Inductor 和 `npu_backend="ascendc"`，判断问题
   出现在 Dynamo、Inductor 通用 lowering 还是 NPU-specific lowering；
4. 临时打开或关闭对应 decomposition，比较“直接 lowering 原算子”和“分解后 lowering 基础算子”
   的结果；
5. 查找 NPU backend 中该 ATen 算子的 lowering 注册、shape/dtype 分支和 extern kernel 声明；
6. 用最小输入构造单算子复现，再逐步加回 layout、动态 shape、autocast、梯度和融合上下文。

一个实用判断是：如果去掉前后节点后单个算子仍无法生成 IR，优先查 lowering；如果单算子可以，
但与邻居融合或特定 layout 组合失败，则继续查 scheduler、fusion 和 layout propagation。

## 四、fallback 是什么

### 1. 总体定义

`fallback` 的意思是：当前编译层无法为某个节点或某段代码生成目标后端的优化实现，于是把它
交给另一条可执行路径。这个“另一条路径”可能是 eager dispatcher、厂商 extern kernel、
CPU 实现、原始 Python 代码，具体取决于 fallback 发生在哪一层。

fallback 的关键不是“报错”，而是“继续执行但绕开当前优化路径”。因此它既可能是有意设计的
兼容机制，也可能是性能问题的隐性来源。

### 2. 三种经常被混淆的 fallback

#### 2.1 Dynamo graph break：退出编译图，回到 eager

Dynamo 在遇到无法追踪的 Python 控制流、数据依赖的 `item()`、不支持的对象操作或其他捕获
限制时，会发生 graph break：

```text
Python 函数
  ├─ 可捕获的 Tensor 区域 ──> FX 图 ──> Inductor 编译
  ├─ graph break 区域 ─────> eager/Python 执行
  └─ 后续可捕获区域 ───────> 可能再次形成 FX 图
```

这不是 Inductor 的“单个算子 fallback”，而是 Dynamo 在 Python 前端切断图边界。它会减少可
融合范围，增加编译函数调用和同步机会。调试时可用 `fullgraph=True` 把 graph break 变成错误，
从而强制定位断点；生产环境不应把 `torch._dynamo.config.suppress_errors=True` 当作修复方案，
否则编译失败可能静默退回 eager。

#### 2.2 Inductor extern/custom op：图还在，但节点变成不透明调用

如果 Inductor 没有通用 lowering，或者后端希望使用厂商优化实现，可以把节点保留为
`ExternKernel`、custom op 或 vendor library call：

```text
FX 节点 → 不做通用 IR lowering
       → ExternKernel / torch.library custom op
       → NPU runtime 调用 AscendC 或厂商库
```

这种方式仍然属于编译图的一部分，通常不会像 graph break 那样回到 Python；但该节点对
Inductor 来说是“黑盒”，前后 pointwise 节点往往不能跨过它融合。若要让它可靠参与
`torch.compile`，通常还需要注册正确的设备 kernel、fake/meta 实现（用于形状和 dtype 推导）、
必要的 autograd 规则，并准确声明 aliasing/mutation。

#### 2.3 Dispatcher/device fallback：落到其他 dispatch 实现或设备

PyTorch dispatcher 可以为某个 device 注册全局或按算子的 fallback。NPU 缺少某个算子的实现
时，可能选择 CPU fallback 或其他 dispatch key 的实现；这与 Inductor 的 extern kernel 不是
一回事：前者是 dispatcher 在算子分发阶段换实现，后者是编译器显式生成的外部调用。

```text
NPU aten.foo
    ↓ 没有 NPU kernel
dispatcher fallback
    ├─ fallthrough 到其他 dispatch key
    ├─ 调用 CPU/通用实现（可能发生设备拷贝）
    └─ 抛出错误
```

CPU fallback 可能触发 NPU→CPU 和 CPU→NPU 拷贝、同步，甚至破坏异步执行；如果只观察最终
loss，往往不容易发现。因此训练热路径不应默认接受隐式 CPU fallback。

### 3. fallback 与 decomposition 的关系

两者都可以用来“绕开后端不支持”，但代价完全不同：

```text
高层算子不支持
    ├─ 有可用 decomposition
    │    └─ 改写成基础算子 → lowering → 仍可调度/融合
    └─ 没有可靠 decomposition
         ├─ 有厂商实现 → ExternKernel/custom op → 图内黑盒调用
         └─ 没有实现 → graph break/eager 或 device fallback
```

优先级通常是：先确认是否存在数值等价且性能可接受的 decomposition；否则选择明确的
NPU extern/custom op；最后才允许 graph break 或 CPU fallback。分解会增加 IR 节点和编译工作，
但仍保留优化空间；fallback 则通常牺牲融合、设备一致性或编译可预测性。

### 4. fallback 的性能、正确性和可维护性风险

| 风险 | 具体表现 | NPU 上的典型后果 |
|---|---|---|
| kernel/调用开销 | 黑盒节点或断图使算子无法融合 | launch 次数增加，短算子占比变高 |
| 同步与拷贝 | CPU fallback 需要跨设备搬运 | NPU pipeline 被打断，出现 D2H/H2D |
| layout/dtype | 外部实现要求不同布局或精度 | 额外 transpose/cast，甚至结果精度变化 |
| autograd | 只注册 forward，没有 backward/fake 实现 | 训练编译失败或梯度路径回 eager |
| alias/mutation | 未正确描述视图和原地写 | 结果错误、缓存复用失效或出现隐蔽 bug |
| 动态 shape | fallback 路径不能表达符号 shape | 频繁重新编译或退回 eager |
| 可观测性 | fallback 没有明确告警 | 性能下降但难以从 loss 判断原因 |

### 5. 如何确认到底发生了哪种 fallback

建议同时看编译日志、FX/IR 和设备侧 profiler，而不要只根据“程序没有报错”判断：

1. **确认是否 graph break**：使用 `torch._dynamo.explain` 或编译日志查看 break reason；
   必要时用 `fullgraph=True` 让第一个断点直接报错。相关 API 属于调试接口，具体日志开关随
   PyTorch 版本变化，应以当前版本文档为准；
2. **确认是否 extern/custom op**：检查生成的 Inductor IR/代码中是否出现 `ExternKernel`、
   custom operator 或 vendor library call，并确认该节点前后是否仍能 fusion；
3. **确认是否 CPU/device fallback**：在 profiler 中查找 CPU 算子、D2H/H2D、同步和异常的
   host time；同时检查该算子的 dispatcher 注册表和 NPU kernel 是否存在；
4. **对照运行**：比较 eager、普通 Inductor、AscendC backend 三种结果的数值、kernel 数量、
   step time 和峰值显存；
5. **最小化复现**：将可疑算子单独放在 NPU 上运行，再逐步加入动态 shape、autocast、梯度、
   view/原地写和上下文融合，确定触发 fallback 的最小条件。

### 6. NPU 项目的 fallback 使用原则

建议把 fallback 当作显式的兼容策略管理，而不是默认的“兜底开关”：

1. 核心训练热路径（attention、MLP、通信、归一化和 optimizer step）禁止无告警的 CPU fallback；
2. 冷路径或控制流算子可以允许 graph break，但要记录发生位置和预期频率；
3. 能用基础 ATen 算子表达时，优先评估 decomposition，并用精度和性能基准验证；
4. 必须使用厂商实现时，使用 `torch.library` 定义 custom op，并补齐 kernel、fake/meta、
   autograd 与 alias/mutation 契约；
5. 在 CI 或 profiling 中把 fallback 当作可观测指标，避免 CANN、PyTorch 或 torch_npu 升级后
   出现静默回退；
6. 临时使用 suppress errors 只用于收集兼容性信息，定位完成后应恢复为显式报错或明确的
   fallback 白名单。

### 7. 一个简化的决策树

```text
某个 FX/ATen 节点无法编译
        ↓
是否存在数值等价的 decomposition？
        ├─ 是 → 分解 → 基础算子 lowering → 调度/融合 → NPU Codegen
        └─ 否
             ↓
        是否有 NPU extern/custom op？
             ├─ 是 → 图内黑盒调用 → 检查 fake/autograd/布局和性能
             └─ 否
                  ↓
        是否允许该区域 graph break？
             ├─ 是 → eager 执行 → 记录断图和同步代价
             └─ 否 → 显式报错，补充 lowering 或 NPU kernel
```

## 五、component 是什么

### 1. 定义

`component` 不是 PyTorch `torch.compile` API 的概念，而是 TorchTitan/TorchTitan-NPU
在训练编排层定义的配置概念：

```python
CompileConfig(
    enable=False,
    components=["model", "loss"],
    backend="inductor",
)
```

它表示训练流程中的哪些部分需要调用 `torch.compile`，不是 NPU device、FX node、decomposition
或物理上的模型切分。

### 2. 当前主要 component

| component | 编译对象 | 典型行为 |
|---|---|---|
| `"model"` | 模型主体 | 编译 TransformerBlock 或模型子模块 |
| `"loss"` | loss 函数 | 编译交叉熵等 loss 计算 |
| `"muon"` | Muon 中的 Newton-Schulz 张量函数 | 编译 Muon 参数更新中的矩阵迭代 |

`"model"` 通常不是把整个模型一次性编译，而是对重复的 TransformerBlock 分别调用：

```python
for _, transformer_block in model.layers.named_children():
    transformer_block.compile(
        backend=compile_config.backend,
        fullgraph=True,
    )
```

`"loss"` 是独立编译区域：

```python
if compile_config.enable and "loss" in compile_config.components:
    loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
```

`"muon"` 编译的是 Newton-Schulz 函数，而不是整个 optimizer：

```python
self._zeropower_fn = torch.compile(
    zeropower_via_newtonschulz5,
    backend=compile_backend,
    fullgraph=True,
    dynamic=True,
)
```

所以：

```python
components=["model", "loss", "muon"]
```

表示三个独立的编译对象，不表示生成一个包含整个训练步骤的统一大图。

### 3. 四个配置项的关系

```text
enable
    是否允许使用 torch.compile

components
    model/loss/muon 中哪些对象调用 torch.compile

backend
    这些对象调用 torch.compile 时使用哪个 Dynamo backend

npu_backend
    Inductor 内部如何生成 NPU Kernel
```

## 六、最终记忆方式

```text
component：编译哪一块？
backend：整张图交给谁？
decomposition：图里的高层算子如何展开？
lowering：每个 FX/ATen 节点如何变成 Inductor IR 或 extern 节点？
fallback：无法编译时，是留在图内、断图执行，还是落到其他设备？
npu_backend：Inductor 最终如何生成 NPU Kernel？
```

一句话总结：

> `component` 决定编译范围；Dynamo backend 决定整图编译入口；decomposition 决定算子图如何改写；
> lowering 决定节点如何落到 Inductor IR；fallback 决定不支持路径如何继续执行；NPU Codegen 决定最终的 NPU Kernel 如何生成。
