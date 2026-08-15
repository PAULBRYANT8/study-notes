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

## 三、component 是什么

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

## 四、最终记忆方式

```text
component：编译哪一块？
backend：整张图交给谁？
decomposition：图里的高层算子如何展开？
npu_backend：Inductor 最终如何生成 NPU Kernel？
```

一句话总结：

> `component` 决定编译范围；Dynamo backend 决定整图编译入口；decomposition 决定算子图如何改写；NPU Codegen 决定最终的 NPU Kernel 如何生成。
