# TorchTitan-NPU 动态量化与模型配置树梳理

本文整理动态量化模块类、动态量化配置类、量化策略配置和模型配置树之间的关系，并沿当前源码说明它们如何生效。核心结论是：配置转换决定「构造什么模块」，模块构造安装参数包装器，矩阵乘执行时才根据当前张量进行低精度计算。

本文是源码阅读笔记，不是 SwigluGroup 修改方案或 A5 性能验证报告。整理日期：2026-09-11。

源码基线：

- 本地 torchtitan-npu：`swiglu_group` 分支，提交 `0fd6d7aa0bb43685ce585b36938059c92b7bb518`。
- 本地上游 torchtitan：提交 `b175497ea3502388bb9513391810a86ca46dbfa8`。
- torchao-npu 部分依据本仓 `experiments/torchao-npu/torchao_npu/` 中的源码；训练实际使用的是环境中安装的 `torchao_npu` 包，部署时仍需核对安装版本。

## 1. 先区分几种名字相似的对象

| 对象 | 回答的问题 | 当前职责 |
| --- | --- | --- |
| `NpuQuantizeConverter.Config` | 转换器应该处理哪些节点、采用什么策略？ | 保存 `filter_fn`、`base_config` 等 |
| `NpuQuantizeConverter` | 怎样修改模型配置树？ | 遍历配置、筛选节点、替换配置对象 |
| 动态量化模块类 `NpuQuantizedXXX` | 最后构造哪一种模块？ | 继承原模块，在构造函数中安装参数包装器 |
| 动态量化配置类 `NpuQuantizedXXX.Config` | 如何构造上面的模块？ | 保留原模块字段，并携带 `_torchao_npu_config` |
| `ParamSwapConfig` | 怎样给模块参数安装量化包装器？ | 组合权重与激活策略，指定 prepare/convert 行为 |
| `BlockQuantizeConfig`、`MXQuantizeConfig` | 张量具体怎样量化？ | 描述 dtype、分块大小、scale 算法等 |

这里有两种不同的「动态」：

- 动态生成类：Python 在调用工厂时创建类对象，属于配置准备阶段。
- 动态量化：执行矩阵乘时根据当前权重、激活计算量化数据和缩放因子，属于张量计算阶段。

动态生成一个类，不代表此时已经把权重量化了；每次 forward 也不会重新创建一个 Python 模块类。

主要入口：[量化适配器][npu-converter]，重点阅读第 87—186 行。

## 2. 动态量化模块类和动态量化配置类是什么关系？

### 2.1 两条继承链，通过 Config 和 _owner 关联

工厂 `_get_npu_quantized_module_cls(parent_cls)` 的核心结构如下。此处省略错误检查、类型注解及缓存逻辑：

~~~python
parent_config_cls = parent_cls.Config

class NpuQuantizedModule(parent_cls):
    @dataclass(kw_only=True, slots=True)
    class Config(parent_config_cls):
        _torchao_npu_config = None

    def __init__(self, config):
        super().__init__(config)
        quantize_(
            self,
            config._torchao_npu_config,
            filter_fn=lambda candidate, _fqn: candidate is self,
        )
~~~

模块类和配置类之间不是继承关系，而是「模块拥有配置类，配置类知道自己要构造哪个模块」的关系：

~~~text
模块继承链：
parent_cls
    └── NpuQuantizedXXX

配置继承链：
parent_cls.Config
    └── NpuQuantizedXXX.Config

关联：
NpuQuantizedXXX.Config         → 量化模块使用的配置类
NpuQuantizedXXX.Config._owner  → NpuQuantizedXXX 模块类
~~~

例如传入当前 NPU patch 中的 `_ClampGroupedExperts` 后，工厂逻辑上生成：

~~~python
class NpuQuantized_ClampGroupedExperts(_ClampGroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(_ClampGroupedExperts.Config):
        _torchao_npu_config = None
~~~

类名中的下划线来自原类名 `_ClampGroupedExperts`。真正决定行为的是继承关系和 `_owner`，不是名称中的 `NpuQuantized` 字符串。

### 2.2 _owner 如何设置？

上游 [Configurable 实现][configurable] 的 `__init_subclass__()` 在外层模块类创建时执行以下绑定，位置在第 162—180 行：

~~~python
config_cls = cls.__dict__["Config"]
config_cls._owner = cls
~~~

因此，创建动态模块类后，会得到：

~~~python
quantized_cls.Config._owner is quantized_cls
~~~

`_owner` 是配置类上的类变量，不是每个配置对象都需要填写的普通 dataclass 字段。仅仅把一个类嵌套写在另一个类里面，不会自动产生这套构建关系；关系由 torchtitan 的 `Configurable` 机制建立。

### 2.3 为什么还需要一个新的 Config 类？

原始配置描述原始模块，例如专家数、输入维度、隐藏维度、`swiglu_limit`。

量化配置在此基础上增加：

~~~python
_torchao_npu_config
~~~

这个字段把 `ParamSwapConfig` 从配置转换阶段带到模块构造阶段。同时，新配置的 `_owner` 指向量化模块类，保证构建时走量化模块的构造函数。

只创建量化模块类、但不把配置树节点替换为对应的量化配置，后续 `build()` 仍可能构造原始模块。只在普通配置对象上保存量化策略，也不会自动调用 `quantize_()`。

### 2.4 derive() 和 build() 不能混淆

这是前面口头说明需要更正的地方：`derive()` 不负责创建 `nn.Module`。

配置转换：

~~~python
new_cfg = derive(
    old_cfg,
    quantized_cls.Config,
    _torchao_npu_config=param_swap_config,
)
~~~

结果只是一个新的配置对象。上游 [derive 实现][derive] 会复制源配置与目标配置共有的字段，再应用显式传入的字段修改；只存在于源配置中的字段会被丢弃。复制是浅拷贝，嵌套配置仍共享引用。

模型构建：

~~~python
module = new_cfg.build()
~~~

上游 `Config.build()` 的核心逻辑是：

~~~python
return self._owner(config=replace(self))
~~~

此时才真正调用量化模块的 `__init__()`。如果后续 override 再次替换了配置节点，最终构建的则是替换后的配置所对应的模块。

### 2.5 为什么不重新编写 forward()？

当前动态量化类没有重写 `forward()`，所以会继承传入的具体 `parent_cls` 的实现，并保留其自定义配置字段。

以 `_ClampGroupedExperts` 为例：

- 原构造函数创建 `w1_EFD`、`w2_EDF`、`w3_EFD` 等参数，并保留 clamp 配置。
- 动态量化构造函数先调用原构造函数，再包装参数。
- forward 仍然使用原专家计算逻辑。
- 被包装的权重参与受支持的矩阵乘时，触发低精度实现。

保留父类并不等于自动接入任何融合算子。如果父类本来不是 `AscGroupedExperts`，继承它不会凭空获得 `AscGroupedExperts.forward()`。

### 2.6 缓存的是类，不是模块实例或权重

`_npu_quantized_module_cache` 的关系是：

~~~text
原始模块类 → 对应的动态量化模块类
~~~

同一种原始模块在多个层出现时，可以复用同一个动态类，但各层仍有独立的模块实例和参数。量化策略通过每个配置对象的 `_torchao_npu_config` 传入，不需要为每个策略重复造类。

转换器还检查配置 owner 是否已经属于缓存中的量化类，从而跳过这类节点。这是当前转换流程中的重复转换保护，不是对任意外部类都适用的通用去重机制。

## 3. BlockQuantizeConfig 和 MXQuantizeConfig 的作用

这两个类定义在 [quant_configs.py][quant-configs]，分别从第 64 行和第 25 行开始。它们是数值策略配置，不是专家模块，也不是用来匹配模型路径的过滤器。

### 3.1 MXQuantizeConfig

MX 指 Microscaling，即微缩放格式。当前类主要描述：

- `block_size`：固定为 32。
- `elem_dtype`：默认 `torch.float8_e4m3fn`，配置层还接受 FP8 E5M2 和 FP4 E2M1。
- `round_mode`：默认 `"rint"`。
- `scale_alg`：缩放因子计算算法；未显式设置时，当前代码对 FP8 选择 1，对 FP4 选择 2。
- `dst_type_max`：目标数据类型的最大值参数；0.0 表示自动推断。

它既可以描述权重量化，也可以描述激活量化。具体可运行的组合仍受 wrapper、算子、设备与 runtime 限制；配置类接受某个 dtype，不等于所有矩阵乘和训练组合均已验证。

### 3.2 BlockQuantizeConfig

该类描述当前 NPU Block FP8 路径使用的权重量化规则：

- `block_size` 固定为 32。
- `elem_dtype` 默认 FP8 E4M3，也允许 FP8 E5M2。
- `round_mode` 固定为 `"rint"`。
- `scale_alg` 固定为 0。
- `dst_type_max` 固定为 0.0。
- 可选字段 `mxfp4_fake_quantize_config` 用于在 Block FP8 矩阵乘之前，对权重加入 MXFP4 fake quant。

当前 grouped GEMM 实现对权重使用 32 × 32 block 量化，并准备不同方向的缩放信息。不能仅凭两个类都有 `block_size=32`，就认为 MX 与 Block 的权重布局和算法完全相同。依据见 [block_ops.py][block-ops] 的 `to_block_fp8_then_grouped_mm()`。

### 3.3 当前 recipe 如何组合它们？

适配器第 211—235 行：

~~~python
# MX 路径
mx_config = MXQuantizeConfig()
ParamSwapConfig(
    weight_config=mx_config,
    activation_config=mx_config,
)

# Block FP8 路径；省略可选的 MXFP4 fake quant
ParamSwapConfig(
    weight_config=BlockQuantizeConfig(),
    activation_config=MXQuantizeConfig(),
)
~~~

因此，`all_block_fp8` 这个 recipe 名称不表示 `weight_config` 和 `activation_config` 都使用 `BlockQuantizeConfig`。当前实现是 Block 权重策略配合 MX 激活策略。

此外，recipe 中的 `all` 指当前筛选范围中的 attention、共享专家和路由专家节点，不代表模型中的每个算子、参数和张量都会变成 FP8。

## 4. weight_config 和 activation_config 是权重与激活吗？

是，但准确表述是「权重的量化策略」与「激活的量化策略」，它们不是张量本身，也不是已经计算好的 scale。

对于普通矩阵乘：

~~~text
Y = A @ B

A：本次矩阵乘的输入激活
B：参与矩阵乘的权重，可能已经做过转置

activation_config：A 的量化规则
weight_config：B 的量化规则
~~~

对于路由专家 grouped GEMM：

~~~text
A：按专家分组排列的 token 激活，形状通常为 [M, K]
B：各专家权重，形状通常为 [E, K, N]
group_list：各专家 token 分组的累计边界
~~~

这里的「激活」不专指 SiLU/SwiGLU 函数。每次矩阵乘的输入特征都是激活，例如第一层矩阵乘输入的 `x`，以及第二层矩阵乘输入的 SwiGLU 输出。

当前实现还有三个需要区分的边界：

1. [MX wrapper][mx-wrapper] 要求两项都是 `MXQuantizeConfig`，且配置比较结果相等；不是可以任意混配。
2. [Block wrapper][block-wrapper] 要求权重为 `BlockQuantizeConfig`、激活为 `MXQuantizeConfig`。
3. Block grouped GEMM 的反向还会用保存的 `config_A` 量化输出梯度 `dY`，所以 `activation_config` 在该实现中的作用不仅限于前向输入。

SwigluGroup 算子接口中的可选 `weight` 则是另一种含义：当前接入用于传递路由分数，不是这里的可训练专家权重，也不是 `weight_config`。

## 5. 模型匹配与量化策略分发是两层机制

### 5.1 第一层：选中模型配置节点

`NpuQuantizeConverter.convert()` 先遍历：

~~~python
model_config.traverse(Linear.Config)
model_config.traverse(BatchedLinear.Config)
model_config.traverse(GroupedExperts.Config)
~~~

再调用：

~~~python
self.config.filter_fn(config, fqn)
~~~

因此决定「哪些模型部件需要量化」的是配置类型和路径过滤器。例如：

~~~text
共享专家：
*.moe.shared_experts.w1
*.moe.shared_experts.w2
*.moe.shared_experts.w3

路由专家：
*.moe.routed_experts.inner_experts
~~~

实际适配器使用的是后缀匹配函数，以上星号写法只是路径示意。

### 5.2 第二层：选中参数包装器实现

模型构建时，`quantize_(module, ParamSwapConfig(...))` 进入 [ParamSwap handler][param-swap]。当前 prepare 路径的关键代码是：

~~~python
params_handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[
    type(config.weight_config)
]
~~~

这里注册表的准确名称是 `_PARAM_SWAP_QUANTIZE_CONFIG_HANDLER`。

分发关系是：

~~~text
BlockQuantizeConfig → BlockTrainingWeightWrapperTensor
MXQuantizeConfig    → MXTrainingWeightWrapperTensor
~~~

参数包装器同时保存权重和激活配置，后续在矩阵乘入口使用。参数仍然作为 `nn.Parameter` 注册在模块中，只是其数据被训练期 tensor wrapper 包装。

因此，`BlockQuantizeConfig` 和 `MXQuantizeConfig` 不负责判断一个节点是不是路由专家；它们负责在节点已经选中后选择和配置低精度参数计算路径。

## 6. 模型配置树是什么？

### 6.1 它是构建模型之前的嵌套配置对象

模型配置树以 `model_spec.model` 为根，由嵌套的 Config 对象和列表等组成。下面只展示与专家量化相关的局部结构，省略其他层和组件：

~~~text
model_spec.model
└── layers[i]：Transformer block 配置
    └── moe
        ├── routed_experts
        │   ├── inner_experts：GroupedExperts.Config
        │   └── token_dispatcher：token 分发配置
        └── shared_experts：FeedForward.Config
            ├── w1：Linear.Config
            ├── w2：Linear.Config
            └── w3：Linear.Config
~~~

配置树和已构建的模块树不同：

- 配置树保存模块类型、维度、子模块配置、算法选项等，供构建阶段使用。
- 模块树由实际 `nn.Module` 对象组成，包含已经创建或注册的参数和 forward 方法。
- 配置树也不是运行时算子图；FX 图、autograd 图描述的是另一层关系。

### 6.2 一个重要的结构差异

[共享专家 FeedForward][feed-forward] 的 `w1/w2/w3` 是三个 `Linear.Config` 子节点，构建后是三个 Linear 子模块。

[路由专家 GroupedExperts][grouped-experts] 则直接创建 `w1_EFD/w2_EDF/w3_EFD` 三个 Parameter；它们并不是三个 `Linear.Config` 子节点。

所以当前转换器的处理单位不同：

- 共享专家：替换内部三个 Linear 的配置，外层 FeedForward 配置类型不因量化而改变。
- 路由专家：替换整个 `inner_experts` 配置，构建时包装其内部专家权重。

这不是说路由专家把整个模块的每一个操作都变成了低精度，而是说量化配置的挂载位置是整个专家模块。

### 6.3 traverse() 怎样定位并替换节点？

上游 `Config.traverse()` 返回：

~~~python
fqn, config, parent, attr
~~~

含义：

- `fqn`：相对于遍历根节点的完整限定名（Fully Qualified Name）。
- `config`：匹配到的配置对象。
- `parent`：持有当前节点的父配置对象或列表。
- `attr`：字段名或列表下标。
- 根节点本身匹配时，`parent` 和 `attr` 为 `None`。

从 `model_spec.model` 开始遍历时，FQN 示例为：

~~~text
layers.0.moe.routed_experts.inner_experts
~~~

适配器打印日志时会另外补上 `model_spec.model.` 前缀；不应把该日志前缀误认为遍历返回值自身的一部分。

当前遍历通过 `isinstance()` 匹配配置子类，默认匹配到节点后不再向其内部继续下降。这与 override 的 `exact=True` 精确类型检查不是同一项规则。

节点替换由 `_replace_config()` 完成：

~~~python
if parent is None:
    return replacement
if isinstance(parent, list):
    parent[attr] = replacement
else:
    setattr(parent, attr, replacement)
~~~

## 7. 从 TrainerEx 到实际低精度计算的完整过程

`TrainerEx` 是 [NPU Trainer 扩展类][trainer-ex]，继承上游 `Trainer`；前面讨论中的「trainEX」指的是这个类，不是一个量化算子或单独的脚本。

当前时序如下：

~~~text
TrainerEx.__init__()
    │
    ├── 量化启用时：apply_quantization_converter()
    │       ├── recipe 创建转换器配置
    │       ├── converter_config.build().convert(model_config)
    │       └── 替换选中的模型配置节点
    │
    └── super().__init__() 进入上游 Trainer
            ├── model_config.update_from_config()
            ├── apply_overrides()
            └── model_config.build()
                    └── 子配置递归 build()
                            └── NpuQuantizedXXX.__init__()
                                    ├── super().__init__() 创建参数
                                    └── quantize_() 安装参数包装器

训练 forward/backward
    └── 被包装权重参与 mm / grouped_mm 等受支持入口
            └── wrapper 分发到低精度计算实现
                    ├── 根据当前张量计算量化数据与 scale
                    ├── 调用低精度矩阵乘
                    └── 自定义 backward 计算并返回梯度
~~~

依据：[NPU Trainer 第 44—61 行][trainer-ex]、[量化适配器第 294—331 行][npu-converter]、[上游 Trainer 第 299—321 行][trainer]。

需要特别注意三个时间点：

- 配置转换时：主要修改配置对象，不处理真实训练数据。
- 模块构造时：调用 `quantize_()` 安装 wrapper；当前 Trainer 在 meta device 上构建模型，此时不是根据真实权重值计算 scale。
- 算子运行时：才拿当前权重、激活执行动态量化和低精度计算。

包装器安装在模型构建期间，早于后续 TP/EP/FSDP 处理和优化器构造，使这些流程能看到最终的参数表示。这是当前适配的组织方式，不意味着所有框架都不能在模型实例化之后进行量化转换；安全边界在于参数替换与并行、优化器等流程的时序是否一致。

## 8. 为什么采用运行时动态量化？

动态量化的直接原因是训练中的权重和激活会变化。MoE 的 token 路由还会改变各专家输入的内容、数量与分组边界，使当前数据的缩放和分组处理更重要；但动态路由并不是动态量化唯一成立的前提，普通 Linear 也使用这种方式。

以当前 Block grouped GEMM 为例，[block_ops.py][block-ops] 的 `_BlockFP8QuantGroupedMM` 做了以下工作：

1. 对当前输入 `A` 沿前向收缩维度执行 MX 动态量化。
2. 按专家分组、沿另一维度准备反向需要的激活量化表示。
3. 对当前权重 `B` 执行 Block FP8 量化，可选地先加入 MXFP4 fake quant。
4. 调用 `npu_grouped_matmul` 执行实际低精度前向矩阵乘。
5. 保存反向需要的量化张量和 scale。
6. 反向时对当前 `dY` 动态量化，复用保存的数据计算输入梯度和权重梯度。

因此，「动态」不意味着前反向所有权重和激活都必须重新量化一遍：当前实现会在 forward 保存一部分结果供 backward 复用。

采用这种组织方式可以在保留训练参数更新路径的同时，利用低精度矩阵乘。是否获得性能收益还取决于量化开销、shape、编译路径、设备和算子实现，不能只依据配置名称下结论。

也不能因为配置类继承了 QAT 相关基类，就把整条路径称为「只做 fake quant」：

- 当前 Block/MX wrapper 使用实际低精度矩阵乘。
- 可选 MXFP4 fake quant 是额外的数值模拟步骤。
- 反向由对应的 autograd 实现计算梯度；不应把所有低精度反向都笼统解释成 STE。
- 参数包装本身不意味着存在一份额外的 FP32 master weight。底层参数 dtype 与训练配置和优化器实现有关。

## 9. 与上游 torchtitan 的相同点和不同点

上游 [MXFP8GroupedExpertsConverter][upstream-mx] 同样先遍历专家配置，通过实际配置 owner 动态创建量化子类，并替换配置节点。这样可以复用不同模型已有的专家高层逻辑，而不是为每一种专家复制完整实现。

但当前上游与本仓 NPU 的低精度接入点不同：

| 方面 | 本地上游 MXFP8 专家 | 当前 NPU 通用量化适配 |
| --- | --- | --- |
| 配置准备 | 遍历并替换 GroupedExperts 配置 | 遍历并替换 Linear、BatchedLinear、GroupedExperts 配置 |
| 动态子类保存策略 | `recipe_name` 生成运算配置 | `_torchao_npu_config` 携带 ParamSwapConfig |
| 矩阵乘替换方式 | 重写 `_grouped_mm()` | 包装参数，由 tensor wrapper 拦截计算 |
| 专家布局配套 | converter 调整 dispatcher 的 padding 配置 | 不能仅凭动态继承就推断等价支持，需核对 NPU 配套路径 |

上游 [GroupedExperts._grouped_mm() 的注释][grouped-experts] 明确解释了一个动机：把矩阵乘放在可重写的方法入口中，而不是隐藏在 tensor subclass 的 `__torch_function__` 后面，可以使 graph_trainer 的 `make_fx` 等 FX tracing 路径捕获该操作。

因此，二者共享「配置树转换＋保留具体父类」的组织思路，但不能说「当前上游也是用 ParamSwap wrapper 实现专家量化」。上游 tracing 兼容性也不能直接作为当前 NPU wrapper 已通过同类验证的证据。

## 10. 与 SwigluGroup 问题的联系

当前 [SwigluGroup override][swiglu-override] 对路由专家和共享专家均设置了 `exact=True`，但目标节点不同。

量化之后：

- 路由专家：`inner_experts` 自身已经变成量化配置子类，后续精确类型匹配无法命中原配置类型。
- 共享专家：变化的是内部 `w1/w2/w3` 的配置，外层共享专家配置仍可以命中对应的 override。
- 共享专家 override 使用 `derive()` 时，可以保留内部已经量化的三个 Linear 配置。

因此，现象来自配置替换粒度、执行时序与精确匹配规则的组合，不应概括为「量化一定不能与 SwigluGroup 共存」。动态量化子类也不会自动把原始专家的 forward 换成融合实现。

本节只说明当前实现与此前问题的联系，不代表 SwigluGroup 量化兼容方案已经修改或验证。

## 11. 建议的源码阅读顺序与核对状态

按下面顺序阅读，可以从模型配置一直跟到实际算子：

1. [trainer.py][trainer-ex]：`TrainerEx.__init__()`，第 44 行。
2. [torchao_converter.py][npu-converter]：`apply_quantization_converter()`，第 294 行；`_recipe_converters()`，第 251 行。
3. 同一文件：`NpuQuantizeConverter.convert()`，第 149 行；动态类工厂，第 87 行。
4. [configurable.py][configurable]：`traverse()`、`build()`、`__init_subclass__()`，分别从第 75、134、162 行开始。
5. [override.py][derive]：`derive()`，第 249 行。与 `build()` 对照，明确配置转换和模块实例化的区别。
6. [param_swap.py][param-swap]：`ParamSwapConfig` 和 `_param_swap_config_transform()`，分别从第 30、112 行开始。
7. [quant_configs.py][quant-configs]：MX 与 Block 量化参数。
8. [block_wrapper_tensor.py][block-wrapper]：`__torch_function__()`，第 55 行；参数 handler，第 154 行。
9. [block_ops.py][block-ops]：`_BlockFP8QuantGroupedMM`，第 200 行；前向第 244 行、反向第 310 行。
10. [上游 mx.py][upstream-mx]：动态专家类工厂，第 104 行；converter，第 149 行。

行号对应本文记录的本地源码基线，后续代码变化时以符号名定位为准。

核对范围：

- 源码静态核对：已执行，覆盖配置替换、owner 绑定、模块构造、策略分发和 Block grouped GEMM 前反向入口。
- A5 运行与 profiling 验证：未执行；本次只整理文档，不把静态分析写成设备验证结果。
- 训练代码修改：未进行。

一句话总结：Config 决定构造哪种模块，量化模块在构造时安装 wrapper，ParamSwapConfig 组合权重与激活策略，Block/MX 配置决定具体数值规则，运行时矩阵乘才真正消费这些规则。

[npu-converter]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/interfaces/torchao_converter.py
[trainer-ex]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/torchtitan_npu/extensions/trainer.py
[configurable]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/config/configurable.py
[derive]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/config/override.py
[trainer]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/trainer.py
[quant-configs]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/experiments/torchao-npu/torchao_npu/quantization/quant_configs.py
[param-swap]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/experiments/torchao-npu/torchao_npu/configs/param_swap.py
[block-wrapper]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/experiments/torchao-npu/torchao_npu/wrapper_tensors/block_wrapper_tensor.py
[mx-wrapper]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/experiments/torchao-npu/torchao_npu/wrapper_tensors/mx_wrapper_tensor.py
[block-ops]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/experiments/torchao-npu/torchao_npu/ops/block_ops.py
[upstream-mx]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/components/quantization/mx.py
[grouped-experts]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/models/common/moe.py
[feed-forward]: /home/zhangwei/下载/NPU/2026_05/torchtitan/torchtitan/models/common/feed_forward.py
[swiglu-override]: /home/zhangwei/下载/NPU/2026_05/torchtitan-npu/torchtitan_npu/override/deepseek_v4/swiglu_group/__init__.py
