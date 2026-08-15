# DeepSeek-V4 接入 SwigluGroup 开发复盘（当前实现）

> 本文基于 torchtitan-npu 的 feat/swiglu-group-a5-fusion 分支和
> [Merge Request !468](https://gitcode.com/cann/torchtitan-npu/merge_requests/468) 整理。
>
> 当前对外配置名为 npu_swiglu_group，实现文件为
> torchtitan_npu/converters/kernels/swiglu_group.py。
>
> 本文描述的是提交 5a6b658 所在分支的接入方式。旧版本中的
> npu_gmm_swiglu、gmm_swiglu.py、局部 _NpuSwigluGroup Autograd 桥以及
> 空 Tensor 分解回退均已不再使用。
>
> 最后更新：2026-08-03。

## 0. 当前结论

现在的接入不是“SwigluGroup converter 重写整个专家模块”，而是三层 converter
组合后，在已有模块上选择 activation 实现：

~~~text
npu_moe_dispatch
  ├─ 接管 MoE dispatch
  └─ 把共享专家适配为 NpuSharedExperts
       └─ 默认 activation：native_shared_expert_activation

npu_gmm
  └─ 把路由 GroupedExperts 转为 NpuGroupedExperts
       └─ 默认 activation：_expert_activation

npu_swiglu_group
  ├─ 校验 A5、converter 顺序和 CANN dispatcher
  ├─ 路由专家：注入 swiglu_group_activation
  └─ 共享专家：注入 swiglu_group_shared_activation
~~~

关键点如下：

1. npu_swiglu_group 只负责后端实现选择，不再定义共享专家结构。
2. 共享专家的结构和默认小算子实现位于通用 MoE 层。
3. 路由专家的 GMM、w13 布局和 activation compile 仍由 gmm.py 管理。
4. torchtitan-npu 直接调用公开的 cann_ops_nn::swiglu_group；Autograd 由
   ops-nn 包负责注册。
5. 当前代码不为空 Tensor 增加特殊分支，输入合约和报错由 CANN 算子决定。

## 1. 数学语义与算子参数

### 1.1 SwiGLU

普通 SwiGLU FFN 的数学形式为：

~~~text
gate = w1(x)
up   = w3(x)
hidden = silu(gate) * up
output = w2(hidden)
~~~

把 gate 和 up 沿最后一维拼接：

~~~text
packed = [gate | up]
~~~

激活函数会把 packed 的最后一维均分为 A 和 B，并计算：

~~~text
silu(A) * B
~~~

因此共享专家构造 packed 时必须保持 w1 在前、w3 在后。顺序颠倒会改变模型语义。

### 1.2 Clamp 和可选权重

SwigluGroup 把 clamp、SwiGLU 和可选的逐行权重乘法放在一个入口中：

~~~text
x = [A | B]

if clamp_limit > 0:
    A = min(A, clamp_limit)
    B = min(max(B, -clamp_limit), clamp_limit)

y_origin = silu(A) * B
y = y_origin                         if weight is None
y = y_origin * weight                otherwise
~~~

当前 Python 接口的参数映射为：

| 参数 | 含义 | 当前接入 |
| --- | --- | --- |
| x | 最后一维为 gate/up 拼接，输出最后一维减半 | 路由专家的 GMM-1 输出或共享专家 w1/w3 输出拼接 |
| weight | 可选的逐 routed row 权重 | 有 routed_scores 时转换为 FLOAT32 且 contiguous；共享专家传 None |
| group_index | 算子内部的分组信息 | 当前传 None |
| clamp_limit | 激活前截断阈值 | 有 swiglu_limit 时传 float；无 clamp 时传 -1.0 |

这里的 weight 是 router score，不是 w1、w2、w3 或 w13 参数。

### 1.3 当前不处理空 Tensor

swiglu_group_activation 不检查 h.numel()，而是直接调用 CANN：

~~~python
return torch.ops.cann_ops_nn.swiglu_group.default(
    h.contiguous(),
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
~~~

所以当前接入的边界是：

- torchtitan-npu 不提供空 Tensor 的 PyTorch 分解回退；
- 是否支持空输入由所安装的 cann_ops_nn 版本决定；
- 当前单元测试只验证异常会透传给 CANN，不声称空输入受支持；
- 如果真实 EP 配置能产生空 routed rows，应在对应环境单独验证，不能从非空训练推断。

## 2. Converter 顺序与职责

### 2.1 启用顺序

DeepSeek-V4 同时包含路由专家和共享专家时，应按下面顺序配置：

~~~python
get_model_converter_config("npu_moe_dispatch"),
get_model_converter_config("npu_gmm"),
get_model_converter_config("npu_swiglu_group"),
~~~

依赖关系是：

- npu_moe_dispatch 可以不依赖 SwigluGroup 独立工作；
- npu_gmm 可以使用默认 activation 独立工作；
- npu_swiglu_group 依赖已经转换好的 NpuGroupedExperts；
- 模型存在 shared_experts 时，npu_swiglu_group 还要求它已经是
  NpuSharedExperts；
- npu_swiglu_group 是可选 A5 能力，不应无条件放进所有设备的默认 converter 列表。

### 2.2 职责表

| 层 | 主要职责 | 不负责 |
| --- | --- | --- |
| torchtitan_npu/models/common/moe.py | NpuSharedExperts 结构、共享专家默认 activation | A5 判断、CANN 导入 |
| npu_moe_dispatch | MoE dispatch、共享专家通用适配 | 选择 SwigluGroup |
| npu_gmm | 路由专家 GMM、w13、state-dict 映射、默认 activation、activation compile | A5-only 算子加载 |
| npu_swiglu_group | A5/算子校验并向已有模块注入融合 activation | 重建专家模块、参数布局、state-dict 映射 |
| cann_ops_nn | SwigluGroup 前后向 dispatcher 和 Autograd 接线 | torchtitan 模型结构 |

这种拆分保证默认路径在没有 npu_swiglu_group 时仍然完整可运行，也避免通用 MoE
层依赖 A5/CANN。

### 2.3 修改模型前先完成校验

NpuSwigluGroupConverter.convert 的执行顺序为：

1. 校验设备类型必须为 A5；
2. 确认至少存在一个 NpuGroupedExperts，否则说明 npu_gmm 未先执行；
3. 查找路径末段名为 shared_experts 的模块；
4. 如果存在但不是 NpuSharedExperts，说明 npu_moe_dispatch 未先执行；
5. 加载 cann_ops_nn.ops 并检查 swiglu_group 和 swiglu_group_backward；
6. 最后才向路由专家和共享专家注入 activation。

因此设备、顺序或算子检查失败时，不会留下只转换一部分模块的中间状态。

## 3. 路由专家数据流

### 3.1 Router 输出之间的关系

Router 接收 token hidden states，并输出三个不同用途的结果：

~~~text
top_scores
  每个 token 被选中专家对应的权重

selected_experts_indices
  每个 token 选择的专家编号

num_tokens_per_expert
  每个专家收到多少 routed rows
~~~

它们不是同一个信息：

- selected_experts_indices 决定 token 去哪个专家；
- num_tokens_per_expert 决定 GMM offsets；
- top_scores 决定专家结果如何加权。

### 3.2 tid2eid 只负责专家选择

DeepSeek-V4 前 n_hash_layers 层使用 tid2eid：

~~~text
input_ids
  └─ tid2eid[input_ids]
       └─ selected_experts_indices
~~~

非 hash 层则通过 router score 的 top-k 得到 selected_experts_indices。

无论专家编号来自 tid2eid 还是 top-k，后续 activation 的数学形式都相同。hash
层的差异属于 router 选择策略，不会要求每层使用不同的 SwiGLU 实现。

### 3.3 通用 NPU MoE dispatch 路径

通用 _run_local_experts 会分别重排 token 和 score，使每个 routed row 与自己的
score 对齐：

~~~text
tokens + selected_experts_indices
  └─ npu_moe_token_permute
       └─ routed_input

top_scores + selected_experts_indices
  └─ npu_moe_token_permute
       └─ routed_scores

NpuGroupedExperts(
    routed_input,
    num_tokens_per_expert,
    routed_scores,
)
  ├─ GMM-1：w13
  ├─ activation(h, swiglu_limit, routed_scores)
  └─ GMM-2：w2

npu_moe_token_unpermute
  └─ top-k 求和并恢复 token 顺序
~~~

在这条通用路径中，swiglu_group_activation 可以把 routed_scores 映射为
SwigluGroup 的 weight。

### 3.4 DeepSeek-V4 当前专用路径

当前 NpuDeepSeekV4MoE.forward 与通用路径不同。它通过模型自身的 reorderer
得到排序后的 token 和 score，然后只把 routed_input 与 counts 传给 experts：

~~~text
router
  ├─ top_scores
  ├─ selected_experts_indices
  └─ num_tokens_per_expert

reorderer
  ├─ top_scores_experts_sorted
  └─ token_indices_experts_sorted

routed_input
  └─ experts(routed_input, num_tokens_per_expert)
       ├─ GMM-1
       ├─ SwigluGroup(weight=None)
       └─ GMM-2

routed_output
  └─ 乘 top_scores_experts_sorted
       └─ scatter_add
~~~

DeepSeek-V4 当前默认 score_before_experts=False，因此 routed score 在专家输出后
单独相乘。也就是说：

- swiglu_group_activation 的统一签名支持 routed_scores；
- 但当前 DeepSeek-V4 模型路径调用 NpuGroupedExperts 时没有传第三个参数；
- 因而这条路径中的 SwigluGroup 实际收到 weight=None；
- 文档不能笼统写成“DeepSeek-V4 已把 routed-score mul 融进 SwigluGroup”。

如果后续要把 DeepSeek-V4 的 score 乘法也融合进算子，必须同时审计
score_before_experts、dispatch/reorder 布局、数值顺序和梯度，不能只在
swiglu_group.py 内部假设 score 已经传入。

## 4. 路由专家的 activation 交接点

### 4.1 默认实现

npu_gmm 把 GroupedExperts 替换成 NpuGroupedExperts，并默认设置：

~~~python
self._expert_activation_fn = _expert_activation
self._expert_activation_compile_key = None
~~~

默认 _expert_activation 执行：

~~~text
可选 clamp
  → torch_npu.npu_swiglu
  → 可选 routed_scores 乘法
~~~

因此只启用 npu_moe_dispatch + npu_gmm 时，不会自动加载或调用 CANN
SwigluGroup。

### 4.2 set_expert_activation 的作用

gmm.py 对后续 activation converter 提供：

~~~python
def set_expert_activation(self, activation_fn):
    self._expert_activation_fn = activation_fn
    self._expert_activation_compile_key = None
~~~

这是路由专家结构与具体 activation 实现之间的公开交接点，作用有两层：

1. 把两次 GMM 之间的 callable 从 native 实现切换为融合实现；
2. 清空旧 compile key，防止已经针对旧 activation 构建的图被误认为仍有效。

npu_swiglu_group 调用这个方法时，不会：

- 再次包装 NpuGroupedExperts；
- 修改 w13/w2；
- 改变 state-dict key；
- 重复注册 GMM state-dict updater。

### 4.3 compile_expert_activation 与 experts[0]

compile_expert_activation 会收集模型中的所有 NpuGroupedExperts，使用第一个模块
当前持有的 activation_fn 构建一次 compiled callable，然后写回全部模块。

当前这样做成立的前提是：

- npu_gmm 给所有路由专家安装同一个 native activation；
- npu_swiglu_group 也给所有路由专家安装同一个 fused activation；
- backend 和 dynamic_tokens 配置是全局一致的。

因此 experts[0] 代表的是“当前全局统一的 activation 实现”，不是第 0 个专家，
也不是共享专家，更不是 router 的第 0 层配置。

hash 路由层与普通 top-k 层的差异只改变 selected_experts_indices，不改变
activation callable，所以不影响这个假设。如果未来出现按层不同的 activation、
clamp 策略或编译选项，就不能继续只取 experts[0]；届时应按 callable/config 分组
编译，或显式校验所有模块配置一致。

## 5. 共享专家接入

### 5.1 通用结构放在 common MoE 层

NpuSharedExperts 定义在：

~~~text
torchtitan_npu/models/common/moe.py
~~~

而不是 swiglu_group.py。它的结构性 forward 固定为：

~~~python
packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
hidden = activation_fn(
    packed,
    getattr(self, "swiglu_limit", None),
)
return self.w2(hidden)
~~~

默认 activation 是 native_shared_expert_activation，使用 PyTorch 小算子完成
可选 clamp 和 SwiGLU。因此共享专家适配本身不依赖 A5，也不依赖 cann_ops_nn。

### 5.2 由 npu_moe_dispatch 安装适配器

NpuMoeDispatchConverter 在转换 MoE 时调用 NpuSharedExperts.convert：

~~~text
FeedForward / DeepSeekV4FeedForward
  └─ 原地转为 NpuSharedExperts
       └─ 默认 native_shared_expert_activation
~~~

convert 通过修改现有对象的类完成适配，并保留：

- w1、w2、w3 子模块和参数对象；
- module identity；
- FQN 和 state-dict key；
- buffer；
- forward/backward hook；
- training/eval 状态；
- swiglu_limit 等已有属性。

这一步属于 MoE 通用层，不属于某个融合算子 converter。

### 5.3 npu_swiglu_group 只切换 callable

融合 converter 对共享专家只执行：

~~~python
module.set_expert_activation(swiglu_group_shared_activation)
~~~

swiglu_group_shared_activation 再调用统一入口，但不传 routed_scores：

~~~text
w1(x) ─┐
       ├─ cat → SwigluGroup(weight=None) → w2
w3(x) ─┘
~~~

所以共享专家没有 routed-score 梯度，也不需要任何 router 信息。

### 5.4 为什么保留 w1/w2/w3

共享专家没有改成 w13，主要因为：

1. checkpoint 与 HF 映射使用现有 key；
2. TP plan 和其他 converter 可能按 shared_experts.w1/w2/w3 查找模块；
3. 参数对象和 FQN 改动会扩大兼容性风险；
4. 本次目标是选择 activation 实现，不是重排共享专家权重。

## 6. SwigluGroup 调用与 Autograd

### 6.1 当前 Python 入口

路由专家的统一入口为：

~~~python
def swiglu_group_activation(h, swiglu_limit=None, routed_scores=None):
    weight = (
        None
        if routed_scores is None
        else routed_scores.to(dtype=torch.float32).contiguous()
    )
    clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
    return torch.ops.cann_ops_nn.swiglu_group.default(
        h.contiguous(),
        weight=weight,
        group_index=None,
        clamp_limit=clamp_limit,
    )
~~~

关键约束：

- h 显式 contiguous；
- routed_scores 存在时转为 FLOAT32 并 contiguous；
- group_index 固定为 None；
- 无 clamp 使用 -1.0 sentinel；
- torchtitan-npu 不直接调用 backward。

### 6.2 Autograd 已移到 ops-nn

旧实现曾在 torchtitan-npu 中定义 _NpuSwigluGroup(torch.autograd.Function)，并在
backward 中直接调用独立反向算子。当前这层桥已经删除。

现在调用链是：

~~~text
torchtitan-npu
  └─ torch.ops.cann_ops_nn.swiglu_group.default
       └─ ops-nn 注册的 SwigluGroup Autograd
            └─ cann_ops_nn::swiglu_group_backward
~~~

因此职责边界变为：

- torchtitan-npu 只准备模型侧输入并选择算子；
- ops-nn 保存反向所需上下文并连接 backward dispatcher；
- CANN 算子计算 grad_x 和可选 grad_weight。

这要求运行环境安装包含 SwigluGroup Autograd 注册的 cann_ops_nn 版本。若仍出现：

~~~text
an autograd kernel was not registered to the Autograd key(s)
~~~

说明实际加载的包仍是旧实现或注册未生效，不能通过 torchtitan-npu 再包一层长期掩盖。

### 6.3 y_origin 的正确语义

带 weight 的前向为：

~~~text
y_origin = silu(clamped gate) * clamped up
y = y_origin * weight
~~~

grad_weight 需要未乘 weight 的 y_origin。因此 ops-nn 的 Autograd 实现必须保证：

- weight=None 时，backward 同时收到 y_origin=None；
- weight 不为 None 时，y_origin 是未加权激活；
- 不能把已加权输出 y 直接当作 y_origin，否则 grad_weight 会额外乘一次 weight；
- 也不能通过 y / weight 恢复，因为 weight 可能为零；
- 即使 x 不需要梯度，只要 weight 需要梯度，也必须进入 Autograd 路径。

这些是 ops-nn 的实现责任。当前 torchtitan-npu 不保存、不重算也不传递 y_origin。

### 6.4 ensure_swiglu_group_ops 能检查什么

converter 加载 cann_ops_nn.ops 后检查：

~~~text
torch.ops.cann_ops_nn.swiglu_group.default
torch.ops.cann_ops_nn.swiglu_group_backward.default
~~~

这能提前发现 dispatcher 缺失，但不能单独证明 Autograd key 注册正确。真实训练还应
执行一次 requires_grad 前向和 backward，并确认没有 fallback warning。

## 7. A5 和软件边界

当前设备类型映射包括：

| 设备标识 | 类型 |
| --- | --- |
| Ascend950DT、Ascend950PR、Ascend910_95、Ascend950 | A5 |
| Ascend910_93 | A3 |
| Ascend910B | A2 |

NpuSwigluGroupConverter 只允许 A5。非 A5 时在导入 CANN 算子前抛出 ValueError。

这意味着：

- A3/A2 可继续使用 npu_moe_dispatch + npu_gmm 的 native activation；
- A5 只有显式添加 npu_swiglu_group 才切换融合 activation；
- common MoE 层和 gmm 默认路径不会因为 SwigluGroup 不可用而失效；
- CANN dispatcher 缺失时 converter 在模型 mutation 前失败。

## 8. torch.compile 与 activation checkpointing

本文涉及 activation-only compile；关于 Dynamo backend、Inductor 内部 NPU Codegen、decomposition
和 TorchTitan `component` 的分层说明，参见 [[PyTorch编译问题总结：Dynamo、Inductor与NPU Codegen]]。

### 8.1 activation-only compile

gmm.py 只编译两次 GMM 中间的 activation bridge：

~~~python
compiled_activation = torch.compile(
    activation_fn,
    backend=backend,
    fullgraph=True,
    options={"custom_partitioner_fn": _NpuGmmAotDefaultPartitioner()},
)
~~~

dynamic_tokens=True 时，会把 h 和可选 routed_scores 的第 0 维标记为动态。

set_expert_activation 清空 compile key 后，后续 parallelize 阶段会对新注入的
SwigluGroup activation 重新编译。

### 8.2 selective activation checkpointing

DeepSeek-V4 parallelize 通过是否启用 npu_gmm 判断 grouped-MM 路径，而不是通过
npu_swiglu_group 判断。原因是：

- GMM 结构由 npu_gmm 决定；
- npu_swiglu_group 只改变中间 activation；
- selective AC 保存 grouped-MM 输出的策略不应依赖某个可选 activation 后端。

## 9. State dict、参数与模型兼容性

当前设计对 checkpoint 的影响如下：

### 路由专家

- npu_gmm 负责 w1/w3 与 w13 的映射；
- GMMStateDictUpdater 仍只注册在 npu_gmm；
- npu_swiglu_group 不新增 state-dict updater。

### 共享专家

- 保留 w1/w2/w3；
- 保留参数对象和 FQN；
- 只切换 _expert_activation_fn；
- state-dict key 不随是否启用 SwigluGroup 改变。

### 模型代码

DeepSeek-V4 模型只保存模型本身的语义，例如 swiglu_limit、router 和 shared
experts 配置；A5/CANN 依赖不进入模型 forward。

## 10. 当前实现与旧实现对照

| 旧描述 | 当前实现 |
| --- | --- |
| 配置名 npu_gmm_swiglu | npu_swiglu_group |
| 文件 gmm_swiglu.py | converters/kernels/swiglu_group.py |
| fusion converter 定义 NpuSharedExperts | NpuSharedExperts 位于 models/common/moe.py |
| fusion converter 原地转换共享专家 | npu_moe_dispatch 安装通用共享专家适配器 |
| fusion converter 重新定义共享专家 forward | 只注入 swiglu_group_shared_activation |
| torchtitan-npu 定义 _NpuSwigluGroup Autograd Function | 直接调用公开 op，Autograd 由 ops-nn 注册 |
| 调用 swiglu_group_quant_backward | 检查并依赖 swiglu_group_backward |
| 空 Tensor 走 PyTorch 分解回退 | 当前无特殊分支，交由 CANN 合约处理 |
| DeepSeek-V4 routed score 一定融合进 SwigluGroup | 当前 DSV4 专用路径在专家输出后单独乘 score |
| npu_swiglu_group 同时承担 GMM 与 activation | npu_gmm 管结构，npu_swiglu_group 只选 activation |

## 11. 当前调用链

### 11.1 原生路由专家

~~~text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
       └─ _run_experts_grouped_mm
            ├─ offsets = cumsum(num_tokens_per_expert, int32)
            ├─ torch._grouped_mm(x, w13)
            ├─ _expert_activation
            │    ├─ 可选 clamp
            │    ├─ torch_npu.npu_swiglu
            │    └─ 可选 score mul
            └─ torch._grouped_mm(hidden, w2)
~~~

### 11.2 融合路由专家

~~~text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
       └─ _run_experts_grouped_mm
            ├─ torch._grouped_mm(x, w13)
            ├─ swiglu_group_activation
            │    └─ cann_ops_nn::swiglu_group
            └─ torch._grouped_mm(hidden, w2)
~~~

在通用 MoE dispatch 中 routed_scores 可作为 weight 传入；在当前
NpuDeepSeekV4MoE 路径中 routed_scores 未传入 experts，score 在 GMM-2 后相乘。

### 11.3 共享专家

~~~text
NpuSharedExperts.forward
  ├─ gate = w1(x)
  ├─ up = w3(x)
  ├─ packed = cat(gate, up)
  ├─ swiglu_group_shared_activation
  │    └─ cann_ops_nn::swiglu_group(weight=None)
  └─ output = w2(hidden)
~~~

### 11.4 反向

~~~text
loss.backward
  └─ ops-nn 为 cann_ops_nn::swiglu_group 注册的 Autograd
       └─ cann_ops_nn::swiglu_group_backward
            ├─ grad_x
            └─ 可选 grad_weight
~~~

torchtitan-npu 的 swiglu_group.py 中不再有 backward 方法。

## 12. 如何验证接入成功

### 12.1 结构和 converter

确认：

- 配置顺序为 npu_moe_dispatch、npu_gmm、npu_swiglu_group；
- 设备类型识别为 A5；
- 路由专家类型为 NpuGroupedExperts；
- 路由专家的 _expert_activation_fn 为 swiglu_group_activation；
- shared_experts 类型为 NpuSharedExperts；
- 共享专家的 _expert_activation_fn 为 swiglu_group_shared_activation；
- dense feed_forward 未被替换；
- state-dict key 和参数 identity 保持不变。

### 12.2 Dispatcher 和 Autograd

建议在目标 A5 环境检查：

~~~python
print(torch._C._dispatch_dump_table("cann_ops_nn::swiglu_group"))
~~~

并执行至少以下梯度场景：

1. weight=None，x 需要梯度；
2. weight 不为 None，x 和 weight 都需要梯度；
3. weight 中包含零值；
4. 只有 weight 需要梯度；
5. clamp_limit=-1.0；
6. clamp_limit 为正数。

需要对比 PyTorch 分解基线的 forward、grad_x 和 grad_weight，并确认没有
Autograd fallback warning。

### 12.3 单元测试

当前仓库相关测试为：

~~~bash
pytest -q \
  tests/unit_tests/converters/test_moe_dispatch.py \
  tests/unit_tests/converters/test_swiglu_group.py \
  tests/unit_tests/models/test_deepseek_v4_gmm_compile.py \
  tests/unit_tests/converters/test_registry.py
~~~

重点覆盖：

- 参数映射、FLOAT32 weight、contiguous 和 no-clamp sentinel；
- 直接 op 的 Autograd 行为；
- 空输入异常透传；
- converter 顺序和 A5 限制；
- 缺少 dispatcher 时不发生部分注入；
- 共享专家原地适配及 native/fused callable 切换；
- activation compile 的共享、幂等和动态 token；
- state-dict key、参数、hook 与 training mode 保持。

### 12.4 Profiling

路由专家融合区间应看到：

~~~text
GMM-1 → SwigluGroup → GMM-2
~~~

共享专家应看到：

~~~text
w1/w3 matmul → cat → SwigluGroup → w2 matmul
~~~

对于 DeepSeek-V4，还应在 GMM-2 之后看到 routed score 乘法；当前不能把这一步的
存在误判为 SwigluGroup 融合失败。

反向 kernel 的准确名称以实际 CANN timeline 为准，但 Python/dispatcher 语义应对应
swiglu_group_backward。

### 12.5 内存与 OOM

仅凭 OOM 栈顶位于 FSDP reduce-scatter 或 aclnnDivs，不能证明分配发生在
SwigluGroup 内部，因为 NPU 默认异步执行。

比较新旧实现时需要：

- 相同进程数、并行策略、checkpoint、随机种子和 batch/sequence；
- 在目标区间前后 synchronize；
- 分别记录 allocated、reserved 和 peak；
- 必要时临时设置 ASCEND_LAUNCH_BLOCKING=1 定位真实失败算子；
- 结合 timeline 区分 op workspace、Autograd 保存张量、checkpoint 重算和通信 buffer。

移除 torchtitan-npu 局部 Autograd 桥后，是否降低峰值仍应以实测为准，不能仅凭
“新版本能跑通”直接断定旧桥就是 OOM 根因。

## 13. 本轮验证状态

本次文档更新依据当前分支代码、测试和已推送的 MR diff。开发机当前 Python 环境
缺少 torch/torch_npu，因此没有在本机重新执行 NPU 单元测试和 A5 训练。

合入前仍应在匹配 ops-nn PR 8085 的 A5 环境完成：

1. 真实 SwigluGroup forward/backward；
2. weight=None 和 weight 非空的梯度对齐；
3. weight-only requires_grad；
4. DeepSeek-V4 前几步 loss/grad_norm 与基线对齐；
5. eager/compile；
6. 峰值内存和稳态性能对比。

## 14. 主要代码位置

| 文件 | 当前职责 |
| --- | --- |
| torchtitan_npu/converters/kernels/swiglu_group.py | A5 检查、dispatcher 检查、融合 activation 与 converter |
| torchtitan_npu/converters/kernels/gmm.py | 路由专家 GMM、默认 activation、公开 setter、compile、state-dict 映射 |
| torchtitan_npu/converters/kernels/moe_dispatch.py | MoE dispatch 和共享专家通用适配 |
| torchtitan_npu/models/common/moe.py | NpuSharedExperts 与 native shared activation |
| torchtitan_npu/models/deepseek_v4/moe.py | DeepSeek-V4 router、swiglu_limit 和模型数学语义 |
| tests/unit_tests/converters/test_swiglu_group.py | SwigluGroup 接线与 converter 测试 |
| tests/unit_tests/converters/test_moe_dispatch.py | 共享专家默认适配测试 |
| tests/unit_tests/models/test_deepseek_v4_gmm_compile.py | GMM activation compile 测试 |

## 参考资料

- [torchtitan-npu Merge Request !468](https://gitcode.com/cann/torchtitan-npu/merge_requests/468)
- [ops-nn PR 8085](https://gitcode.com/cann/ops-nn/pull/8085)
- [torchtitan-npu NPU fused ops 文档](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/feature_guides/npu_fused_ops.md)
- [PyTorch Autograd 文档](https://docs.pytorch.org/docs/stable/autograd.html)
