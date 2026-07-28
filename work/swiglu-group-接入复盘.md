# DeepSeek-V4 接入 `SwigluGroup` 开发复盘

> 本文基于 `torchtitan-npu` 的 `feat/swiglu-group-a5-fusion` 分支整理。当前实现整合提交为
> `d290fdf`，文档精简提交为 `8d59053`，共享专家复用统一激活入口的重构提交为 `b67fe80`；
> 当前 converter 拆分完成于 `4c4b633`。
>
> 最后更新：2026-07-28。

## 1. 背景与目标

### 1.1 SwiGLU 是什么

SwiGLU（Swish-Gated Linear Unit）是 Transformer FFN 中常用的门控激活。以
DeepSeek-V4 的前馈网络为例，输入 `x` 分别经过两条升维线性层：

```text
gate = w1(x)
up   = w3(x)
```

随后计算：

```text
hidden = silu(gate) * up
output = w2(hidden)
```

其中：

- `w1` 产生门控分支；
- `w3` 产生被门控的数值分支；
- `silu(gate)` 提供平滑的非线性门控；
- `w2` 把中间维度投影回模型 hidden size。

如果把 `gate` 和 `up` 沿最后一维拼成一个 Tensor：

```text
packed = [gate | up]
```

普通 SwiGLU 算子会把最后一维均分为 `A`、`B`，再计算：

```text
y = silu(A) * B
```

因此在本次接入中必须保持 `w1(x)` 在前、`w3(x)` 在后，否则会改变模型数学语义。

### 1.2 `SwigluGroup` 比普通 SwiGLU 多做什么

CANN `SwigluGroup` 不仅实现 `silu(A) * B`，还支持把 clamp 和 token 权重乘法一并放进
算子。其逻辑可以概括为：

```text
x = [A | B]

if clamp_limit > 0:
    A = min(A, clamp_limit)
    B = min(max(B, -clamp_limit), clamp_limit)

y_origin = silu(A) * B

if weight is not None:
    y = y_origin * weight
else:
    y = y_origin
```

主要参数如下：

| 参数 | 作用 | 本次接入中的用法 |
| --- | --- | --- |
| `x` | 最后一维为 `[A \| B]` 的输入，输出最后一维减半 | 两次 GMM 之间的 `w13` 输出，或共享专家 `w1/w3` 输出的拼接结果 |
| `weight` | 对每个 token 的激活结果乘权 | 路由专家传 `routed_scores.float()`；共享专家传 `None` |
| `group_index` | count 模式下各分组的 token 数量 | 当前两条路径均传 `None` |
| `clamp_limit` | 激活前的截断阈值 | 路由专家和 DeepSeek-V4 共享专家传模型的 `swiglu_limit`；没有该属性时传 `-1.0` 表示不 clamp |

官方算子说明同时列出 Ascend 950PR 和 Ascend 950DT 为支持产品。参见
[ops-nn `SwigluGroup` 文档](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group/README.md)。

### 1.3 路由专家与共享专家

DeepSeek-V4 的 MoE 层不只有一种专家。

#### 路由专家（routed experts）

Router 为每个 token 选择若干专家，并产生对应的 routed score。token 会按专家重排，然后
进入分组专家计算：

```text
token
  ├─ router 选择专家并产生 routed score
  ├─ dispatch / token reorder
  ├─ 专家 w1/w3
  ├─ SwiGLU
  ├─ 乘 routed score
  ├─ 专家 w2
  └─ combine
```

不同 token 可能进入不同专家，所以路由专家需要 GMM、token 计数、dispatch/combine 等
MoE 基础设施。

#### 共享专家（shared experts）

共享专家不由 router 选择。通常每个 token 都会经过同一套共享 FFN：

```text
token
  └─ shared_experts
      ├─ w1
      ├─ w3
      ├─ SwiGLU
      └─ w2
```

它的作用是为所有 token 提供稳定的公共知识通路，补充稀疏路由专家只处理部分 token 的
路径。共享专家与 SwiGLU 没有绑定关系；它本质上是一个普通 `FeedForward`，只是该 FFN
内部恰好使用 SwiGLU，因此也可以接入 `SwigluGroup`。

### 1.4 接入前的路由专家执行路径

`npu_gmm` 已经把专家 `w1/w3` 和 `w2` 替换为两次 grouped matmul。融合前，两次 GMM
中间仍是多个 PyTorch/NPU 操作：

```text
token rows
  └─ GMM-1：x @ w13
      └─ h = [gate | up]
          ├─ chunk gate/up
          ├─ clamp(up, -limit, limit)
          ├─ clamp(gate, max=limit)
          ├─ cat(gate, up)
          ├─ npu_swiglu
          ├─ cast routed_scores
          └─ mul routed_scores
              └─ GMM-2：hidden @ w2
```

其中 `w13` 是路由专家内部由 `w1`、`w3` 融合得到的参数。多个 elementwise 算子位于两个
计算量较大的 GMM 之间，带来额外的 kernel 调度和中间 Tensor 读写。

### 1.5 接入前的共享专家执行路径

共享专家保留独立的 `w1`、`w3`。DeepSeek-V4 在接入融合算子前已通过
`DeepSeekV4FeedForward` 对 gate/up 应用与路由专家相同的 `swiglu_limit`：

```text
x
  ├─ w1(x) ──> clamp(gate, max=limit) ──> silu ──┐
  └─ w3(x) ──> clamp(up, -limit, limit) ─────────┼─> mul ─> w2
                                                 ┘
```

共享专家没有 routed score，也没有路由分组信息。这里可融合的是 clamp、SwiGLU 及其内部
乘法，线性层本身不属于 `SwigluGroup` 的职责。对于没有 `swiglu_limit` 属性的其他共享
`FeedForward`，融合入口使用 `-1.0` 保持原有无 clamp 语义。

### 1.6 本次接入目标

本次接入包含四个连续阶段：

1. 在 Ascend 950 上，把路由专家两次 GMM 之间的 clamp、SwiGLU 和 routed-score 乘法
   替换为一个 `SwigluGroup` 前向算子。
2. 复用同一 converter，把共享专家的 clamp 与 SwiGLU 激活替换为 `SwigluGroup`，同时
   保留 `w1/w2/w3` 参数结构，并透传可选的 `swiglu_limit`。
3. 在 CANN 9.2.0 上显式桥接 `swiglu_group_backward`，使 eager 和
   `torch.compile` 训练都能执行反向传播。
4. 将 A5 专属能力从通用 `npu_gmm` 拆到替代型 converter
   `npu_gmm_swiglu`：只有显式选择新名称才走融合路径。

同时需要保护以下既有路径：

- `npu_gmm` 在 A3、A5 等平台都只使用经过验证的拆分激活；
- 融合路径通过新配置名 `npu_gmm_swiglu` 显式启用，两者不同时配置；
- 不改变模型 checkpoint 和 state dict key；
- 不把共享专家的 `w1/w3` 合并成 `w13`；
- 本次不接入量化前向或量化反向接口。

## 2. 硬件与软件接口边界

### 2.1 Ascend 950PR 与 Ascend 950DT

根据华为公开的 Ascend 路线说明，950PR 与 950DT 使用相同的 Ascend 950 Die，但面向的
主要场景和 HBM 配置不同：

| 产品 | 主要定位 | 公开说明中的内存侧重点 |
| --- | --- | --- |
| Ascend 950PR | Prefill、推荐 | 面向计算密集型 Prefill/推荐，采用 HiBL 1.0 |
| Ascend 950DT | Decode、训练 | 面向访存和互联带宽要求更高的 Decode/训练，采用 HiZQ 2.0，公开规格为 144 GB、4 TB/s |

来源：[华为《以开创的超节点互联技术，引领AI基础设施新范式》](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech)。

这个产品定位不等于代码层必须维护两套实现。对本算子而言：

- CANN 文档声明 `SwigluGroup` 同时支持 950PR 和 950DT；
- `torchtitan-npu` 把两种设备都归类为能力代号 `A5`；
- `npu_gmm_swiglu` 只校验“是否具备 A5 算子能力”，不根据 PR/DT 分叉；
- `npu_gmm` 不再做设备判断，也不加载 `cann_ops_nn`。

当前设备映射为：

```python
_NPU_DEVICE_TYPE_MAP = {
    "Ascend950DT": "A5",
    "Ascend950PR": "A5",
    "Ascend910_95": "A5",
    "Ascend950": "A5",
    "Ascend910_93": "A3",
    "Ascend910B": "A2",
}
```

因此在日志中看到 `Ascend950PR_9589` 时，`get_npu_device_type()` 会返回 `A5`，
满足 `npu_gmm_swiglu` 的硬件校验；是否融合仍取决于用户是否显式配置该 converter。

### 2.2 以 dispatcher schema 判断 CANN 能力

CANN 包的 Python distribution metadata 在实测环境中仍显示 `1.0.0`，不能据此判断
9.2.0 是否提供新接口。更可靠的方式是直接查询 PyTorch dispatcher schema：

```python
torch._C._dispatch_find_schema_or_throw(
    "cann_ops_nn::swiglu_group",
    "",
).schema()
```

CANN 9.2.0 实测前向 schema 为：

```text
cann_ops_nn::swiglu_group(
    Tensor x,
    *,
    Tensor? weight=None,
    Tensor? group_index=None,
    float clamp_limit=-1.
) -> Tensor
```

实测反向 schema 为：

```text
cann_ops_nn::swiglu_group_backward(
    Tensor grad_output,
    Tensor x,
    *,
    Tensor? weight=None,
    Tensor? y_origin=None,
    Tensor? group_index=None,
    float clamp_limit=0.
) -> (Tensor, Tensor?)
```

同时观察到：

```text
swiglu_group_grad              NOT REGISTERED
swiglu_group_backward          PrivateUse1: True
swiglu_group_quant_grad        NOT REGISTERED
swiglu_group_quant_backward    PrivateUse1: True
```

也就是说，9.2.0 对外入图接口已从 `*_grad` 改为 `*_backward`，与 PTA 侧命名保持一致。
接入代码应检测真实 dispatcher 入口，而不是根据包名、metadata version 或旧接口名称推断。
以上 schema、dispatch key 和 distribution metadata 均属于目标 A5/CANN 9.2.0 环境的实测
结果，不是 `ops-nn` 当前 `master` README 对 Python dispatcher 的稳定承诺。复现时应同时
记录具体 CANN build、设备和命令输出。

### 2.3 前向算子没有自动 Autograd 注册

实测 `cann_ops_nn::swiglu_group` 的关键 dispatch 状态为：

```text
PrivateUse1:              True
Autograd:                 False
AutogradPrivateUse1:      False
CompositeExplicitAutograd: False
CompositeImplicitAutograd: False
```

这说明设备前向 kernel 已注册，但 PyTorch 不知道应该如何从该前向节点走到
`swiglu_group_backward`。仅仅把旧字符串改为新字符串不能解决训练反向问题，插件侧必须
建立显式的 Autograd 桥接。

## 3. 方案设计与取舍

### 3.1 Converter 方案对比

| 方案 | 做法 | 优点 | 问题 | 结论 |
| --- | --- | --- | --- | --- |
| 新增 activation-only `npu_swiglu_group` converter | 要求与 `npu_gmm` 叠加 | 激活边界表面上独立 | 执行顺序和共享专家转换容易与 GMM 路径分离 | 不采用 |
| 在 DeepSeek-V4 模型中硬编码 A5 判断 | 模型 forward 直接调用 CANN 算子 | 改动直观 | 模型代码耦合 NPU/CANN；绕过 converter 注册机制；不利于复用 | 不采用 |
| 让 `npu_gmm` 按设备自动切换 | 在通用 converter 中同时放入 A5 融合与回退 | 无需新配置 | A5-only 依赖进入默认 GMM 路径，能力边界不清晰 | 重构后不采用 |
| 新增替代型 `npu_gmm_swiglu` converter | 自身完成 GMM 转换并注入融合激活 | 不需要与 `npu_gmm` 叠加；A5 依赖显式隔离 | 用户需将 converter 名从 `npu_gmm` 替换为新名称 | 采用 |

启用融合时将原来的 `npu_gmm` 替换为：

```python
get_model_converter_config("npu_gmm_swiglu")
```

`NpuGmmSwigluConverter` 会覆盖：

- 路由 `GroupedExperts` 的 GMM 与中间 `SwigluGroup`；
- 名为 `shared_experts` 的公共 `FeedForward` 激活替换。

### 3.2 最终结构

```text
npu_gmm_swiglu converter（仅 A5）
    │
    ├─ GroupedExperts
    │    └─ NpuGroupedExperts
    │         ├─ GMM-1
    │         ├─ SwigluGroup Autograd bridge
    │         └─ GMM-2
    │
    └─ *.shared_experts: FeedForward       （仅 A5）
         └─ NpuSharedExperts
              ├─ 原 w1 / w3
              ├─ cat
              ├─ SwigluGroup Autograd bridge
              └─ 原 w2
```

配置与硬件规则为：

```text
npu_gmm
  └─ 不判断设备：使用既有拆分激活，不加载 cann_ops_nn

npu_gmm_swiglu
  ├─ A5：路由专家和共享专家使用 SwigluGroup
  └─ 非 A5：转换前抛出 ValueError，不静默回退
```

### 3.3 为什么共享专家保留 `w1/w3`

共享专家接入采用“保留线性层，只替换激活”的方案：

```text
w1(x) ─┐
       ├─ cat ─> SwigluGroup ─> w2
w3(x) ─┘
```

没有把 `w1`、`w3` 融合为路由专家式的 `w13`，原因是：

1. 用户明确要求保留 `w1/w3`。
2. 共享专家已有 TP plan 以 `w1/w2/w3` 的 FQN 定位模块。
3. checkpoint 和 HF state dict 使用既有参数名。
4. MXFP8 等其他 converter 可能按原 FQN 查找三个 Linear。
5. 本次目标是激活融合，不应顺带改变参数布局和加载语义。

这属于“保护已验证路径”的取舍：虽然 `cat` 会产生一个中间 Tensor，但避免了更大范围的
模型结构、并行和 checkpoint 改造。

### 3.4 接入范围与非目标

本次覆盖：

- Ascend 950PR/950DT（代码能力类型 A5）；
- 路由专家前向与一阶反向；
- 共享专家前向与一阶反向；
- eager 与 `inductor_npu` fullgraph；
- routed-score 梯度；
- 有 clamp 和无 clamp 两种场景。

本次不覆盖：

- `swiglu_group_quant`；
- `swiglu_group_quant_backward`；
- 二阶梯度；
- 非专家 dense FFN 的批量替换；
- 非 A5 平台强制使用新算子；
- 把共享专家 `w1/w3` 合并成新参数。

## 4. 路由专家接入

### 4.1 两次 GMM 之间的融合点

路由专家核心函数为 `_run_experts_grouped_mm()`。它不在函数内部判断设备，而是接收一个
统一签名的 `activation_fn`：

```python
h = torch._grouped_mm(
    x.bfloat16(),
    w13.bfloat16().transpose(-2, -1),
    offs=offsets,
)

h = activation_fn(h, swiglu_limit, routed_scores)

out = torch._grouped_mm(
    h,
    w2.bfloat16().transpose(-2, -1),
    offs=offsets,
).type_as(x)
```

`npu_gmm` 构造 `NpuGroupedExperts` 时注入 `_expert_activation`；
`npu_gmm_swiglu` 注入 `_swiglu_group_activation`。选择由 converter 名称决定，
不在 `NpuGroupedExperts` 内再判断设备。`SwigluGroup` 仍位于升维 GMM 和降维 GMM 之间。

### 4.2 A5 融合路径

融合路径的核心逻辑为：

```python
weight = (
    None
    if routed_scores is None
    else routed_scores.to(dtype=torch.float32).contiguous()
)
clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)

return _swiglu_group_op()(
    h.contiguous(),
    weight=weight,
    group_index=None,
    clamp_limit=clamp_limit,
)
```

这里有四个重要语义。

#### `h` 必须 contiguous

GMM 输出在进入自定义算子前显式整理为连续内存，避免算子对 stride 的隐式假设与实际输入
不一致。

#### `routed_scores` 必须为 FLOAT32

官方约束要求 `weight` 为 FLOAT32，因此即使激活为 BF16，也会先执行：

```python
routed_scores.to(dtype=torch.float32).contiguous()
```

不能为了少一次 cast 直接把 BF16 score 传给算子。

#### `group_index=None`

前面的 dispatch 和 GMM 已经只物化有效 routed rows，`num_tokens_per_expert` 则用于计算
GMM offsets。中间激活对这些有效行逐行处理即可，不需要再次通过 `group_index` 截断，
所以当前调用传 `None`。

`SwigluGroup` 名称中的 “Group” 并不意味着每次都必须传 `group_index`。该参数是可选的，
当前性能收益主要来自 clamp、SwiGLU 和 weight 乘法融合。

#### `swiglu_limit=None` 映射为前向 `-1.0`

前向接口使用：

```text
clamp_limit = -1.0  → 不执行 clamp
clamp_limit > 0     → 执行 clamp
```

因此 Python 侧的 `None` 不能直接传入 float 属性，而是映射为算子约定的 `-1.0`。

### 4.3 `npu_gmm` 拆分路径

`npu_gmm` 在所有支持平台都执行原有逻辑：

```python
if swiglu_limit is not None:
    gate, up = h.chunk(2, -1)
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)
    h = torch.cat([gate, up], dim=-1)

h = torch_npu.npu_swiglu(h, dim=-1)

if routed_scores is not None:
    h = h * routed_scores.to(h.dtype)
```

这条路径不导入 `cann_ops_nn`。非 A5 用户应选择 `npu_gmm`；如果显式误选
`npu_gmm_swiglu`，converter 会直接报错，而不是在同一 converter 内静默回退。

### 4.4 路由专家的理论融合收益

融合前需要执行：

```text
chunk/views + clamp(gate) + clamp(up) + cat + npu_swiglu + cast/mul
```

融合后核心变为：

```text
SwigluGroup(clamp + silu + gate multiply + routed-score multiply)
```

预期收益来自：

- 减少多个 elementwise kernel 的下发开销；
- 减少 clamp、SwiGLU 和 score 乘法之间的中间 Tensor 写回与再次读取；
- 让两个 GMM 之间的激活区间更紧凑。

但端到端收益仍受 GMM、通信、dispatch/combine、activation checkpointing 和 shape 影响，
不能仅凭“少了几个算子”直接承诺固定加速比例。

## 5. 共享专家接入

### 5.1 共享专家替换入口

共享专家在模型中是普通 `FeedForward`，不能用 `isinstance(module, FeedForward)` 无差别
替换，否则 dense transformer block 的 FFN 也会被改变。最终匹配条件和转换方式是：

```python
elif (
    isinstance(module, FeedForward)
    and name.rsplit(".", 1)[-1] == "shared_experts"
):
    NpuSharedExperts.convert(module)
```

也就是同时满足：

1. `npu_gmm_swiglu` 已在遍历模块前校验设备是 A5；
2. 模块类型是公共 `FeedForward`；
3. 模块路径最后一段精确等于 `shared_experts`。

这个规则比写死完整 DeepSeek-V4 FQN 更通用，又不会把所有普通 FFN 都替换掉。`npu_gmm_swiglu`
在遍历模块前已统一调用 `_load_swiglu_group_ops()`，这里不再单独加载算子。

### 5.2 复用原模块和参数

共享专家采用原地类转换，不创建新的模块对象：

```python
class NpuSharedExperts(FeedForward):
    @classmethod
    def convert(cls, parent: FeedForward) -> "NpuSharedExperts":
        parent.__class__ = cls
        return cast("NpuSharedExperts", parent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return npu_shared_experts_forward(self, x)
```

由于对象本身没有被替换，它会保留原模块的：

- `w1`；
- `w2`；
- `w3`；
- 参数对象；
- 子模块名；
- buffer；
- forward/backward hook；
- training/eval 状态；
- 模块 identity；
- state dict key。

因此从 checkpoint、TP plan 和参数初始化角度看，模块结构保持稳定，改变的只是 forward
实现。

### 5.3 共享专家前向

```python
def npu_shared_experts_forward(self, x):
    packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
    hidden = _swiglu_group_activation(
        packed,
        swiglu_limit=getattr(self, "swiglu_limit", None),
    )
    return self.w2(hidden)
```

参数含义为：

| 参数 | 取值 | 原因 |
| --- | --- | --- |
| `x` | `[w1(x) \| w3(x)]` | 对齐 `silu(w1(x)) * w3(x)` |
| `weight` | `None` | 共享专家没有 routed score |
| `group_index` | `None` | 所有输入行都参与计算，不按路由分组截断 |
| `clamp_limit` | DeepSeek-V4 传 `self.swiglu_limit`；属性不存在时映射为 `-1.0` | 保持模型原有 clamp/无 clamp 语义 |

### 5.4 共享专家实际融合了几个小算子

DeepSeek-V4 共享专家融合前：

```text
w1(x) + w3(x) + clamp(gate) + clamp(up) + silu + mul + w2
```

融合后：

```text
w1(x) + w3(x) + cat + SwigluGroup + w2
```

因此严格来说，它融合的是激活阶段的：

- gate/up clamp（模型配置了 `swiglu_limit` 时）；
- `silu`；
- `silu结果 * w3结果` 的 elementwise `mul`。

同时为了满足算子 `[A | B]` 输入格式，新增一次 `cat`。所以共享专家是否取得净性能收益要看：

```text
减少的 clamp/silu/mul 调度与中间读写
    是否大于
新增 cat 的分配和拷贝成本
```

这也是为什么方案没有预先承诺共享专家一定加速，而是要求通过 950 上的 timeline 区间和
稳态 step time 实测。

## 6. CANN 9.2.0 反向接入

### 6.1 为什么需要本地 `torch.autograd.Function`

前向 `torch.ops.cann_ops_nn.swiglu_group` 只有 NPU 设备 kernel，没有 Autograd kernel。
如果直接从模型 forward 调用它，PyTorch 不会自动推导该自定义算子的梯度，也不会自动去
调用名字相似的 `swiglu_group_backward`。

最终在 `gmm_swiglu.py` 中增加局部 Autograd 桥：

```text
_NpuSwigluGroup.apply
  ├─ forward  → cann_ops_nn::swiglu_group
  └─ backward → cann_ops_nn::swiglu_group_backward
```

没有向全局 dispatcher 再注册 Autograd kernel，原因是：

- 只影响 `npu_gmm_swiglu` 的目标调用点；
- 不污染其他项目直接使用 `cann_ops_nn::swiglu_group` 的行为；
- 以后 CANN 包若自带 Autograd 注册，不会与插件的全局注册冲突。

### 6.2 同时懒加载前向和反向入口

```python
def _load_swiglu_group_ops():
    importlib.import_module("cann_ops_nn.ops")
    forward_op = torch.ops.cann_ops_nn.swiglu_group.default
    backward_op = torch.ops.cann_ops_nn.swiglu_group_backward.default
    return forward_op, backward_op
```

真实实现还会缓存两个 op 对象。任一入口缺失时，在 converter 应用阶段抛出包含算子名的
`RuntimeError`，避免训练运行到第一次 backward 才发现包能力不足。

使用局部导入而不是模块顶层强依赖的原因是：

- 通用 `npu_gmm` 不需要该包；
- 只有显式启用 `npu_gmm_swiglu` 时才检查；
- 能给出比普通 `ImportError` 更明确的环境提示。

这是一种“按 converter 延迟导入、在转换阶段提前校验”的方式：导入 `gmm.py` 不会依赖
`cann_ops_nn`；`npu_gmm_swiglu` 应用时会主动调用 `_load_swiglu_group_ops()`，在成图前完成
Python 模块导入、dispatcher 注册和入口校验。Autograd forward/backward 仍各自调用 loader，
以避免依赖隐含的初始化顺序；正常训练中会直接命中缓存。

### 6.3 Autograd 前向保存什么

```python
def forward(ctx, x, weight, group_index, clamp_limit):
    forward_op, _ = _load_swiglu_group_ops()
    ctx.save_for_backward(x, weight, group_index)
    ctx.clamp_limit = clamp_limit
    return forward_op(
        x,
        weight=weight,
        group_index=group_index,
        clamp_limit=clamp_limit,
    )
```

反向所需信息包括：

- 前向原始输入 `x`；
- 可选 token 权重 `weight`；
- 可选分组信息 `group_index`；
- clamp 阈值。

前向输出没有额外暴露未加权激活 `y_origin`，所以 routed-score 梯度需要在反向阶段处理。

### 6.4 路由专家为什么要重算 `y_origin`

路由专家前向为：

```text
y_origin = swiglu(clamped x)
y        = y_origin * weight
```

对 `weight` 求梯度需要未加权激活：

```text
grad_weight = reduce(grad_y * y_origin)
```

但前向算子只返回 `y`。不能用：

```text
y_origin = y / weight
```

因为 routed score 可能为零或非常接近零，会造成除零、Inf、NaN 或数值放大。最终方案是在
反向中用同一个 `x/group_index/clamp_limit` 重算一次不带 weight 的前向：

```python
y_origin = None
if weight is not None:
    y_origin = forward_op(
        x,
        weight=None,
        group_index=group_index,
        clamp_limit=ctx.clamp_limit,
    )
```

然后调用：

```python
grad_x, grad_weight = backward_op(
    grad_output.contiguous(),
    x,
    weight=weight,
    y_origin=y_origin,
    group_index=group_index,
    clamp_limit=backward_clamp_limit,
)
```

这意味着路由专家 backward 附近通常会看到：

```text
1 × SwigluGroup               （重算 y_origin）
1 × SwigluGroupBackward       （计算 grad_x / grad_weight）
```

重算带来额外计算，但它保证了零 routed score 下的梯度正确性。

### 6.5 共享专家为什么不需要重算

共享专家传入 `weight=None`，反向只需要输入梯度，不需要 routed-score 梯度。因此：

```text
y_origin = None
```

即可直接调用反向算子。共享专家 backward 预期只有一个融合反向 kernel，不包含为了
`grad_weight` 进行的额外 `SwigluGroup` 重算。

### 6.6 `clamp_limit` 前后向 sentinel 不一致

这是 CANN 9.2.0 适配中实际触发训练失败的问题。

前向约定：

```text
clamp_limit = -1.0 → 不启用 clamp
```

反向约定：

```text
clamp_limit >= 0.0
clamp_limit = 0.0  → 不启用 clamp 反向掩码
```

如果无 clamp 路径把前向值原样传到新反向接口，就会报错：

```text
RuntimeError: clamp_limit must be >= 0.0
```

最终只在反向调用边界做精确映射：

```python
backward_clamp_limit = (
    0.0 if ctx.clamp_limit == -1.0 else ctx.clamp_limit
)
```

需要强调：

- 前向继续使用 `-1.0`；
- routed `y_origin` 重算也是前向调用，继续使用 `-1.0`；
- 只有 `swiglu_group_backward` 收到 `0.0`；
- 正数阈值原样传递，例如前向 `0.5`，反向仍为 `0.5`；
- 不使用 `max(ctx.clamp_limit, 0)`，避免把其他非法负数静默改成合法值。

实测 `swiglu_group_backward` wrapper 明确要求该参数非负；当前公开的
`SwigluGroupQuantGrad` 算子说明也给出 `clampLimit >= 0`，且 `0` 表示不启用 clamp
掩码。参见
[ops-nn `SwigluGroupQuantGrad` 文档](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group_quant_grad/README.md)。

### 6.7 梯度返回与二阶梯度边界

Autograd Function 最终按输入顺序返回：

```python
return (
    grad_x if ctx.needs_input_grad[0] else None,
    grad_weight if weight is not None and ctx.needs_input_grad[1] else None,
    None,  # group_index
    None,  # clamp_limit
)
```

其中 `group_index` 和 `clamp_limit` 不可微。反向使用
`@torch.autograd.function.once_differentiable`，明确当前只支持一阶梯度，不承诺 double
backward。

### 6.8 `torch.compile` 的激活桥接与分图处理

当前保留 activation-only compile 设计：只编译两个 GMM 之间的激活桥，不把两次
`torch._grouped_mm` 一起包进编译区域。`_compile_expert_activation()` 接收统一的
`activation_fn`，因此可以分别编译：

```text
npu_gmm：        _expert_activation
npu_gmm_swiglu： _swiglu_group_activation
```

激活桥使用自定义 partitioner：

```python
class _NpuGmmAotDefaultPartitioner(CustomPartitionerFn):
    def __call__(self, gm, joint_inputs, **kwargs):
        return default_partition(gm, joint_inputs, ...)
```

原因是 Inductor 的 min-cut partitioner 可能把 clamp 梯度掩码相关节点提前拉进前向区域，
改变期望的前反向边界。使用 AOTAutograd 默认 partition 能让这些反向相关逻辑留在
backward 图中。

编译入口为：

```python
torch.compile(
    fn,
    backend="inductor_npu",
    fullgraph=True,
    options={"custom_partitioner_fn": _NpuGmmAotDefaultPartitioner()},
)
```

自定义 partitioner 的 `uuid()` 还用于区分编译缓存，避免与其他 partition 方案共用错误的
cache entry。`compile_expert_activation()` 还按稳定的 `activation_key` 对专家分组，并使用：

```python
compile_key = (backend, dynamic_tokens, activation_key)
```

区分 native/fused 图，避免两种激活模式复用错误的编译结果。同一模式只编译一次并共享给
匹配的 `NpuGroupedExperts`。EP 场景每步 routed token 数可能变化，当
`dynamic_tokens=True` 时，激活输入 `h` 和可选 `routed_scores` 的第 0 维会被标记为动态。

## 7. 接入过程中的问题与修正

### 7.1 Profiling 的 Device 与 Ascend Hardware 名称不同

接入路由专家后，曾出现以下现象：

- Profiling 的 framework/device 侧能找到 `_swiglu_group_activation` 或 `swiglu_group`；
- Ascend Hardware 视图中一开始没有找到完全相同的字符串。

这不能直接说明替换失败。Profiling 中存在多个命名层级：

| 层级 | 可能看到的名称 | 表达的含义 |
| --- | --- | --- |
| Python / Framework | `_swiglu_group_activation`、`_NpuSwigluGroup`、`activation_key=swiglu_group` | Python 调用或图节点范围 |
| Torch dispatcher | `cann_ops_nn::swiglu_group`、`cann_ops_nn::swiglu_group_backward` | 对外入图接口 |
| Ascend Hardware | `SwigluGroup`、`SwigluGroupQuantGrad` 或包内 kernel 名 | 实际下发到 AI Core 的 kernel |

对外算子名和底层 kernel 名不必一字不差。更可靠的判断方式是把调用关系、shape、时间位置
和前后相邻算子结合起来看。

### 7.2 Ascend Hardware 找不到算子不是因为共享专家未接入

路由专家和共享专家是两条独立调用路径。只要路由专家已经调用 `SwigluGroup`，即使共享
专家尚未替换，路由调用也应产生相应设备 kernel。

共享专家接入影响的是：

- 调用次数；
- 新增的 shape；
- timeline 中共享 FFN 区间的算子结构。

它不会决定路由专家的 `SwigluGroup` 是否能出现在 Ascend Hardware 中。因此“硬件视图没
找到”应优先检查名称映射、过滤条件、时间窗口和调用链，而不是归因于共享专家未接入。

### 7.3 最可靠的前向替换证据

实际 profiling 后，在两个 GMM 算子之间找到了 `SwigluGroup`，并确认它替代了此前的
clamp/SwiGLU/mul 等算子。这比只搜索一个名称更有说服力：

```text
GMM-1
  └─ SwigluGroup
       └─ GMM-2
```

同时原分解路径的 clamp、`npu_swiglu` 和 routed-score elementwise mul 不再出现在同一
区间，形成“新节点出现 + 旧节点消失 + 拓扑位置正确”的闭环证据。

### 7.4 反向不是简单的接口改名

旧认知中的 Python 入口为：

```text
swiglu_group_grad
swiglu_group_quant_grad
```

CANN 9.2.0 实际入口变为：

```text
swiglu_group_backward
swiglu_group_quant_backward
```

但进一步查询发现前向没有 Autograd 注册，所以不能只把代码中的 `grad` 字符串替换成
`backward`。最终修正包含：

1. 同时加载前向和新反向入口；
2. 增加局部 `torch.autograd.Function`；
3. 保存反向所需输入；
4. routed weight 场景重算 `y_origin`；
5. 返回 `grad_x` 和 `grad_weight`；
6. 显式限定只支持一阶梯度。

### 7.5 不能相信 Python distribution metadata

实测 CANN 目录和接口已经是 9.2.0，但：

```text
cann_ops_nn version: 1.0.0
cann-ops-nn version: 1.0.0
```

如果代码写成：

```python
if metadata.version("cann_ops_nn") >= "9.2.0":
    ...
```

会得到错误判断。最终采用“能力检测”而不是“版本判断”：

```text
能否 import cann_ops_nn.ops
是否存在 torch.ops.cann_ops_nn.swiglu_group.default
是否存在 torch.ops.cann_ops_nn.swiglu_group_backward.default
```

这也是接入外部算子包时更稳健的通用做法。

### 7.6 `clamp_limit=-1.0` 导致首次训练 backward 失败

重构后的 A5 训练在首个 backward 报错：

```text
gmm_swiglu.py -> _NpuSwigluGroup.backward
  -> cann_ops_nn.swiglu_group_backward
  -> RuntimeError: clamp_limit must be >= 0.0
```

根因不是 optimizer、FSDP、HCCL 或共享专家结构，而是前后向接口使用了不同的 no-clamp
sentinel：

```text
forward : -1.0
backward:  0.0
```

当前整合提交 `d290fdf` 已包含该修复；重整前的开发提交 `e60b2de` 在反向边界完成了
精确映射，并增加回归断言：

```python
assert forward_calls == [(None, None, -1.0)]
assert backward_calls == [(None, None, None, 0.0)]
```

另一个带 clamp 的测试继续断言反向收到 `0.5`，防止修复无 clamp 场景时破坏正数阈值。

## 8. 如何判断替换成功

判断融合算子是否真正生效，不能只看训练“能跑”，也不能只搜一个 profiling 字符串。建议
按五个层次验证。

### 8.1 第一层：Converter 和模型结构

确认配置中启用了：

```python
get_model_converter_config("npu_gmm_swiglu")
```

确认设备识别为：

```text
get_npu_device_type() == "A5"
```

模型转换后应满足：

- routed `GroupedExperts` 被替换为 `NpuGroupedExperts`；
- `_expert_activation_key == "swiglu_group"`，且 eager/current activation 均指向融合入口；
- `*.shared_experts` 原地转换为 `NpuSharedExperts`，模块 identity 保持不变；
- dense `feed_forward` 仍是原模块；
- 共享专家 state dict key 仍为 `w1/w2/w3`。

这只能证明替换逻辑被安装，不能单独证明设备 kernel 已执行。

### 8.2 第二层：Framework/dispatcher 节点

Profiling 或图导出中应能看到：

```text
cann_ops_nn::swiglu_group
```

训练反向还应看到：

```text
cann_ops_nn::swiglu_group_backward
```

如果只看到 Python 范围 `_swiglu_group_activation`，仍需继续向下展开，确认范围内部存在真实
`torch.ops` 调用。

### 8.3 第三层：Ascend Hardware timeline

#### 路由专家前向

期望从：

```text
GMM-1
  → clamp / clamp / cat
  → SwiGLU
  → mul routed score
  → GMM-2
```

变为：

```text
GMM-1
  → SwigluGroup
  → GMM-2
```

#### 共享专家前向

期望从：

```text
w1 matmul / w3 matmul
  → clamp gate / clamp up
  → silu
  → mul
  → w2 matmul
```

变为：

```text
w1 matmul / w3 matmul
  → cat
  → SwigluGroup
  → w2 matmul
```

#### 路由专家反向

带 routed score 时，期望看到：

```text
SwigluGroup             # y_origin 重算
SwigluGroupBackward     # 或对应底层 Grad kernel 名
```

#### 共享专家反向

不带 weight，不需要重算，期望只有：

```text
SwigluGroupBackward     # 或对应底层 Grad kernel 名
```

需要考虑 activation checkpointing：它会在 backward 期间重新执行更大范围的 forward，
因此 profiling 中出现额外前向算子不一定是本地 `y_origin` 重算。判断时要结合调用栈、shape
和所在时间区间。

### 8.4 第四层：数值与梯度

算子出现并不等于训练语义正确。至少要对齐：

- routed forward 输出；
- shared forward 输出；
- `x.grad`；
- 路由专家 `w13.grad`、`w2.grad`；
- `routed_scores.grad`；
- 共享专家 `w1/w2/w3` 梯度；
- 所有输出和梯度为 finite；
- 完整训练的 loss、grad_norm 无 NaN/Inf，并与同 checkpoint 基线保持合理一致。

仓库中的单元测试使用 fake dispatcher 验证 Python 接线、参数和梯度返回，不执行真实 CANN
kernel，也不能替代数值对齐。真正的训练对齐仍应固定 checkpoint、数据顺序、随机种子和
并行配置。

### 8.5 第五层：性能

性能验收建议同时看：

1. 融合区间 kernel 数量；
2. 融合区间累计耗时；
3. 稳态单步时间；
4. TPS；
5. MFU；
6. device active time；
7. 是否因 `cat`、重算或同步引入新的瓶颈。

不要用第一次执行耗时判断性能，因为第一次可能包含：

- CANN Python 模块和 dispatcher 首次加载；
- torch.compile 图捕获与编译；
- cache 建立；
- profiler warmup；
- checkpoint 或通信初始化。

应排除 warmup 和 compile step，比较多个稳态 step。

### 8.6 推荐验证命令

CPU/普通开发环境中的 converter、Autograd 和 compile 选择逻辑测试：

```bash
python3 -m pytest \
  tests/unit_tests/converters/test_gmm_swiglu_group.py \
  tests/unit_tests/models/test_deepseek_v4_gmm_compile.py \
  -q
```

当前仓库没有 `SwigluGroup` 专用的 `tests/smoke_tests/features/test_gmm.py` 用例，上述单元
测试也使用 fake CANN op/fake `torch.compile`，不能证明真实 NPU kernel 或
`inductor_npu` 可以运行。A5 实机验证应分别覆盖：

```text
routed eager forward/backward
routed inductor_npu forward/backward
shared eager forward/backward
shared inductor_npu forward/backward
```

最后运行真实 DeepSeek-V4 训练，至少跨过：

```text
forward → loss → backward → optimizer.step
```

只完成 forward 或只进入 profiling step 不能证明训练接入完成。

### 8.7 截至本文的验证状态

- 路由专家前向替换已通过 profiling 确认：`SwigluGroup` 位于两个 GMM 之间，原 clamp、
  SwiGLU 等分解节点被替换。
- 共享专家代码已接入 `npu_gmm_swiglu` converter，并透传模型的 `swiglu_limit`。
- CANN 9.2.0 新反向入口已接入本地 Autograd Function。
- `clamp_limit=-1.0` 直接传反向导致的 A5 错误已修复，并包含在当前整合提交 `d290fdf` 中。
- `4c4b633` 已完成 converter 拆分；在服务器的一次性干净副本中，
  相关的 57 个单元测试全部通过。
- 拆分前 `b67fe80` 实现已在 A5 服务器验证可运行；如需形成可复查证据，还应在本文补充实际
  训练配置、执行命令和日志/Profiling 路径。
- 本地非 NPU 环境已通过语法、ruff、format、codespell 和 diff 检查；由于本地没有
  `torch`，仓库内单元测试未在该本地环境实际执行。

## 9. 最终调用链对照

### 9.1 路由专家：`npu_gmm` 拆分路径

```text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
      └─ _run_experts_grouped_mm
          ├─ offsets = cumsum(num_tokens_per_expert)
          ├─ torch._grouped_mm(x, w13)
          ├─ activation_fn = _expert_activation
          │    ├─ chunk gate/up
          │    ├─ clamp gate/up（可选）
          │    ├─ torch_npu.npu_swiglu
          │    └─ mul routed_scores（可选）
          └─ torch._grouped_mm(hidden, w2)
```

### 9.2 路由专家：`npu_gmm_swiglu` A5 融合路径

```text
NpuGroupedExperts.forward
  └─ npu_grouped_experts_forward
      └─ _run_experts_grouped_mm
          ├─ offsets = cumsum(num_tokens_per_expert)
          ├─ torch._grouped_mm(x, w13)
          ├─ activation_fn = _swiglu_group_activation
          │    └─ _swiglu_group_op
          │         └─ _swiglu_group
          │              └─ _NpuSwigluGroup.apply
          │                   └─ cann_ops_nn::swiglu_group
          │                        ├─ clamp（可选）
          │                        ├─ SwiGLU
          │                        └─ mul routed score（可选）
          └─ torch._grouped_mm(hidden, w2)
```

### 9.3 路由专家反向

```text
_NpuSwigluGroup.backward
  ├─ weight is not None
  │    └─ swiglu_group(weight=None)
  │         └─ 重算 y_origin
  ├─ -1.0 forward sentinel → 0.0 backward sentinel（仅无 clamp）
  └─ swiglu_group_backward
       ├─ grad_x
       └─ grad_weight → routed_scores.grad
```

### 9.4 DeepSeek-V4 共享专家：原路径

```text
DeepSeekV4FeedForward.forward
  ├─ gate = w1(x)
  ├─ up = w3(x)
  ├─ clamp gate/up（swiglu_limit）
  ├─ hidden = silu(gate) * up
  └─ output = w2(hidden)
```

### 9.5 共享专家：A5 融合路径

```text
NpuSharedExperts.forward
  └─ npu_shared_experts_forward
      ├─ gate = w1(x)
      ├─ up = w3(x)
      ├─ packed = cat(gate, up)
      ├─ _swiglu_group_activation(
      │    swiglu_limit=getattr(self, "swiglu_limit", None),
      │  )
      │    └─ _NpuSwigluGroup.apply
      │         └─ cann_ops_nn::swiglu_group(
      │              weight=None,
      │              group_index=None,
      │              clamp_limit=swiglu_limit 或 -1.0,
      │            )
      └─ output = w2(hidden)
```

### 9.6 共享专家反向

```text
_NpuSwigluGroup.backward
  ├─ weight is None
  ├─ y_origin = None，不执行额外前向重算
  ├─ 正数 clamp_limit 原样传递
  ├─ 仅无 clamp 时执行 -1.0 → 0.0
  └─ swiglu_group_backward
       └─ grad_x
```

## 10. 代码改动范围

### 10.1 主要实现文件

`torchtitan_npu/converters/kernels/gmm.py` 保留通用 GMM 能力：

- 两次 grouped MM、TP 处理和 `_expert_activation`；
- `NpuGroupedExperts`、`GMMStateDictUpdater` 与 `npu_gmm` 注册；
- 通用的 activation callable/key 注入和 activation-only compile 框架。

`torchtitan_npu/converters/kernels/gmm_swiglu.py` 承担 A5 专属能力：

- CANN 前向/反向 op 懒加载与 `_NpuSwigluGroup` Autograd Function；
- `_swiglu_group_activation` 和共享专家原地转换；
- A5 校验与 `npu_gmm_swiglu` 注册；
- 复用 `gmm.NpuGroupedExperts` 和 `gmm.GMMStateDictUpdater`。

完整改动还包括单元测试、DeepSeek-V4 compile/AC 检测、DeepSeek-V3.2 TP
限制和融合算子用户文档。

### 10.2 测试文件

`tests/unit_tests/converters/test_gmm_swiglu_group.py` 覆盖：

- 输入格式和 optional 参数；
- routed score 转 FLOAT32/contiguous；
- 前向 no-clamp sentinel；
- Autograd forward/backward 连接；
- `y_origin` 重算；
- routed-score 梯度返回；
- backward no-clamp sentinel 映射；
- 共享专家 `w1/w3` 拼接顺序；
- 共享专家原地转换后保留 identity、参数、buffer、hook、training 状态和 state dict key；
- `npu_gmm_swiglu` 只转换名为 `shared_experts` 的 `FeedForward`，不改变 dense FFN；
- 共享专家 `swiglu_limit` 透传；
- `npu_gmm` 不加载 CANN，使用 native activation；
- 非 A5 选择 `npu_gmm_swiglu` 时在修改模型前报错；
- 两个 GMM converter 不能重复应用。

`tests/unit_tests/models/test_deepseek_v4_gmm_compile.py` 覆盖：

- `activation_fn` 确实在两次 GMM 之间调用；
- native/fused 两种激活模式的选择；
- compile 结果按模式共享并保持幂等；
- compile key 包含 `activation_key`；
- 动态 token 维度标记。

上述测试使用 fake dispatcher 或 fake `torch.compile`，不执行真实 CANN kernel。当前仓库没有
为本功能新增 `tests/smoke_tests/features/test_gmm.py` 用例，A5 运行结果来自仓库外的实机
验证。

### 10.3 关键提交

| Commit | 内容 |
| --- | --- |
| `d290fdf` | 整合 A5 路由/共享专家 `SwigluGroup`、Autograd bridge、单元测试和文档 |
| `8d59053` | 精简融合算子用户文档并移除 installation 中不再需要的说明 |
| `b67fe80` | 共享专家复用 `_swiglu_group_activation` 统一入口 |
| `4c4b633` | 将 A5 融合从 `npu_gmm` 拆到显式的 `npu_gmm_swiglu` converter |

早期开发过程中曾使用 `91b1263`、`9ebddc4`、`e60b2de` 等提交记录前向接入、反向桥接和
sentinel 修复；它们不是当前 `feat/swiglu-group-a5-fusion` 分支 `HEAD` 的祖先，当前 PR
应以上表四个功能提交为准。

## 附录 A：相关概念详解

### A.1 GMM 是什么

GMM（Grouped Matrix Multiplication）把多组 shape 兼容的矩阵乘法放进一次 grouped op。
MoE 中不同专家执行相同结构但使用不同权重，非常适合 GMM：

```text
expert 0: tokens_0 @ weight_0
expert 1: tokens_1 @ weight_1
expert 2: tokens_2 @ weight_2
...
```

通过 `num_tokens_per_expert` 的前缀和生成 offsets，GMM 知道每段 token rows 应该使用哪组
专家权重。

`SwigluGroup` 不是 GMM。它位于两个 GMM 之间，负责逐行激活和可选权重乘法。两者组合
才构成完整的路由专家 FFN：

```text
GMM-1 → SwigluGroup → GMM-2
```

### A.2 `w1`、`w3` 与 `w13`

普通 SwiGLU FFN 使用两个独立升维权重：

```text
w1 → gate
w3 → up
```

路由专家为了让一次 GMM 同时产生 gate/up，通常把两个权重融合为 `w13`：

```text
w13 = concat(w1, w3)
x @ w13 → [gate | up]
```

共享专家本次保留独立 `w1/w3`，在输出侧执行 `cat`。两种布局数学上可以等价，但参数对象、
state dict、TP plan 和其他 converter 的观察结果不同，不能只凭数学等价就直接替换。

### A.3 `weight` 为什么就是 routed score

MoE combine 需要把每个选中专家的输出乘以 router 产生的 score。原路径在 SwiGLU 后执行：

```python
h = h * routed_scores.to(h.dtype)
```

`SwigluGroup` 的可选 `weight` 正好表达这一逐 token 乘法，所以路由专家把
`routed_scores` 作为 `weight` 传入。这里的 weight 不是线性层权重，也不是 `w1/w2/w3`。

### A.4 为什么 `group_index` 可以为空

`group_index` 用于算子内部的 count/group 截断场景。但当前路由流程在进入激活前已经：

1. 根据 router 结果重排 token；
2. 根据 `num_tokens_per_expert` 构造 GMM offsets；
3. 只生成实际参与专家计算的 rows。

因此 `SwigluGroup` 处理所有输入行即可。把 `group_index=None` 不会取消 GMM 的专家分组，
因为 GMM 分组已经在前一个算子完成。

### A.5 `y_origin` 是什么

`y_origin` 是未乘 routed score 的 SwiGLU 激活：

```text
y_origin = silu(A) * B
y = y_origin * weight
```

它主要用于计算 `grad_weight`。名称中的 origin 表示 weight 乘法之前的原始激活，而不是
未 clamp 的输入或整个 FFN 的原始输出。

### A.6 `PrivateUse1` 与 `AutogradPrivateUse1`

PyTorch dispatcher 通过 dispatch key 为同一个算子选择不同实现。对 NPU 自定义设备而言：

- `PrivateUse1=True` 表示存在 NPU 设备实现；
- `AutogradPrivateUse1=True` 表示存在针对该设备的 Autograd 包装；
- `Autograd=False` 也说明没有通用 Autograd 实现；
- Composite key 为 false，说明不能由一组 PyTorch 原生可微算子自动分解出反向。

所以“前向能执行”与“训练能反向”是两项独立能力。

### A.7 Python op 名、IR 名和硬件 kernel 名

同一次计算可能有三种名称：

```text
Python wrapper:  cann_ops_nn.ops.swiglu_group(...)
Dispatcher:      cann_ops_nn::swiglu_group
Hardware kernel: SwigluGroup / 内部实现名
```

反向也可能表现为：

```text
Dispatcher:      cann_ops_nn::swiglu_group_backward
Hardware kernel: SwigluGroupQuantGrad（取决于实际 CANN 包实现）
```

因此 profiling 搜索时应同时尝试大小写、`backward/grad` 和 kernel 实现名，并结合调用位置
判断。

### A.8 为什么前向和反向使用不同 no-clamp 值

sentinel 是接口协议，不是数学运算本身。前向 schema 为了与既有接口保持一致，使用默认
`-1.0` 表示关闭 clamp；新反向接口把属性范围限定为非负，使用 `0.0` 表示不生成 clamp
梯度掩码。

插件位于两个协议之间，必须在边界显式翻译：

```text
Python/forward no clamp: -1.0
             │
             └─ Autograd bridge mapping
                    │
backward no clamp:  0.0
```

这类问题说明即使前后向属性同名，也不能假设取值约定完全相同。

### A.9 为什么不直接对 CANN 前向做全局 Autograd 注册

全局注册的优点是任何调用方都能自动求导，但代价是影响进程中所有
`cann_ops_nn::swiglu_group` 调用，并可能与未来 CANN 自带注册重复。

本次只在 `_swiglu_group()` 局部入口使用 `torch.autograd.Function`：

```text
torchtitan-npu 专家路径 → 有显式 backward
其他直接调用 CANN op  → 保持 CANN 包原行为
```

这是插件仓中更保守的影响范围。

### A.10 共享专家为什么可能提升性能，也可能被 `cat` 抵消

对于 DeepSeek-V4，融合算子减少了 gate/up clamp、`silu` 和 `mul` 的独立下发与中间访存，
但为了构造 `[gate | up]` 新增 `cat`。净收益与以下因素有关：

- batch/token 数；
- hidden/intermediate size；
- `cat` 带宽效率；
- 原 `silu/mul` kernel 效率；
- kernel launch 开销；
- 与前后 matmul 的并行和流水关系。

因此共享专家优化应以真实 shape 的 profiling 为准，而不是以融合算子数量推导结论。

### A.11 为什么 routed backward 有额外前向而 shared backward 没有

区别来自是否传入 `weight`。当前实现只要 `weight is not None` 就重算 `y_origin`，随后再根据
`ctx.needs_input_grad[1]` 决定是否向 Autograd 返回 `grad_weight`：

```text
routed experts:
  weight = routed_scores
  → 需要 y_origin
  → 额外重算一次无 weight 前向

shared experts:
  weight = None
  不存在 grad_weight
  → 不需要 y_origin
  → 不重算
```

这也是 profiling 区分两条反向路径的一个重要特征。

### A.12 activation checkpointing 对 Profiling 的影响

activation checkpointing 为节省显存，会在 backward 中重新执行一段 forward。因此某个
时间区间再次出现 `GMM` 或 `SwigluGroup`，可能来自：

1. checkpoint 的整段 forward recompute；
2. `_NpuSwigluGroup.backward` 为 routed score 梯度执行的局部 `y_origin` recompute。

区分方法包括：

- 看是否连同整个 block 一起重算；
- 看调用栈；
- 看输入 shape；
- 看重算后是否紧邻 `swiglu_group_backward`；
- 在小型无 checkpoint 对照实验中建立基线。

### A.13 “训练能跑”为什么仍不等于接入正确

训练不报错只能证明当前输入没有触发显式异常，还不能证明：

- 融合路径真的执行；
- gate/up 顺序正确；
- clamp 语义正确；
- routed score 梯度正确；
- 共享专家参数仍能得到梯度；
- 通用 `npu_gmm` 路径没有被 A5-only 依赖改变；
- checkpoint 可兼容加载；
- 性能获得提升。

完整证据链应包含：代码选择、图节点、硬件 kernel、数值/梯度和端到端性能五个层次。

## 参考资料

- [ops-nn：`SwigluGroup`](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group/README.md)
- [ops-nn：`SwigluGroupQuantGrad`](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group_quant_grad/README.md)
- [ops-nn 算子列表](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/op_list.md)
- [华为：Ascend 950PR 与 Ascend 950DT 定位说明](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech)
- [PyTorch `torch.autograd.Function`](https://docs.pytorch.org/docs/stable/autograd.html#function)
- `torchtitan-npu` 当前 `feat/swiglu-group-a5-fusion` 分支：`d290fdf`、`8d59053`、`b67fe80`、`4c4b633`

> 说明：上面的 `ops-nn` 链接指向可变的 `master`。若本文用于长期归档，应替换为与目标
> CANN 9.2.0 环境对应的 tag 或 commit 链接。
