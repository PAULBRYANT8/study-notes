# SwigluGroup 融合算子接入介绍

> 本文用于向熟悉 PyTorch 和模型训练、但不了解本次改动的研发同事介绍
> `SwigluGroup` 的作用及其在 `torchtitan-npu` 中的接入方式。
>
> 对应代码分支：`feat/swiglu-group-a5-fusion`  
> 对应 PR：[cann/torchtitan-npu#468](https://gitcode.com/cann/torchtitan-npu/pull/468)

## 一句话概括

本次接入面向 Ascend 950（A5），使用 CANN `SwigluGroup` 将 MoE 专家 FFN 中的
clamp、SwiGLU 和可选的 routed-score 缩放融合成一次算子调用，并通过独立的
`npu_gmm_swiglu` converter 显式启用，从而降低两个 GMM 之间的 kernel 调度和中间
Tensor 读写开销。

## 1. SwiGLU 和 SwigluGroup 分别做什么

### 1.1 SwiGLU

SwiGLU 是 Transformer FFN 中常见的门控激活。输入分别经过 gate 和 up 两条升维线性层：

```text
gate = w1(x)
up   = w3(x)
hidden = silu(gate) * up
output = w2(hidden)
```

路由专家通常把 `w1` 和 `w3` 合并为一份 `w13` 权重。第一次 GMM 的输出最后一维排列为
`[gate | up]`，SwiGLU 将其均分后计算 `silu(gate) * up`。

### 1.2 SwigluGroup

CANN `SwigluGroup` 在普通 SwiGLU 的基础上，还能融合两类操作：

- 在激活前对 gate 和 up 执行 `swiglu_limit` clamp；
- 在激活后乘每个 routed token 对应的 routed score。

其计算语义可以简化为：

```text
input = [gate | up]

gate = clamp_max(gate, limit)
up   = clamp(up, -limit, limit)
hidden = silu(gate) * up

if weight is not None:
    hidden = hidden * weight
```

`SwigluGroup` 不融合前后的矩阵乘法。本次优化的范围，是把两个 GMM 之间的多个小算子合并：

```text
融合前：GMM-1 -> clamp -> SwiGLU -> routed-score multiply -> GMM-2

融合后：GMM-1 -> SwigluGroup                         -> GMM-2
```

预期收益主要来自减少 kernel launch、减少中间 Tensor 写回和再次读取。实际端到端收益仍会
受到 GMM、通信、token shape 和并行策略等因素影响，因此不能仅根据算子数量承诺固定加速比。

## 2. 为什么使用独立的 converter

早期实现把 A5 融合逻辑直接放在通用 `npu_gmm` 中，并根据设备类型自动选择实现。这样会让
A5 专属的 `cann_ops_nn` 依赖进入默认 GMM 路径，也会扩大 A3 等已验证路径的影响范围。

最终将两种能力拆开：

| Converter | 支持范围 | 两次 GMM 之间的激活 | CANN SwigluGroup 依赖 |
| --- | --- | --- | --- |
| `npu_gmm` | A3、A5 等现有平台 | clamp + `npu_swiglu` + score multiply | 无 |
| `npu_gmm_swiglu` | 仅 A5 | `SwigluGroup` 融合实现 | 有 |

`npu_gmm_swiglu` 是 `npu_gmm` 的替代项，不是需要叠加的第二个 converter。使用融合算子时，
应把配置中的 `npu_gmm` 替换为 `npu_gmm_swiglu`，不能同时配置两者。

这种拆分有三个目的：

1. 保护默认 `npu_gmm` 在 A3 等平台上的既有行为；
2. 让 A5 专属依赖和功能边界在配置中清晰可见；
3. 配置错误时直接失败，避免同一个 converter 内静默回退导致用户误以为融合已经生效。

## 3. 整体接入架构

`npu_gmm_swiglu` 仍然遵循仓库的 ModelConverter 注册模式，没有在模型 forward 中硬编码
设备判断或 CANN 调用。

```text
模型配置选择 npu_gmm_swiglu
                │
                v
      NpuGmmSwigluConverter
                │
                ├─ 校验当前设备为 A5
                ├─ 校验模型尚未被另一个 GMM converter 转换
                ├─ 加载并校验 SwigluGroup forward/backward dispatcher
                │
                ├─ routed GroupedExperts
                │     └─ NpuGroupedExperts
                │          ├─ GMM-1
                │          ├─ _swiglu_group_activation
                │          └─ GMM-2
                │
                └─ shared_experts FeedForward
                      └─ NpuSharedExperts
                           ├─ 原 w1 / w3
                           ├─ SwigluGroup
                           └─ 原 w2
```

实现没有复制一套完整的 GMM 模块。通用 `gmm.py` 允许注入不同的 `activation_fn` 和
`activation_key`：

- `npu_gmm` 注入原生 `_expert_activation`；
- `npu_gmm_swiglu` 注入 `_swiglu_group_activation`；
- 两者复用 `NpuGroupedExperts`、TP/GMM 逻辑及 `GMMStateDictUpdater`。

这样把平台专属融合能力放在 `gmm_swiglu.py`，同时保留了一条统一的专家 GMM 主干。

## 4. 路由专家如何接入

路由专家的核心执行路径为：

```text
重排后的 routed token
        │
        v
GMM-1：x @ w13
        │  输出布局为 [gate | up]
        v
SwigluGroup
        │  clamp + SwiGLU + routed-score scaling
        v
GMM-2：hidden @ w2
```

融合入口主要处理三个参数：

### 4.1 routed score

算子的 `weight` 要求使用 FP32，因此 routed score 会先转换成连续的 FP32 Tensor：

```python
weight = routed_scores.to(dtype=torch.float32).contiguous()
```

共享专家没有 routed score，对应的 `weight` 为 `None`。

### 4.2 group_index

当前传入 `group_index=None`。原因是进入第一次 GMM 前，token 已经完成专家路由和重排，
GMM offsets 也已经由每个专家的 token 数计算完成。中间激活只需要逐行处理这些有效 token，
不需要再次按 `group_index` 分组。

### 4.3 swiglu_limit

模型配置中的 `swiglu_limit` 会透传给融合算子，以保持原有 gate/up clamp 语义。没有 clamp
配置时，前向使用 CANN 约定的 `-1.0` 哨兵值表示不执行 clamp。

## 5. 共享专家如何接入

共享专家是普通 `FeedForward`，没有 GMM 和 routed score，但其中的 clamp 与 SwiGLU 同样
可以融合：

```text
w1(x) ─┐
       ├─ cat([gate, up]) -> SwigluGroup -> w2
w3(x) ─┘
```

这里有三个需要保护的兼容性点：

1. 拼接顺序固定为 `w1(x)` 在前、`w3(x)` 在后，对应 `[gate | up]`；
2. 只转换模块名最后一段为 `shared_experts` 的 `FeedForward`，避免误改普通 dense FFN；
3. 通过原地改变模块类型完成转换，保留模块 identity、参数、buffer、hook、training 状态，
   以及 `w1/w2/w3` 的 state-dict key。

因此接入不会把共享专家的 `w1/w3` 合并成新参数，也不会改变已有 checkpoint 的键名映射。

## 6. 训练反向如何接入

CANN 提供了独立的：

```text
swiglu_group
swiglu_group_backward
```

仅调用前向 dispatcher 并不能保证 PyTorch 自动找到对应反向实现，因此在插件侧增加了局部
`torch.autograd.Function`：

```text
_NpuSwigluGroup.forward
    └─ cann_ops_nn.swiglu_group

_NpuSwigluGroup.backward
    └─ cann_ops_nn.swiglu_group_backward
```

反向接入还处理了两个容易忽略的数值问题。

### 6.1 routed-score 梯度

计算 routed-score 梯度需要未乘权的 SwiGLU 输出 `y_origin`。不能用加权结果除以 score 来
恢复，因为 score 可能为零。当前做法是在需要 weight 梯度时额外重算一次不带 weight 的
SwigluGroup 前向，再把 `y_origin` 交给 backward 算子。

### 6.2 no-clamp 哨兵值

CANN 9.2 的前向和反向接口对“无 clamp”使用不同哨兵值：

```text
forward： -1.0
backward： 0.0
```

Autograd bridge 会在反向前完成转换；正数 `swiglu_limit` 则保持原值传递。

当前桥接使用 `once_differentiable`，支持正常训练所需的一阶反向，不承诺二阶梯度。

## 7. 为什么采用按 converter 延迟加载

`cann_ops_nn.ops` 不在模块顶层导入，而是在应用 `npu_gmm_swiglu` 时加载并缓存 forward、
backward dispatcher。

这与普通顶层导入的区别是：

- 仅使用 `npu_gmm` 的用户不需要安装或加载 A5 专属算子包；
- 非融合路径导入 `gmm.py` 时不会被 `cann_ops_nn` 环境问题阻断；
- 选择融合 converter 后，会在模型转换阶段提前检查 forward/backward 是否完整注册；
- 依赖缺失时给出包含算子名称的明确错误，而不是训练到第一次 forward/backward 才失败。

因此它不是把依赖问题推迟到训练阶段，而是“按功能加载，在成图前校验”。

## 8. torch.compile 与 activation checkpointing

现有 GMM 路径只编译两次 GMM 之间的 activation bridge，不把 GMM 本身一起放入编译区域：

```text
GMM-1 -> [compiled activation bridge] -> GMM-2
```

拆分 converter 后，这套机制仍然复用：

- `npu_gmm` 编译 `_expert_activation`；
- `npu_gmm_swiglu` 编译 `_swiglu_group_activation`；
- 编译缓存 key 包含 `(backend, dynamic_tokens, activation_key)`，防止 native 和 fused 图
  复用错误缓存；
- EP 场景的 routed token 数可能逐步变化，activation 输入的 token 维会按需标记为动态；
- selective activation checkpointing 继续保存 GMM 输出，避免 backward 重算大开销 GMM。

## 9. 如何启用

### 9.1 A5 使用融合实现

把原来的：

```python
get_model_converter_config("npu_gmm")
```

替换为：

```python
get_model_converter_config("npu_gmm_swiglu")
```

不要同时配置两个 GMM converter。`npu_gmm_swiglu` 会同时完成 GMM 转换和 SwigluGroup
激活注入。

### 9.2 A3 等其他平台

继续使用：

```python
get_model_converter_config("npu_gmm")
```

如果非 A5 设备显式选择 `npu_gmm_swiglu`，converter 会在修改模型前抛出 `ValueError`，
不会静默回退。

## 10. 兼容性与验证结论

本次接入重点保护了以下兼容性：

- 默认 `npu_gmm` 的 A3/A5 原生激活路径不变；
- 路由专家继续复用原 GMM state-dict updater；
- 共享专家保留模块对象及 `w1/w2/w3` checkpoint key；
- native/fused activation 使用独立编译缓存；
- converter 在模型修改前检查硬件和重复转换，避免留下半转换状态。

相关单元测试覆盖前向参数、Autograd backward、零 routed score、no-clamp sentinel、共享专家
原地转换、converter 硬件边界、互斥保护及 compile cache。代码曾在服务器的一次性干净副本
中执行相关测试，结果为 `57 passed`。拆分前的融合计算路径已在 A5 实机验证，本次 converter
拆分保持了同一前向、反向计算主体；拆分后的硬件选择和注入逻辑由单元测试覆盖。

本次没有新增 SwigluGroup 专属 smoke 测试。单元测试中的 dispatcher 和 `torch.compile` 使用
fake 实现，因此测试结论与 A5 实机验证应结合看待。

## 11. 可以直接这样向别人介绍

> 我们在 Ascend 950 上为 MoE 专家 FFN 接入了 CANN SwigluGroup。原来两次 GMM 之间要依次
> 执行 clamp、SwiGLU 和 routed-score 乘法，现在把它们融合成一次算子调用，从而减少中间
> Tensor 和 kernel launch。
>
> 接入没有写进模型 forward，而是通过新的 `npu_gmm_swiglu` ModelConverter 完成。这个
> converter 复用原来的 GMM、TP、state-dict 和 activation-only compile 框架，只注入新的
> 激活函数；路由专家和共享专家都能使用融合算子。
>
> 训练侧增加了局部 Autograd bridge，前向调用 `swiglu_group`，反向调用
> `swiglu_group_backward`，并处理了零 routed score 和前后向 no-clamp 哨兵值差异。
> 因为算子目前只支持 A5，所以默认 `npu_gmm` 保持不变，只有显式把它替换为
> `npu_gmm_swiglu` 才会启用融合，其他平台不会受到影响。

## 12. 常见问题

### 为什么不直接继续使用 `npu_swiglu`？

`npu_swiglu` 只完成激活，clamp 和 routed-score 乘法仍是独立操作。`SwigluGroup` 可以把
这些操作一并融合。

### 为什么新 converter 不能和 `npu_gmm` 一起配置？

`npu_gmm_swiglu` 自身已经完成 GMM 转换。两者叠加会重复转换相同的 `GroupedExperts`，
因此转换前会显式检查并报错。

### 算子名带 Group，为什么 `group_index` 传 `None`？

因为 routed token 在进入 GMM 前已经完成重排和专家分组，中间激活不需要再次分组。
`group_index` 是算子的可选能力，不是当前调用必须提供的参数。

### 为什么共享专家也能使用它？

共享专家虽然不参与路由，但同样执行 clamp 和 SwiGLU。它只是不传 routed score，线性层仍
保持普通 `w1/w3/w2`。

### 为什么不让 `npu_gmm` 在 A5 自动融合、其他平台自动回退？

显式 converter 能清楚表达能力和依赖边界，保护默认路径，也能避免配置融合后实际静默回退
却不易发现的问题。

### 主要代码在哪里？

- 通用 GMM 框架：`torchtitan_npu/converters/kernels/gmm.py`
- A5 SwigluGroup 接入：`torchtitan_npu/converters/kernels/gmm_swiglu.py`
- 单元测试：`tests/unit_tests/converters/test_gmm_swiglu_group.py`
- 融合算子说明：`docs/feature_guides/npu_fused_ops.md`

## 参考资料

- [CANN SwigluGroup 算子说明](https://gitcode.com/cann/ops-nn/blob/master/activation/swiglu_group/README.md)
- [torchtitan-npu PR 468](https://gitcode.com/cann/torchtitan-npu/pull/468)
- `DeepSeek-V4 接入 SwigluGroup 开发复盘`：
  `/home/zhangwei/文档/my-wiki/work/swiglu-group-接入复盘.md`
