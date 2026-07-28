# `inplace_partial_rotary_mul` 接入 torchtitan-npu 开发复盘

> 本文基于 `torchtitan-npu` 的 `inplace-rope-latest` 分支整理，对应 PR：
> [cann/torchtitan-npu!444](https://gitcode.com/cann/torchtitan-npu/merge_requests/444)。当前实现基线为
> `ef6102f`。DeepSeek-V4 是首个接入模型，但 converter 和算子本身不限定模型。
>
> 最后更新：2026-07-28。

## 1. 背景与目标

### 1.1 DeepSeek-V4 为什么需要 Partial RoPE

RoPE（Rotary Position Embedding）通过对 Query、Key 的特征维度做二维旋转，把位置信息编码进 Attention。若把相邻两个元素记为 `(a, b)`，旋转可简化表示为：

```text
a' = a * cos - b * sin
b' = a * sin + b * cos
```

常规 RoPE 会旋转整个 head dimension。DeepSeek-V4 的一些 Attention、Indexer 和 Compressor 路径只旋转最后一部分维度，因此代码以 `partial_slice=[start, end]` 指定需要旋转的区间：

```text
x = [未旋转区域 | RoPE 区域 | 未旋转区域]
                  ^ start:end
```

算子必须满足两个基本语义：

1. 只更新 `x[..., start:end]`。
2. `start:end` 之外的数据保持不变。

### 1.2 原始执行路径

DeepSeek-V4 原始 partial RoPE 根据调用场景有 `split`、切片以及可选 `clone` 等分支，但核心流程相同：

```text
x
 ├─ 可选 clone
 ├─ split / slice，取出 x[..., start:end]
 ├─ apply_rotary_emb
 │    └─ npu_rotary_mul，生成旋转后的新 Tensor
 └─ torch.cat，将未旋转区域和旋转结果重新拼接
      └─ 返回新的完整 Tensor
```

这里需要区分两个概念：

- `torch.split`、普通切片通常返回 view，本身不一定复制底层数据。
- `npu_rotary_mul` 的输出以及最后的 `torch.cat` 会产生新的 Tensor；`cat` 还要把未旋转区域和旋转区域重新写入完整输出。

因此，主要开销来自：

- partial RoPE 旋转结果的中间 Tensor；
- `torch.cat` 产生的完整输出 Tensor；
- 拼接过程对未旋转区域的额外显存读写；
- 多个 eager 算子带来的调度开销；
- 部分调用点为保持原语义执行的额外 `clone`。

### 1.3 接入目标

`cann_ops_transformer.ops.inplace_partial_rotary_mul` 能直接在输入 Tensor 的指定区间内完成旋转：

```text
x
 ├─ 准备 cos / sin 和 partial_slice
 └─ inplace_partial_rotary_mul
      ├─ 原地更新 x[..., start:end]
      ├─ 保持其他区域不变
      └─ 返回时继续使用原 x
```

本次接入的目标是：

1. 用一个融合算子替代 partial RoPE 的旋转结果生成和重新拼接过程。
2. 减少临时 Tensor、`cat` 和额外显存读写。
3. 不改变已经验证的 `npu_rope` 路径。
4. 通过独立 converter 按需启用，不把 A5 专属能力变成所有平台的默认行为。

## 2. 方案设计与取舍

### 2.1 方案对比

| 方案 | 做法 | 优点 | 问题 | 结论 |
| --- | --- | --- | --- | --- |
| 直接修改 `npu_rope` | 在原 converter 中把 partial RoPE 改成原地算子 | 配置项少 | 会改变现有路径；非 A5 平台也会进入新逻辑；难以保证旧路径数值和 autograd 图不变 | 不采用 |
| 在模型里硬编码平台判断 | DeepSeek-V4 `model.py` 根据设备选择算子 | 实现直接 | 模型代码耦合 NPU/CANN 和平台逻辑，绕过 converter 注册机制 | 不采用 |
| 新增独立 converter | 保留 `npu_rope`，新增 `npu_rope_inplace_partial` | 两条路径隔离；可以单独校验平台和配置；回退关系清晰 | 多一个显式配置项 | 采用 |

### 2.2 最终结构

模型调用点只依赖一个可替换入口 `apply_partial_rotary_emb_`。这个入口默认指向
fallback，只有匹配到该绑定并启用 inplace converter 时才会被替换：

```text
DeepSeek-V4 call sites
        │
        ▼
apply_partial_rotary_emb_
        │
        ├─ 未启用 npu_rope_inplace_partial
        │    └─ apply_partial_rotary_emb_fallback
        │         └─ slice → apply_rotary_emb → cat
        │              └─ 启用 npu_rope 时使用 npu_rotary_mul
        │
        └─ 启用 npu_rope_inplace_partial 且匹配到绑定
             └─ npu_apply_rotary_emb_partial_complex_
                  └─ CANN inplace_partial_rotary_mul
```

两条 converter 路径彼此独立：

| Converter | 文件 | 作用 |
| --- | --- | --- |
| `NpuRoPEConverter` | `rope.py` | 替换普通 RoPE 函数，不修改 partial RoPE 的入口绑定 |
| `NpuInplacePartialRoPEConverter` | `inplace_partial_rope.py` | 仅在模型所属模块存在 `apply_partial_rotary_emb_` 时，把该绑定替换为 CANN 原地实现 |

`NpuInplacePartialRoPEConverter` 不继承 `NpuRoPEConverter`，也不会遍历 `_ROPE_REPLACEMENTS`。这样不会为了接入新算子而重构或复用原 `npu_rotary_mul` 的内部实现。
两者仅共享 `_select_precomputed_rope_cache()` 这一段 cache 选择逻辑，避免复制相同的
`positions` 索引和形状断言；算子替换范围、执行入口和依赖检查仍彼此独立。

`apply_partial_rotary_emb_` 在模型模块加载时就绑定为
`apply_partial_rotary_emb_fallback`。这里的 fallback 是原有的
`split/slice → RoPE → cat` 实现，不是 converter 后来新增的一条逻辑。
最终解耦后，`npu_rope` 不再负责“恢复”这个绑定，只执行原有的普通 RoPE
函数替换；只有显式启用 `npu_rope_inplace_partial` 时，才会把该入口切换为
CANN 原地实现。两个 converter 修改的是不同绑定，可以在同一个配置中同时启用。

当前 DeepSeek-V4 已提供这个入口，因此会匹配；其他模型如果没有同名绑定，converter
直接返回，不做任何替换，也不会因为平台或可选依赖不满足而报错。这里没有
DeepSeek-V4 模块名前缀拦截，算子的适用范围由绑定契约决定。其他模型若提供同名绑定，
还必须保证第二个参数是预计算 `float32` interleave cos/sin cache。

### 2.3 配置约束

最终方案包含以下显式约束：

- 对匹配到 `apply_partial_rotary_emb_` 的模型，`npu_rope_inplace_partial` 当前只允许在 A5（Ascend 950）平台启用。
- 未匹配到该绑定的模型直接跳过，不执行平台和依赖检查。
- 匹配到绑定后，只接受 `[2, cache_seq_len, rotary_dim]` 的预计算 `float32` cache；complex cache 直接 fail-fast。
- `npu_rope` 与 `npu_rope_inplace_partial` 可以同时配置，二者分别处理普通 RoPE 和 partial RoPE。
- DeepSeek-V4 默认 converter 列表继续使用 `npu_rope`。
- 在 Ascend 950 上使用新算子时，在现有 converter 列表中**追加** `npu_rope_inplace_partial`。
- 运行环境需要安装与 CANN 配套、包含 `inplace_partial_rotary_mul` 及其反向实现的 `ops-transformer`。
- 当前仓库 `master` 配套表使用
  [`9.2.0_daily0725`](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260725000025310/)
  CANN daily 包；具体版本始终以 `docs/user-guides/installation.md` 的最新配套表为准。

## 3. 具体接入实现

### 3.1 在模型侧保留稳定的替换入口

文件：`torchtitan_npu/models/deepseek_v4/model.py`

模型侧把原有实现命名为 `apply_partial_rotary_emb_fallback`，再让实际调用点通过变量 `apply_partial_rotary_emb_` 间接调用：

```python
def apply_partial_rotary_emb_fallback(
    x,
    freqs_cis,
    partial_slice,
    inverse=False,
    positions=None,
):
    ...


apply_partial_rotary_emb_ = apply_partial_rotary_emb_fallback
```

Attention、Indexer 和 Compressor 中的调用点不需要知道当前使用哪个算子，只调用 `apply_partial_rotary_emb_`。converter 负责在模型转换阶段替换这个变量指向的实现。

最终实现中的 fallback 校验 `partial_slice` 后，对 `x[..., start:end]` 调用
`apply_rotary_emb`，再通过 `torch.cat` 组装完整输出。TND 输入的形状适配由
`apply_rotary_emb` 及其共享 helper 处理；fallback 和 inplace 入口都只保留实际
调用点需要的五个参数。

### 3.2 按需导入 CANN 算子

文件：`torchtitan_npu/converters/kernels/inplace_partial_rope.py`

```python
def npu_apply_rotary_emb_partial_complex_(...):
    ...
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    cos = cos.to(device=x_for_op.device, dtype=torch.float32).contiguous()
    sin = sin.to(device=x_for_op.device, dtype=torch.float32).contiguous()

    inplace_partial_rotary_mul(
        x_for_op,
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[start, end],
    )
    return x
```

这里没有在模块顶层直接导入 `ops-transformer`，原因是：

1. 默认 `npu_rope` 路径不需要依赖该算子。
2. `NpuInplacePartialRoPEConverter.convert()` 会先导入检查，缺少算子时在转换阶段快速失败。
3. 真正执行融合路径时，融合入口再进行局部导入；最终实现没有使用 `@cache` 或可变的模块级函数变量。

`_apply_inplace_partial_rotary_interleave_()` 只有一个调用点，当前实现已将它直接内联到
融合入口中，避免为了单次调用保留一层薄封装。

### 3.3 融合入口的数据处理

融合入口为：

```python
npu_apply_rotary_emb_partial_complex_(
    x,
    freqs_cis,
    partial_slice,
    inverse=False,
    positions=None,
)
```

函数名中的 `complex_` 是为减少无关调用点改动而保留的既有内部命名，不代表当前入口仍
接受 complex cache；实际数据契约以预计算 `float32` cache 为准。

它按以下顺序处理输入。

#### 第一步：校验旋转区间

`partial_slice` 必须恰好包含两个整数，并满足：

```text
0 <= start <= end <= x.shape[-1]
```

融合入口只接受预计算实数 cache，要求形状为
`[2, cache_seq_len, end - start]`，且 dtype 为 `float32`。该格式是当前支持路径的内部
不变量，由 `_select_precomputed_rope_cache()` 的断言 fail-fast；complex cache 不再支持。

#### 第二步：取得本地 Tensor

如果 `x` 或 `freqs_cis` 是 DTensor，当前实现先通过 `to_local()` 取得本 rank 的 local Tensor：

```python
x_local = x.to_local() if isinstance(x, DTensor) else x
rope_cache_local = freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
```

参数名仍沿用模型调用契约中的 `freqs_cis`，但 inplace 实现把取得的本地 Tensor 命名为
`rope_cache_local`，明确它必须是预计算 cache。inverse RoPE 保持 cos 不变并对 sin 取负。

#### 第三步：检查 autograd leaf

PyTorch 不允许直接原地修改一个需要梯度的 leaf Tensor。因此入口会显式拒绝这种输入：

```python
if x_local.requires_grad and x_local.is_leaf:
    raise RuntimeError(...)
```

DeepSeek-V4 的正常调用点传入的是线性层、归一化等操作产生的中间结果，通常属于 non-leaf Tensor。

#### 第四步：整理输入形状

`positions` 先经过上游 `_maybe_wrap_positions()` 处理；如果结果是 DTensor，也会转为
local Tensor。随后 CANN 算子按四维形式处理输入，
`_reshape_partial_rotary_x_for_op()` 只通过 `unsqueeze` 增加维度，不改变元素顺序：

| 原始语义 | 输入形状示意 | 送入算子的形状示意 |
| --- | --- | --- |
| BSND | `[B, S, N, D]` | `[B, S, N, D]` |
| 三维普通输入 | `[B, S, D]` | `[B, S, 1, D]` |
| TND Attention | `[T, N, D]` | `[1, T, N, D]` |
| TND Compressor | `[T, D]` | `[1, T, 1, D]` |

是否按 TND 解释，取决于 `positions` 的元素数量是否与 `x.size(0)` 匹配。

#### 第五步：选择当前位置的预计算 cache

`_select_precomputed_rope_cache()` 接收
`[2, cache_seq_len, rotary_dim]` 的 `float32` Tensor，第 0、1 维切片分别为 interleave
格式的 cos、sin。`positions is None` 时直接取前 `seqlen` 个位置；传入 `positions` 时按
位置索引 cache；二维 `positions` 使用第一行，符合当前融合 RoPE 对 batch 间共享位置的
约束。

#### 第六步：复用 interleave 格式的 cos 和 sin

DeepSeek-V4 的 `update_from_config()` 在配置了 `npu_rope` 或
`npu_rope_inplace_partial` 任一 converter 时都会设置 `use_npu_rope=True`。模型在
`precompute_rope_cache()` 中一次性完成 complex 到 interleave cos/sin cache 的转换：

```python
cache = torch.view_as_real(freqs_cis).movedim(-1, 0).repeat_interleave(2, dim=-1)
```

Compressor/MTP 存在多种压缩比例时，`precompute_rope_cache()` 还会把各比例的下采样
cache 分段拼在基础 cache 后面，并由 `_rope_cache_compression_offsets()` 记录偏移；
Compressor 在进入 partial RoPE 前选出对应分段。传入显式 `positions` 时则使用基础
位置段并按 `positions` 选择。

DeepSeek-V4 融合路径直接对选中的 cache 执行 `unbind(0)` 得到 cos、sin，再补充
batch 和 head 广播维。这样每次 partial RoPE 调用不再重复执行 real/imag 和
`repeat_interleave(2)`；传入 `positions` 时仍需对预计算 cache 做一次位置索引。

随后融合入口直接把 cos、sin 移到与 `x_for_op` 相同的设备，转换为算子要求的
`float32` 和连续内存格式：

```text
cache: [2, S, D] → unbind → [S, D] → [1, S, 1, D] float32 contiguous
```

#### 第七步：调用 CANN 原地算子

```python
inplace_partial_rotary_mul(
    x_for_op,
    cos,
    sin,
    rotary_mode="interleave",
    partial_slice=[start, end],
)
```

算子直接更新 `x_for_op[..., start:end]`。`x_for_op` 要么就是 `x_local`，要么是由 `x_local` 通过 `unsqueeze` 得到的 view；两者共享底层数据，所以更新会反映到原输入。

函数最终返回原来的 `x`，而不是创建一个完整的新输出 Tensor：

```python
return x
```

### 3.4 converter 的选择与校验

#### 原 `npu_rope` 路径

模型模块加载时，`apply_partial_rotary_emb_` 已经指向
`apply_partial_rotary_emb_fallback`。`NpuRoPEConverter.convert()` 不再修改这个
partial RoPE 入口，只执行原有 `_ROPE_REPLACEMENTS`：

```python
def convert(self, model):
    for func_name, impl in _ROPE_REPLACEMENTS.items():
        self._replace_one(func_name, impl, model)
```

因此选择 `npu_rope` 时，普通 RoPE 的 `npu_rotary_mul` 函数实现、dtype 转换、
DTensor 包装以及 DeepSeek-V4 的 fallback `slice → RoPE → cat` 路径都保持
原样。也就是说，`npu_rope` 走的是原有普通 RoPE 替换逻辑；这里的 fallback
只是模型原有 partial RoPE 实现的明确命名。

`_replace_one()` 现在是 `NpuRoPEConverter` 的类内 helper。它负责处理上游公共 RoPE
函数以及 `from ... import ...` 形成的模型内局部绑定，不再作为 partial converter 的
共享替换接口。

#### 新 inplace 路径

`NpuInplacePartialRoPEConverter` 是独立 converter：

```python
class NpuInplacePartialRoPEConverter(ModelCustomConverter):
    def convert(self, model):
        binding_name = "apply_partial_rotary_emb_"
        model_module = sys.modules.get(model.__class__.__module__)
        if model_module is None or not hasattr(model_module, binding_name):
            return

        _validate_inplace_partial_rope_platform()
        try:
            from cann_ops_transformer.ops import inplace_partial_rotary_mul
        except ImportError as exc:
            raise RuntimeError(...) from exc
        if getattr(model_module, binding_name) is npu_apply_rotary_emb_partial_complex_:
            return
        setattr(model_module, binding_name, npu_apply_rotary_emb_partial_complex_)
        logger.info(...)
```

转换顺序很重要：

1. 先取得模型类所属模块，并检查模块是否提供 `apply_partial_rotary_emb_`。
2. 没有目标绑定时直接返回，不做任何替换，也不执行平台和依赖检查。
3. 匹配到目标绑定后，要求 `get_npu_device_type()` 返回 `A5`。
4. 再检查当前环境能否导入提供 `inplace_partial_rotary_mul` 的兼容 `ops-transformer` 包。
5. 最后在 `convert()` 内对这个模型模块执行精确 `setattr`，并保证重复执行 converter 时幂等。

`_enable_inplace_partial_rope()` 同样只有一个调用点，当前实现已将其内联到 converter。
平台校验仍保留为独立函数，因为它表达的是明确、可单独理解的硬件约束，而不是简单透传。

当前实现不检查 `torchtitan_npu.models.deepseek_v4` 模块名前缀。DeepSeek-V4 是首个
提供该绑定的模型；未来其他模型只要采用相同五参数调用契约，也可以由同一 converter
接入。与普通 RoPE 的 `_replace_one()` 不同，这里只有一个明确的模块级扩展点，精确
替换模型所属模块即可；扩大搜索范围反而可能误改其他模块中的同名绑定。

### 3.5 注册与配置方式

两个 converter 分别注册：

```python
@register_model_converter("npu_rope")
class RoPEModelConfig(ModelCustomConfig):
    model_converter = NpuRoPEConverter


@register_model_converter("npu_rope_inplace_partial")
class InplacePartialRoPEModelConfig(ModelCustomConfig):
    model_converter = NpuInplacePartialRoPEConverter
```

默认配置保持：

```python
_DEFAULT_CONVERTERS = (
    "npu_rms_norm",
    "npu_moe_dispatch",
    "npu_gmm",
    "npu_rope",
    "npu_smla",
    "npu_mhc_pre",
)
```

在 Ascend 950 的目标配置中，在原有列表中追加 `npu_rope_inplace_partial`。例如：

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.converters import get_model_converter_config


model_converters = ModelConvertersContainer.Config(
    converters=[
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_rope_inplace_partial"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
    ]
)
```

两者可以共同配置：`npu_rope` 继续优化普通 RoPE 函数，
`npu_rope_inplace_partial` 只接管 partial 入口。DeepSeek-V4 的配置解析只要检测到其中
任意一个，就会生成 `[2, S, rotary_dim]` 的预计算实数 cache；因此只启用 partial
converter 也能获得 cache 复用，但不会自动启用普通 RoPE 的 `npu_rotary_mul` 替换。

## 4. 接入过程中的问题与修正

### 4.1 中间版本影响了原 `npu_rope` 路径

接入过程中曾观察到：在中间版本上使用相同 checkpoint、相同数据顺序，并同时开启固定 seed 和 deterministic 后，分别配置 `npu_rope` 与 `npu_rope_inplace_partial` 时，首步 loss 存在约 `0.001` 的差异。

新算子与旧算子的浮点执行顺序不同，二者 loss 出现小幅差异可以单独分析；但更关键的问题是：仅使用原 `npu_rope` 时，也不应该因为接入了一个未启用的新 converter 而改变计算结果。

这项现象本身不能证明全部差异都来自原路径变更，但对照接入基线后可以确认：中间实现为了复用逻辑，确实对原 `npu_rope` 做了两类不必要的改动：

1. 给原 `npu_rotary_mul` 路径增加了共享 wrapper。
2. 用继承和开关让 inplace converter 复用 `NpuRoPEConverter`。

这让原 converter 和新 converter 重新耦合，并使原路径中的 dtype 转换以及 `clone`、`split/slice/cat`、autograd 图结构不再能直接等同于接入前基线。无论它能解释多少 loss 差异，这都违反了“未启用新算子时保护旧路径”的设计目标。

最终修正为：

- 删除共享的 `npu_rotary_mul` wrapper，恢复原函数内直接调用 `torch_npu.npu_rotary_mul`。
- 将标准 RoPE 和 inplace partial RoPE 拆到 `rope.py`、`inplace_partial_rope.py` 两个文件中。
- `NpuInplacePartialRoPEConverter` 不再继承 `NpuRoPEConverter`，也不复用其广域函数替换 helper。
- 将 DeepSeek-V4 的 partial RoPE fallback 保存在 `apply_partial_rotary_emb_fallback()` 中，并在模块加载时作为默认入口。
- `NpuRoPEConverter` 只遍历 `_ROPE_REPLACEMENTS`，不再修改 partial RoPE 的模块级绑定。
- `NpuInplacePartialRoPEConverter` 只精确替换匹配模型所属模块中的 `apply_partial_rotary_emb_`；未匹配模型直接返回。
- 删除 DeepSeek-V4 专用模块拦截，让同一绑定契约可以被其他模型复用。
- 在模型初始化阶段预计算 `[2, S, rotary_dim]` 的 cos/sin cache，partial 调用直接复用，避免每次重复 real/imag 和 `repeat_interleave`。
- 删除当前支持路径不会使用的 complex-cache 分支，只接受预计算 `float32` cache，格式不符直接 fail-fast。
- 删除 partial 文件对 `_complex_to_interleaved_cos_sin`、`_select_freqs_cis` 的跨模块私有依赖，以及测试中的 `COMPLEX_CONVERSION_HELPER` 绕行字符串。
- 将只有一个调用点的算子调用和模型绑定安装逻辑直接内联，删除两个薄封装。

这个修正的核心不是让两种算子的浮点结果强行 bit-wise 相同，而是保证“未选择新
converter 时，旧路径完全不受接入影响”，同时让两个职责不同的 converter 能安全共同配置。

### 4.2 测试布局、流水线问题与 PR 状态

当前 RoPE converter 测试统一保留在
`tests/unit_tests/converters/test_rope.py`，没有另建
`test_rope_inplace_partial.py`；模型 cache 行为单独由
`tests/unit_tests/models/test_deepseek_v4_rope_cache.py` 覆盖。主要验证点包括：

- 标准 converter 与 partial converter 相互独立，并可共同配置；
- 不支持的模型未匹配绑定时直接返回，不触发 A5 或 CANN 依赖检查；
- 匹配模型只替换 `apply_partial_rotary_emb_`，不会替换标准
  `apply_rotary_emb_single_complex`；
- 匹配模型在非 A5 平台报错，缺少 CANN 算子时快速失败；
- complex cache 会触发格式断言，不再进入融合算子；
- 预计算实数 cache 路径的 `positions`、`inverse`、TND 形状及原地写回语义；
- 配置任一 RoPE converter 时，DeepSeek-V4 正确生成和重建预计算 cache。

此前一轮流水线记录为 `2 failed, 349 passed, 7 skipped`。两个失败来自测试模块持有旧函数
对象，而 `test_registry.py` 在同一 pytest 进程中 reload converter 模块后生成了新对象，
导致对象身份断言比较了 reload 前后的不同实例；并非融合计算结果失败。`a0a7678` 已改为
从当前 live module 取得被测对象，避免测试顺序依赖。

`e1f5938` 将 `_replace_one()` 收回 `NpuRoPEConverter` 类内；当前 PR 分支提交
`ef6102f` 进一步删除 complex-cache 死分支和两个单调用点薄封装，并同步调整测试。
远端 Git Hooks 已通过。本地相关 pytest 已尝试执行，但环境缺少 `torch`，在加载
`tests/conftest.py` 时即被阻断，因此没有新的本地 UT 通过数字；最终状态以 PR 444
最新流水线为准。

## 5. 最终调用链对照

### 5.1 `npu_rope`

```text
DeepSeek-V4 forward
  └─ apply_partial_rotary_emb_
      └─ apply_partial_rotary_emb_fallback
          ├─ slice
          ├─ apply_rotary_emb
          │   └─ npu_apply_rotary_emb_single_complex
          │       └─ torch_npu.npu_rotary_mul
          └─ torch.cat
              └─ 返回新 Tensor
```

### 5.2 `npu_rope_inplace_partial`

```text
DeepSeek-V4 forward
  └─ apply_partial_rotary_emb_
      └─ npu_apply_rotary_emb_partial_complex_
          ├─ DTensor → local Tensor
          ├─ positions / TND 或 BSND 形状处理
          ├─ 校验并选择预计算 [2, S, rotary_dim] cache
          │    └─ unbind → cos/sin（不再 repeat_interleave）
          ├─ inverse：sin 取负
          ├─ cos/sin → x_for_op device、fp32、contiguous
          └─ cann_ops_transformer.ops.inplace_partial_rotary_mul
              ├─ 原地更新 start:end
              └─ 返回原 x
```

### 5.3 两条路径的关键差异

| 项目 | `npu_rope` | `npu_rope_inplace_partial` |
| --- | --- | --- |
| 旋转算子 | `torch_npu.npu_rotary_mul` | CANN `inplace_partial_rotary_mul` |
| partial 处理 | slice 后旋转，再 cat | 算子内部只更新指定区间 |
| 输出对象 | 新 Tensor | 原输入 Tensor |
| 完整输出分配 | 需要 | 不需要 |
| 原路径兼容性 | 保持既有行为 | 独立 opt-in 路径 |
| 默认启用 | 是 | 否 |
| 目标绑定 | 普通 RoPE 函数及其 import binding | 仅模型所属模块的 `apply_partial_rotary_emb_` |
| cache 输入 | 支持 complex 或 DeepSeek-V4 预计算实数 cache | 只接受预计算 `float32` 实数 cache，complex fail-fast |
| 平台校验 | 沿用原支持范围 | 匹配到目标绑定后要求 A5（Ascend 950） |
| 能否共同配置 | 是 | 是 |

## 附录 A：相关概念详解

### A.1 Full RoPE、Partial RoPE 与 interleave

#### Full RoPE

Full RoPE 对一个 attention head 的全部旋转维度执行位置旋转。

#### Partial RoPE

Partial RoPE 只旋转最后一部分或指定的一部分维度。未旋转维度仍参与 Attention，但不编码旋转位置信息。

#### interleave

`interleave` 表示以相邻元素组成旋转对：

```text
(x0, x1), (x2, x3), (x4, x5), ...
```

因此每个 complex 频率需要展开为两个相同的 `cos` 和两个相同的 `sin` 系数，这就是
`repeat_interleave(2)` 的原因。当前 DeepSeek-V4 路径在模型 cache 初始化时只展开一次，
后续 standard/partial RoPE 共同复用。普通 `npu_rope` 仍保留 complex 输入支持；
`npu_rope_inplace_partial` 不再在调用内展开 complex 频率，而是对 complex cache 直接 fail-fast。

### A.2 Tensor 对象、底层 storage 与元数据

理解 view 和原地修改时，需要把 Tensor 拆成两层：

1. **底层 storage**：真正保存数值的内存。
2. **Tensor 元数据**：shape、stride、dtype、device、storage offset 等，用来解释这块内存。

两个 Tensor 可以拥有不同的 shape、stride 和 Python 对象身份，但指向同一块 storage。例如：

```python
base = torch.arange(8)
view = base.view(2, 4)
view[0, 0] = 99
```

修改 `view` 后，`base[0]` 也会改变，因为二者共享 storage。

### A.3 什么是 view

view 是“用另一套元数据解释同一块底层数据”的 Tensor。常见 view 操作包括：

- `view()`；
- 某些不需要复制的 `reshape()`；
- 切片；
- `split()`；
- `unsqueeze()` / `squeeze()`；
- `transpose()`。

view 的优点是不需要复制数据；代价是它与 base Tensor 存在别名关系，任何原地写入都可能同时影响 base 和其他 view。

需要特别注意：不能简单地把 `to_local()` 等同于报错中的 view。此前错误里的 `UnsqueezeBackward0` 明确指向 `unsqueeze()` 产生的 view；`to_local()` 负责取得 DTensor 的本地分片，是另一层边界。

### A.4 什么是 in-place 操作

in-place 操作直接修改输入 storage，PyTorch 中通常以下划线结尾，例如：

- `add_()`；
- `copy_()`；
- `mul_()`；
- 对切片赋值；
- 本次使用的 `inplace_partial_rotary_mul`。

PyTorch 为 Tensor storage 维护 version counter。每次原地修改都会增加版本号。autograd 在反向传播时会检查保存的 Tensor 版本，防止使用已经被意外改写的数据计算梯度。

### A.5 autograd 图中的“边”是什么

一个需要梯度的运算会在结果上记录 `grad_fn`。运算之间的输入输出关系组成 autograd 图：

```text
base --LinearBackward--> x --UnsqueezeBackward--> x_view --CustomBackward--> output
```

所谓“一条 autograd 边”，可以理解为反向传播时从一个梯度节点到前一个节点的依赖关系。

问题不是 PyTorch“不知道两条边谁先执行”。PyTorch知道图的拓扑顺序。真正的冲突是：

1. `unsqueeze()` 产生了共享 storage 的 view；
2. 自定义 `autograd.Function` 又原地修改这个 view；
3. 为正确处理普通 view+in-place，PyTorch 可能需要重建或调整 view 的反向历史；
4. 这种调整可能覆盖自定义 Function 自己声明的 backward 语义；
5. PyTorch 无法证明这样做一定正确，因此直接禁止该组合。

这就是下面这类错误的含义：

```text
Output of UnsqueezeBackward0 is a view and is being modified inplace.
```

### A.6 自定义 `autograd.Function` 的作用

CANN Python 包通常通过类似下面的形式接入自定义前反向：

```python
InplacePartialRotaryMulFn.apply(x, cos, sin, ...)
```

`forward()` 调用设备算子并保存 backward 所需信息，`backward()` 根据输出梯度计算输入梯度。

当自定义 Function 原地修改输入或原样返回某个输入时，它必须与 PyTorch 的 alias、view 和 version counter 规则保持一致。否则即使前向数值看起来正确，PyTorch 也不能保证反向梯度正确。

### A.7 leaf Tensor 与 non-leaf Tensor

#### leaf Tensor

通常由用户直接创建并设置 `requires_grad=True`，没有产生它的 `grad_fn`：

```python
x = torch.randn(4, requires_grad=True)
```

#### non-leaf Tensor

由另一个可微运算产生：

```python
y = x + 0
```

`y` 有 `grad_fn`，因此是 non-leaf。

PyTorch 对需要梯度的 leaf Tensor 原地修改限制更严格，因为 leaf 往往是参数或用户希望累积 `.grad` 的根节点。本次接入在算子入口显式拒绝需要梯度的 leaf Tensor。

### A.8 `clone()` 为什么能绕开一部分 view+in-place 问题

`clone()` 会分配新的 storage 并复制数值：

```text
原 Tensor/storage ──clone──> 新 Tensor/新 storage
```

新 Tensor 不再与原 view 共享 storage，因此在新 Tensor 上执行原地写入，不会修改原来的 base Tensor。

`clone()` 本身是可微的，梯度仍能通过 clone 节点传回输入。但它有明确成本：

- 分配一份同尺寸显存；
- 完整复制输入；
- 增加显存带宽开销；
- autograd 图中增加 clone 节点。

因此，`clone()` 是偏正确性和兼容性的解决方式，但会削弱原地融合算子的性能收益。能否不 clone，需要算子、view 关系和 backward 契约共同提供保证，不能仅根据“只修改一段数据”推断安全。

### A.9 DTensor 是什么

DTensor 用一个逻辑上的全局 Tensor 表示多卡分布式数据。它主要包含：

- 全局 shape；
- `DeviceMesh`，描述参与计算的设备网格；
- `placements`，描述全局 Tensor 如何放置在 mesh 各维；
- 每个 rank 实际持有的 local Tensor。

常见 placement：

| Placement | 含义 |
| --- | --- |
| `Shard(dim)` | 全局 Tensor 沿 `dim` 切分，每个 rank 持有一部分 |
| `Replicate()` | 每个 rank 持有完整副本 |
| `Partial()` | 每个 rank 持有待归约的部分结果，通常需要后续 collective |

placement 是“全局数据如何映射到各 rank”的分布式元数据；storage 是“当前设备上真正保存数值的内存”。RoPE 修改的是 local storage 中的数值，不会因为旋转一段数据就自动修改 placement。

### A.10 `to_local()` 与 `from_local()`

#### `to_local()`

从 DTensor 取得当前 rank 的 local Tensor：

```python
x_local = x_dtensor.to_local()
```

它不会执行一次完整的全局聚合，也不会把 Shard 自动变成 Replicate。得到的只是当前 placement 对应的本地数据。

#### `from_local()`

把一个 local Tensor 按指定 mesh 和 placements 包装回 DTensor：

```python
y_dtensor = DTensor.from_local(
    y_local,
    device_mesh=x.device_mesh,
    placements=x.placements,
    run_check=False,
)
```

`to_local()` / `from_local()` 都可以参与 autograd。若在 local Tensor 上生成了一个新的逻辑结果，通常需要用 `from_local()` 把这个结果重新接回 DTensor 计算链。

当前 inplace 路径利用共享 storage 修改 local 数据后返回原 DTensor。它在默认场景下用于功能接入；涉及 TP、view 和自定义 backward 的组合仍应作为独立问题验证，不能把“前向共享数据已经改变”等同于“反向图一定正确”。

### A.11 共享 storage 与 placement 元数据的区别

可以用下面的类比理解：

```text
storage         = 某一本书当前页上的实际文字
Tensor 元数据    = 用几行几列、从哪个偏移开始阅读这页
DTensor placement = 这套书如何分发到不同读者手里
```

原地 RoPE 改的是“当前页上的文字”。`unsqueeze` 改的是“如何阅读这页”的 shape/stride。placement 描述的是“这一页属于全书的哪一部分”。三者相关，但不是同一类信息。

### A.12 TP=2、EP=2、4 卡时分别切什么

在简化的四卡逻辑网格中，每个 rank 可以看成同时拥有一个 EP 坐标和一个 TP 坐标：

```text
             TP 0          TP 1
EP 0       (ep0,tp0)     (ep0,tp1)
EP 1       (ep1,tp0)     (ep1,tp1)
```

实际 rank 编号由 `DeviceMesh` 的维度顺序决定，上图只表达逻辑关系。

#### EP 切什么

EP（Expert Parallel）主要切 MoE experts。不同 EP rank 保存或处理不同专家，token 根据 router 结果发送到目标专家。

#### TP 切什么

TP（Tensor Parallel）切单个稠密算子内部的特征维度。对于 DeepSeek-V4 的：

```python
"attention.pre_attention.wq_b": colwise_parallel(
    use_local_output=False,
)
```

`wq_b` 采用 column-wise parallel，切的是线性层输出特征，也就是 Query heads 对应的输出维，而不是在 RoPE 位置直接切 sequence。

`use_local_output=False` 让 `wq_b` 输出保持为 DTensor，使后续运算和反向传播能够保留分布式 placement 信息。因此 `wq_b → unflatten → partial RoPE` 这一段可能收到 DTensor。

Sequence Parallel 是另一种并行语义。模型其他位置可能临时按 sequence 切分再在 Attention 入口重新分发，但不能因此把 `wq_b` 的 TP 简单理解成“RoPE 把序列切成两半”。

### A.13 为什么“只修改一段数据”仍可能触发 autograd 错误

原地修改范围小，只说明数值写入区域有限；autograd 关心的还有：

- 输入是不是 view；
- view 的 base 是否还被其他节点使用；
- backward 是否保存了修改前的数据；
- 自定义 Function 是否声明了正确的 dirty/alias 关系；
- 输出是否原样返回输入；
- version counter 是否符合预期。

因此，“只改 `start:end`”不能自动推出“不需要 clone”或“反向一定安全”。性能优化减少的是数据移动，autograd 正确性约束解决的是计算历史和别名关系，两者是不同维度的问题。

### A.14 `torch.compile` 的基本流程

`torch.compile` 并不是简单地把 Python 函数交给某个算子库。典型流程是：

```text
Python/PyTorch 程序
  └─ TorchDynamo 捕获计算图
      └─ AOTAutograd 处理前向和反向
          └─ TorchInductor 优化 IR
              └─ 硬件后端 Codegen
                  └─ 执行编译结果
```

Dynamo 成图期间希望看到稳定、可追踪的 Tensor 运算。文件 I/O、即时编译扩展、动态导入副作用等 Python 行为不一定能进入图。

### A.15 “算子支持图模式”不等于任意 Python wrapper 都能被 `torch.compile`

算子文档中的“支持图模式”通常表示该算子能够作为图中的算子节点，由对应图编译或执行系统处理。

但 `torch.compile` 首先需要追踪 Python wrapper。如果 wrapper 在第一次前向中执行下面的行为：

```python
torch.utils.cpp_extension.load(...)
```

Dynamo 仍可能因为不能追踪动态编译和加载过程而中断。两句话并不矛盾：

- 算子内核支持进入图；
- 负责找到、编译或加载该内核的 Python 初始化过程未必可追踪。

通常需要在成图前完成算子注册和加载，或者让 wrapper 在图内只调用已经注册好的 `torch.ops.*`/自定义算子入口。

### A.16 `inductor_npu_ext` 的作用

`inductor_npu_ext` 是 NPU 的 TorchInductor 扩展。导入它的主要作用是注册 NPU scheduling、wrapper codegen 等后端能力，让 Inductor 能把图编译到 Ascend NPU。

```python
import inductor_npu_ext
```

这行代码通常依赖 import side effect，本身不直接参与 RoPE 数学计算。

它与 `inplace_partial_rotary_mul` 的关系是：

- `inplace_partial_rotary_mul` 提供具体算子及其前反向实现；
- `inductor_npu_ext` 提供 NPU 上更广义的 Inductor 编译后端能力。

安装了 `inductor_npu_ext` 不代表任意外部算子的 Python 加载代码都一定可被 Dynamo 追踪；反过来，删除这行 import 也不能证明算子 wrapper 本身支持完整 NPU compile 流程。

### A.17 延迟导入与函数缓存的取舍

延迟导入是把依赖加载从模块 import 阶段推迟到第一次实际使用。函数缓存是其中一种
可选实现方式：

```python
@cache
def get_op():
    from package import op
    return op
```

第一次调用执行 import 并缓存结果，后续调用直接返回同一个对象。本次最终实现采用
融合入口内的局部导入，没有使用上面示例中的 `@cache`。匹配到目标绑定后，converter
阶段会先导入算子做依赖检查；真正执行时再次出现的 import 会命中 Python 的模块缓存，
但仍会执行局部名称绑定。

它解决的是：

- 默认路径不必提前加载可选依赖；
- 避免显式可变全局变量；
- 对匹配模型在 converter 阶段快速暴露缺失依赖。

它不自动解决的是：

- 算子二进制与 CANN/torch_npu 的版本兼容；
- 第一次调用发生在 `torch.compile` tracing 内时的可追踪性；
- 自定义算子的 autograd、view 和 in-place 合法性。

## 参考资料

- [torchtitan-npu PR !444](https://gitcode.com/cann/torchtitan-npu/merge_requests/444)
- [torchtitan-npu RoPE 融合算子说明](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/feature_guides/npu_fused_ops.md#in-place-partial-rope)
- [torchtitan-npu 安装与版本配套](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/installation.md)
- [CANN 9.2.0 daily0725 下载](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260725000025310/)
- [ops-transformer：`inplace_partial_rotary_mul`](https://gitcode.com/cann/ops-transformer/blob/master/torch_extension/cann_ops_transformer/docs/zh/inplace_partial_rotary_mul.md)
- [PyTorch DTensor 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch `torch.compile` 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [inductor-npu-ext README](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/README.md)
