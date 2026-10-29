# `inplace_partial_rotary_mul` 接入介绍

> 适用场景：向 torchtitan-npu 开发者介绍本算子的接入背景、最终架构和执行流程。  
> 当前实现基线：`inplace-rope-latest` 分支，提交 `ef6102f`，对应
> [torchtitan-npu PR !444](https://gitcode.com/cann/torchtitan-npu/merge_requests/444)。  
> 更完整的开发过程见：[[inplace-partial-rotary-mul-接入复盘]]。

## 1. 开场可以这样讲

这次接入的目标，是优化 DeepSeek-V4 中的 partial RoPE。

原实现先从输入中切出需要旋转的维度，调用普通 RoPE 得到新 Tensor，再通过
`torch.cat` 把旋转区间和未旋转区间重新拼接。这个过程会产生旋转结果和完整输出等
临时 Tensor，也会对未旋转区域做额外读写。

`cann_ops_transformer.ops.inplace_partial_rotary_mul` 可以直接修改
`x[..., start:end]`，并保持其他维度不变。因此这次接入主要做了三件事：

1. 在模型侧提供一个稳定、可替换的 partial RoPE 入口。
2. 用独立 ModelConverter 把这个入口替换为 CANN 原地融合算子。
3. 复用模型初始化阶段生成的 cos/sin cache，避免每次调用重复转换频率。

## 2. 原路径为什么有优化空间

原 partial RoPE 的核心流程可以概括为：

```text
x
 ├─ slice：取 x[..., start:end]
 ├─ apply_rotary_emb：生成旋转后的新 Tensor
 └─ torch.cat：和未旋转区域重新拼接
      └─ 返回新的完整 Tensor
```

主要开销包括：

- partial RoPE 旋转结果的临时 Tensor；
- `torch.cat` 产生的完整输出 Tensor；
- 拼接时对未旋转区域的额外显存读写；
- 多个 eager 算子的调度开销。

融合后的流程是：

```text
x + cos/sin cache + [start, end]
  └─ inplace_partial_rotary_mul
       ├─ 只更新 x[..., start:end]
       ├─ 其他区域保持不变
       └─ 返回时继续使用原 x
```

## 3. 为什么使用独立 converter

最终没有把新逻辑直接塞进原来的 `npu_rope`，而是新增
`npu_rope_inplace_partial`，原因是两者职责不同：

| Converter | 负责的内容 |
| --- | --- |
| `npu_rope` | 替换普通 RoPE 函数，底层使用 `torch_npu.npu_rotary_mul` |
| `npu_rope_inplace_partial` | 只替换模型模块中的 `apply_partial_rotary_emb_` 入口 |

这样做有三个好处：

- 未启用新 converter 时，已经验证的普通 RoPE 路径不受影响；
- A5 平台和 `ops-transformer` 依赖只在匹配到 partial RoPE 入口后检查；
- 两个 converter 修改不同绑定，可以同时配置。

在 A5 配置中，不需要用 partial converter 替换 `npu_rope`，而是在原列表中追加：

```python
get_model_converter_config("npu_rope")
get_model_converter_config("npu_rope_inplace_partial")
```

## 4. 模型侧如何提供替换入口

文件：`torchtitan_npu/models/deepseek_v4/model.py`

模型保留原有功能实现作为 fallback：

```python
def apply_partial_rotary_emb_fallback(
    x,
    freqs_cis,
    partial_slice,
    inverse=False,
    positions=None,
):
    start, end = partial_slice
    x_rot = apply_rotary_emb(
        x[..., start:end],
        freqs_cis,
        inverse=inverse,
        positions=positions,
    )
    return torch.cat([x[..., :start], x_rot, x[..., end:]], dim=-1)


apply_partial_rotary_emb_ = apply_partial_rotary_emb_fallback
```

Attention、Indexer、Compressor 等调用点只调用 `apply_partial_rotary_emb_`，不感知当前
使用 fallback 还是融合算子。converter 在模型转换阶段修改这个模块级绑定。

这也是插件仓常用的接入方式：模型提供稳定扩展点，NPU 算子选择放在 converter 中，
不在模型 forward 里硬编码平台判断或 CANN 依赖。

## 5. converter 如何生效

文件：`torchtitan_npu/converters/kernels/inplace_partial_rope.py`

converter 的执行顺序如下：

```text
取得 model.__class__.__module__ 对应模块
  ├─ 模块不存在或没有 apply_partial_rotary_emb_
  │    └─ 直接 return，不做任何替换
  └─ 找到目标绑定
       ├─ 校验设备必须是 A5
       ├─ 校验能否导入 inplace_partial_rotary_mul
       ├─ 已经替换过则直接 return
       └─ setattr 精确替换当前模型模块的绑定
```

关键代码可以简化为：

```python
binding_name = "apply_partial_rotary_emb_"
model_module = sys.modules.get(model.__class__.__module__)
if model_module is None or not hasattr(model_module, binding_name):
    return

_validate_inplace_partial_rope_platform()
from cann_ops_transformer.ops import inplace_partial_rotary_mul

if getattr(model_module, binding_name) is npu_apply_rotary_emb_partial_complex_:
    return
setattr(model_module, binding_name, npu_apply_rotary_emb_partial_complex_)
```

这里没有 DeepSeek-V4 模块名前缀拦截。DeepSeek-V4 是当前首个接入模型，但算子本身
不是模型专用。其他模型如果没有这个绑定会直接跳过；如果提供同名绑定，则必须遵循
相同的五参数接口和预计算 cache 契约。

## 6. cos/sin cache 是怎样准备的

DeepSeek-V4 的配置解析在检测到 `npu_rope` 或 `npu_rope_inplace_partial` 任一 converter
时，都会设置：

```python
use_npu_rope = True
```

随后模型在 `precompute_rope_cache()` 中一次性把 complex 频率转换为 interleave
cos/sin cache：

```python
cache = (
    torch.view_as_real(freqs_cis)
    .movedim(-1, 0)
    .repeat_interleave(2, dim=-1)
)
```

cache 的格式为：

```text
[2, cache_seq_len, rotary_dim], float32
 │
 ├─ cache[0]：cos
 └─ cache[1]：sin
```

对于 Compressor 或 MTP 的不同压缩比例，模型会把下采样后的 cache 分段拼接，并记录
各分段偏移。调用 partial RoPE 前，Compressor 会先选出对应压缩比例的 cache 段。

当前 inplace partial 路径只接受这种预计算 FP32 cache，不再兼容 complex cache。
如果传入 complex cache，`_select_precomputed_rope_cache()` 的格式断言会直接 fail-fast。

虽然融合函数当前仍名为 `npu_apply_rotary_emb_partial_complex_`，但 `complex_` 只是保留的
既有内部命名，不代表它仍接受 complex cache。

## 7. 融合入口内部的数据流

融合入口收到输入后，按下面的顺序处理：

```text
x / freqs_cis / partial_slice / inverse / positions
  │
  ├─ 校验 partial_slice=[start, end]
  ├─ DTensor → local Tensor
  ├─ 拒绝 requires_grad=True 的 leaf Tensor
  ├─ 根据 positions 判断 BSND 或 TND 形状
  ├─ 从预计算 cache 中选择当前位置
  ├─ unbind(0) 得到 cos、sin
  ├─ inverse=True 时执行 sin = -sin
  ├─ cos/sin → x_for_op.device、float32、contiguous
  └─ inplace_partial_rotary_mul
       ├─ rotary_mode="interleave"
       ├─ partial_slice=[start, end]
       └─ 原地更新后返回原 x
```

### 7.1 输入形状适配

CANN 算子按四维形式处理输入，当前只通过 `unsqueeze` 增加维度：

| 输入语义 | 原形状 | 算子形状 |
| --- | --- | --- |
| BSND | `[B, S, N, D]` | `[B, S, N, D]` |
| 三维普通输入 | `[B, S, D]` | `[B, S, 1, D]` |
| TND Attention | `[T, N, D]` | `[1, T, N, D]` |
| TND Compressor | `[T, D]` | `[1, T, 1, D]` |

`unsqueeze` 得到的 Tensor 与原 Tensor 共享 storage，因此算子对 `x_for_op` 的原地更新
会反映到原输入上。

### 7.2 positions 与 inverse

- `positions is None`：直接使用 cache 的前 `seqlen` 个位置。
- 传入 `positions`：对预计算 cache 做一次位置索引。
- 二维 `positions`：当前融合 RoPE 使用第一行，表示 batch 间共享位置。
- `inverse=True`：cos 保持不变，只对 sin 取负。

## 8. 当前实现刻意删除了什么

最终版本进一步做了三项收敛：

1. 删除 complex-cache 兼容分支，因为启用 partial converter 后，DeepSeek-V4 必定生成
   预计算 FP32 cache，complex 分支不在当前支持路径上。
2. 删除 partial 文件对 `_complex_to_interleaved_cos_sin`、`_select_freqs_cis` 的依赖，
   只复用 `_select_precomputed_rope_cache`。
3. 将只有一个调用点的 `_apply_inplace_partial_rotary_interleave_()` 和
   `_enable_inplace_partial_rope()` 直接内联，减少薄封装。

这使普通 RoPE 和 inplace partial RoPE 的职责更清晰：共享 cache 选择规则，但不共享
算子替换流程。

## 9. 测试主要覆盖什么

测试统一放在：

- `tests/unit_tests/converters/test_rope.py`
- `tests/unit_tests/models/test_deepseek_v4_rope_cache.py`

主要覆盖：

- 普通 RoPE converter 不修改 partial RoPE 绑定；
- partial converter 与普通 converter 相互独立；
- 未匹配模型在平台检查和 CANN 依赖检查前直接返回；
- 匹配模型在非 A5 平台报错；
- 缺少 `inplace_partial_rotary_mul` 时快速失败；
- 只替换 `apply_partial_rotary_emb_`，不替换普通 RoPE 函数；
- complex cache 触发格式断言；
- 预计算 cache 的连续位置、`positions`、`inverse`、TND 形状和原地写回；
- 配置任一 RoPE converter 时，模型正确生成和重建实数 cache。

当前提交已通过远端 Git Hooks。本地相关 pytest 曾尝试执行，但本地 Python 环境缺少
`torch`，在加载 `tests/conftest.py` 时即被阻断，因此最终 UT 结果应以 PR 流水线为准。

## 10. 介绍时建议强调的设计点

可以把整个接入总结成下面五句话：

1. **模型只提供稳定入口。** 调用点统一走 `apply_partial_rotary_emb_`。
2. **converter 决定是否替换。** 未匹配模型不受影响，匹配后才检查 A5 和 CANN 依赖。
3. **普通 RoPE 路径保持不变。** 新能力使用独立 converter，并且可以和 `npu_rope` 共存。
4. **热点调用只消费预计算 cache。** real/imag 和 `repeat_interleave` 在模型初始化时完成一次。
5. **融合算子原地写指定区间。** 避免 partial 输出和完整 `cat` 输出的额外分配。

## 11. 常见问题

### Q1：为什么不直接修改 `npu_rope`？

因为新算子有 A5 和 `ops-transformer` 依赖，而且只优化 partial RoPE。独立 converter
可以保护已经验证的普通 RoPE 路径，并允许按需启用。

### Q2：两个 converter 能同时配置吗？

可以。`npu_rope` 替换普通 RoPE 函数，`npu_rope_inplace_partial` 只替换 partial 入口，
两者没有绑定冲突。

### Q3：为什么其他模型没有匹配到时直接 return？

算子不是 DeepSeek-V4 专用，但并不是所有模型都有 partial RoPE 入口。没有目标绑定说明
该 converter 对当前模型不适用，因此不做替换，也不应触发 A5 或可选依赖错误。

### Q4：其他模型只要有同名函数就能使用吗？

还需要满足完整契约：五个参数语义一致，第二个参数是
`[2, cache_seq_len, rotary_dim]` 的预计算 FP32 interleave cos/sin cache，并且输入形状和
`partial_slice` 满足算子要求。

### Q5：为什么不再支持 complex cache？

启用 partial converter 后，DeepSeek-V4 已经通过 `use_npu_rope=True` 生成预计算实数
cache。继续保留 complex 分支只会维护当前支持路径用不到的代码，并重新引入对普通
RoPE complex helper 的耦合。

### Q6：为什么函数名里还有 `complex_`？

这是保留的内部函数名，用于减少与本次目标无关的调用点重命名。真正的数据契约由
docstring 和 `_select_precomputed_rope_cache()` 的格式断言决定。

### Q7：为什么算子在 converter 和融合入口中各导入一次？

converter 中的导入用于在模型转换阶段快速检查依赖；融合入口中的局部导入用于实际
取得算子对象。Python 会复用已经加载的模块，不需要额外的可变全局变量或缓存函数。

## 12. 一分钟总结

这次接入没有改写普通 `npu_rope`，而是让 DeepSeek-V4 的 partial RoPE 调用统一经过
一个可替换入口，再由独立 converter 在 A5 上把它切换为
`inplace_partial_rotary_mul`。模型初始化时预先生成 FP32 interleave cos/sin cache，
融合入口只做位置选择、广播形状整理和必要的 dtype/device 处理，然后由 CANN 算子原地
更新指定旋转区间。最终减少了 partial 中间结果、`torch.cat` 和完整输出分配，同时保持
未启用该 converter 时的旧路径不变。

## 参考链接

- [torchtitan-npu PR !444](https://gitcode.com/cann/torchtitan-npu/merge_requests/444)
- [`inplace_partial_rotary_mul` 算子说明](https://gitcode.com/cann/ops-transformer/blob/master/torch_extension/cann_ops_transformer/docs/zh/inplace_partial_rotary_mul.md)
- [CANN 9.2.0 daily0725](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260725000025310/)
