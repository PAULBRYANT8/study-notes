# DeepSeek-V4 PP 适配重构:hc_head 放进最后一个 transformer layer

> 创建:2026-05-29 · 更新:2026-05-30
> 最终基线:master `0a70af2`
> 涉及文件:`model.py`、`parallelize.py`、`pipeline_parallel.py`、`state_dict_adapter.py`、`tests/unit_tests/models/test_deepseek_v4_pp_static.py`
>
> 说明:文档名里的 "norm" 是早期方案残留。最终落地的是**把 hc_head 放进最后一个主干 layer**(下面第 3 节),"挂 norm 下"那版(scheme ②)已被证伪、放弃(见第 2 节)。

---

## 1. 背景:DeepSeek-V4 是超连接(Hyper-Connections / MHC)模型

DeepSeek-V4 的残差流不是普通的 `[B, S, D]`,而是带了一个"超连接多路"维度 `hc_mult`:

- `tok_embeddings` 之后,隐藏态被扩展成 `[B, S, hc_mult, D]`;
- **每个 transformer block 的输入/输出都是 `[B, S, hc_mult, D]`**(block 内部用 `hc_pre`/`hc_post` + 各自的 `hc_attn_*`/`hc_ffn_*` 参数维护这 `hc_mult` 条并行残差流);
- `hc_head` 做**出口聚合**:把 `hc_mult` 条流坍缩回 `[B, S, D]`,再走 `norm` → `output` 出 logits。

有**两类**名字相近但完全不同的 "hc" 参数,别混:
- `hc_attn_*` / `hc_ffn_*`:**每个 block 各一份**,block 内部流混合用;
- `hc_head_*`:出口聚合用。

### MTP 与 hc_head 的关系(关键)

上游 commit `94a3bd9`(MR!263)起,**主干和 MTP 的 hc_head 已经分开**:
- 主干:一份 `hc_head`;
- 每个 MTP 模块:**各自**的 `mtp_hc_head` + `mtp_norm`,且**在 MTP 模块自己的 forward 内部**调用,返回 `(prev_embed, normed)`。

这一点很重要:**MTP 不再共享主干 hc_head**,所以后面动主干 hc_head 的位置,完全不影响 MTP。

---

## 2. 演进历程(为什么最后选"进最后一层")

PP 切分依赖上游 `pipeline_module_split` + `generate_llm_fqn_per_model_part`,通用切分器只认顶层 `{tok_embeddings, layers.*, norm, output}`,**不在保留名单里的顶层模块会在每个 stage 被置 None**。

| 方案 | 做法 | 结果 |
|---|---|---|
| 初始(splice) | hc_head 作 root 模块 + 在 PP FQN 清单里 splice 进最后一段 | ✅ 能跑(FSDP:root 箱子全程开;PP:splice)。但要维护 ~100 行 splice/validate |
| scheme ②(norm 下) | hc_head 挂成 `norm.hc_head`,但**在 norm.forward 之外**调用 | ❌ **FSDP 崩**(见下) |
| **最终(进最后一层)** | hc_head 放进最后一个主干 layer,**在该 layer 的 forward 内部**聚合 | ✅ FSDP 正确 + PP 零定制 + 不连累 MTP |

### scheme ② 为什么崩(FSDP 箱子开关时机)

FSDP2 把参数按模块分"箱子"(FSDP 单元),**只在某个箱子的"主人模块"开始 forward 的那一刻才 all-gather 该箱子的参数,算完即拆回**。上游 `apply_fsdp`(`llama4/.../parallelize.py`)把 `[model.norm, model.output]` 包成**一个箱子**。

scheme ② 把 hc_head 挂在 `norm` 下 → 它的参数归 `[norm, output]` 这个箱子;但调用顺序是先 `hc_head(h)` 再 `norm(h)`,**hc_head 跑的时候 norm.forward 还没开始,箱子没 all-gather**,于是 hc_head 参数还是 sharded(DTensor),和普通激活一混就报:
```
RuntimeError: got mixed torch.Tensor and DTensor
```
(实测命中,见 `log/...20260530091620.log`。)

**根因一句话:把 hc_head 挂到 norm 下、却在 norm.forward 之外调用它,等于用一个"还没解 shard"的箱子里的参数。**

### 为什么"进最后一层"就对了

让 hc_head **在最后一个主干 layer 的 forward 内部**运行:调用 `last_layer(h)` → 触发该 layer 的 `__call__` → 前置钩子先 all-gather 这个 layer 的箱子(含 hc_head 参数)→ 再跑 layer.forward,此刻 hc_head 参数已拼齐(普通 Tensor)。**这正是 MTP 的 `mtp_hc_head` 的做法**(在 MTP 模块自己 forward 内调用)。而且 MTP 已解耦,主干 hc_head 怎么动都不影响它。

---

## 3. 最终方案与逐文件改动(基线 `0a70af2`)

核心:`hc_head` 从 root 模块改为**最后一个主干 layer 的子模块**,在该 layer forward 末尾聚合。

### 3.1 `model.py`
- `DeepSeekV4TransformerBlock`:加 `self.is_last_layer = layer_id == n_layers - 1`;**仅最后一层** build `self.hc_head`;forward 末尾:
  ```python
  x = self.hc_post(x, residual, post, comb)   # [B,S,hc_mult,D]
  if self.is_last_layer:
      x = self.hc_head(x)                      # → [B,S,D]
  return x
  ```
  init_weights 里对应初始化 hc_head 三个参数。MTP 层 `is_last_layer=False`(它有自己的 `mtp_hc_head`),不受影响。
- `DeepSeekV4Model`:删掉 root `self.hc_head`;forward 里删 `self.hc_head(h)` 与 `_validate_last_stage_hc_head`(最后一层已聚合,`h` 到这里已是 `[B,S,D]`);init_weights 删 root hc_head 初始化;删 `_validate_last_stage_hc_head` 方法。

### 3.2 `pipeline_parallel.py`(净删约 150 行)
hc_head 进 layer 后 root 结构变成纯 `{tok_embeddings, layers, norm, output}`,**通用切分零定制**:删 `generate_deepseek_v4_fqn_per_model_part`(splice)、`validate_deepseek_v4_stage_modules` 及辅助、`DEEPSEEK_V4_OUTPUT_MODULES`;`_build_deepseek_v4_stage_split` 直接调用上游 `generate_llm_fqn_per_model_part`。

### 3.3 `parallelize.py`
- 删 root 的 hc_head TP 块;
- 在**按层循环**里,对 `is_last_layer` 的 block:注册 `hc_head_fn/base/scale`(Replicate)+ `layer_plan["hc_head"] = hc_head_plan`(hc_head 在 block 内的输入就是 hc_post 输出,和原来 root 下同一个 Shard(1) 布局,plan 平移)。

### 3.4 `state_dict_adapter.py`
- 主干 hc_head FQN:`hc_head_*` → **`layers.{n-1}.hc_head.*`**(from_hf 用具体末层下标);
- 因为 to_hf 对任何 `layers.*` key 走 `_map_layer_key`(抽象 key 查表),额外补抽象逆映射 `layers.{}.hc_head.* → hc_head_*`,保证导出方向也对。

### 3.5 测试
更新 `test_deepseek_v4_pp_static`:删生成器/校验用例,改为断言"pipeline 已无 splice/validate/常量、用上游生成器"+"model 在最后一层 build/调用 hc_head"。

---

## 4. 为什么这样能同时满足 FSDP + PP + MTP

| | FSDP(EP) | PP 切分 | MTP |
|---|---|---|---|
| hc_head 在最后一层 forward 内聚合 | ✅ 该 layer 箱子开时参数已拼齐 | ✅ root 结构通用,零 splice;最后一层随通用切分进最后 stage | ✅ MTP 各自 `mtp_hc_head`,与主干 hc_head 无关 |

---

## 5. 好处
1. **PP 切分零定制**:删掉 ~150 行 splice/validate,降低上游同步维护成本;
2. **FSDP 天生正确**:hc_head 在 owning 模块 forward 内运行,参数按需 all-gather;
3. **不连累 MTP**:MTP 已解耦,主干 hc_head 独立处理;
4. **语义清晰**:和 MTP 的 `mtp_hc_head` 同构(都在各自 block forward 内做出口聚合)。

---

## 6. 验证情况

**本机已验(无 torch,仅静态层面):**
- 5 个文件 `py_compile` 全过;hermetic 静态测试 **7/7**;pre-commit 轻量钩子全过(trailing/eof/check-ast/**ufmt/flake8/codespell**)。

**需在 NPU / 训练环境补验:**
1. **能跑通**:`bash scripts/run_train_deepseekv4.sh`(EP+FSDP),确认 loss 正常、无 DTensor 报错;
2. **checkpoint**:FQN 变 `layers.{n-1}.hc_head.*`,确认 HF 权重经 adapter 能加载/导出;
3. **TP(tp>1)**:hc_head TP 已搬进按层循环,但 block 内布局与 norm/输出侧的交互**需实测**(代码注释已标注);
4. **PP(pp>1)**:确认最后一层(含 hc_head)被通用切分正确放到最后 stage;
5. loss 对比:`--debug.seed=42 --debug.deterministic`(结构重构、数学等价,但参数布局变了)。

---

## 7. 注意:checkpoint 不向后兼容
旧布局参数 FQN 是 `hc_head.*`(root),新布局是 `layers.{n-1}.hc_head.*`。
- 从 **HF 权重**经 `state_dict_adapter` 加载:已更新映射,直接可用;
- 从**旧的、按旧 FQN 存的 torchtitan checkpoint** 加载:需要做一次 key 迁移(`hc_head.* → layers.{n-1}.hc_head.*`)。
