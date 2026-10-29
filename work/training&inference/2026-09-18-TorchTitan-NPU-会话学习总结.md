# TorchTitan-NPU 会话学习总结：反复追问的概念与简单例子

整理日期：2026-09-18

这份笔记按主题整理本次会话，重点记录：不熟悉的术语、反复追问的原因、帮助理解的小例子，以及后来修正过的结论。代码路径和执行顺序以会话核对时的版本为背景；后续重构可能改变路径或行为。

## 一、这次主要在解决什么问题？

主线是为 DeepSeek-V4 接入 `swiglu_group` 融合激活，并让低精训练下的路由专家使用它。围绕这条主线，讨论了：

1. 对比远端 `swiglu_group` 分支与本地实现，调整实现方式。
2. 理解量化模块如何继承融合专家的 `forward`，以及配置转换和 override 的执行顺序。
3. 排查低精 + 融合激活触发的符号形状、连续性判断和 AOTAutograd 报错。
4. 比较 `cat` 与预分配 buffer 后 slice 写入的区别，最终根据实测恢复 `cat`。
5. 分析量化 BMM、前向、反向重计算和 profiler 中的调用次数。
6. 补充并精简 UT，审查 PR、解决冲突、更新 README 和执行脚本。
7. 开启确定性计算，保存第 20 step 的权重，再加载权重训练。

其中追问最多的是：**连续到底指什么、为什么运行时知道 shape 而编译时不知道、为什么前向能走而反向构图会报错，以及“加载权重”和“断点续训”有什么区别。**

## 二、重点一：SwiGLU、W1/W3 和 grouped_mm 的关系

### 反复提出的问题

- SwiGLU 只是计算激活，为什么涉及 W1、W3？
- W1 和 W3 是在调用融合算子时拼接的吗？正常情况下是否分开？
- `torch._grouped_mm()` 的作用是什么？
- 示例里的 `gate / up / cat / swiglu_group` 对应哪段真实代码？

### 1. SwiGLU 接收两路投影结果

一个专家的简化计算是：

```python
gate = x @ W1
up = x @ W3
hidden = silu(gate) * up
out = hidden @ W2
```

其中 `silu(z) = z * sigmoid(z)`，乘法 `*` 是逐元素乘法。

- W1：产生 gate 分支。
- W3：产生 up 分支。
- SwiGLU：对 gate 做 SiLU，再与 up 逐元素相乘。
- W2：把中间隐藏维度投影回输出维度。

因此，W1/W3 属于 SwiGLU 所在的整个 FFN 结构；不是 SwiGLU 激活算子内部必然拥有的权重。

### 2. 当前讨论的 cat 拼接的是激活，不是权重

```python
gate = torch._grouped_mm(x, W1, ...)  # [R, F]
up = torch._grouped_mm(x, W3, ...)    # [R, F]
packed = torch.cat((gate, up), dim=-1)  # [R, 2F]
hidden = swiglu_group(packed, ...)      # [R, F]
out = torch._grouped_mm(hidden, W2, ...)
```

实际参数常存成 `[E,F,D]` 等布局，调用矩阵乘前还需要转置。上面的公式省略了这部分。

**纠正容易混淆的说法：W1/W3 仍是独立参数，拼接的是 `x @ W1` 和 `x @ W3` 的结果。** `cat` 的反向先把梯度传回 gate/up，再经矩阵乘的反向计算 W1/W3 的梯度。

### 3. grouped_mm：按分组边界使用不同专家的矩阵

假设有 5 个已按专家排列的 token：前 2 个给专家 0，后 3 个给专家 1。

```python
counts = [2, 3]
offsets = [2, 5]  # 累计结束位置
```

分组矩阵乘可以理解为：

```python
y0 = x[0:2] @ W[0]
y1 = x[2:5] @ W[1]
y = torch.cat((y0, y1), dim=0)
```

这里的 `cat` 只是解释分组结果如何按行组合，与上面 gate/up 沿最后一维拼接不是同一个位置。

它允许不同专家拿到不同数量的 token；普通 BMM 通常要求每个 batch 使用相同的矩阵维度。

### 4. 常见维度字母

| 符号 | 本次讨论中的含义 |
| --- | --- |
| E | 专家数，实际参与计算的参数也可能是分片后的本地专家 |
| D | 模型输入/输出隐藏维度 |
| F | FFN 中间隐藏维度 |
| R | 当前本地专家输入中路由后的 token 行数 |

R 不等于某一个专家的 token 数。各专家的数量在 `counts` 中，总和通常构成本次输入的 R。各专家分配不均，不代表 R 在所有并行配置下都一定动态；跨 rank 路由等路径才可能让本地 R 依赖运行时数据。

## 三、重点二：shape、stride 和“连续”到底是什么意思？

### 这是本次反复追问最多的一组问题

- 为什么梯度 shape 是 `[R,F]`，stride 却是 `[2F,1]`？
- 每一行中间隔着另一分支的数据，跳过去读不就行了吗？
- 普通二维张量换行也要移动 F 个元素，为什么就算连续？
- 连续是不是指物理地址连续？

### 1. shape 管“有多少”，stride 管“怎么找”

二维张量元素的存储位置可简化写成：

```text
位置(i,j) = storage_offset + i × stride[0] + j × stride[1]
```

stride 的单位是**元素个数**，不是字节。数据类型的每元素字节数决定最终字节偏移。

==PyTorch 的连续性说的是张量逻辑顺序与底层存储偏移的关系，不要求从操作系统物理页是否连续来理解==。

### 2. 为什么 cat 后切出的梯度 stride 是 `(2F,1)`？

设 `R=2，F=3`。packed 梯度的逻辑内容是：

```text
          gate 部分       up 部分
第 0 行   g00 g01 g02     u00 u01 u02
第 1 行   g10 g11 g12     u10 u11 u12
```

整个 packed 的 shape 是 `[2,6]`，stride 是 `(6,1)`。

`cat` 反向把它沿最后一维切开。gate 对应的视图：

```python
dgate = dpacked[:, :3]
# shape = [2,3]
# stride = (6,1)
```

这个视图每行只看到 3 个元素，但行首之间仍隔 6 个元素，因为共享了原来 packed 的存储。

```text
gate 逻辑遍历：g00 → g01 → g02 → g10 → g11 → g12
  对应存储偏移： 0     1     2     6     7     8
```

偏移 3、4、5 是 up 分支的数据，所以 gate 视图不是普通行连续布局。

### 3. 普通连续张量“换行跳 F”为什么没有矛盾？

普通 `[2,3]` 张量的 stride 为 `(3,1)`：

```text
对应存储偏移：0 → 1 → 2 → 3 → 4 → 5
```

“行首到下一行行首相差 3”和“当前行末到下一行首相差 1”描述的是不同起点。

**连续性看的是按逻辑顺序遍历时是否紧密排列，不是只看行首之间有没有位移。**

### 4. 非连续不等于不能计算

“只要跳过去就能读”是对的：支持 stride 的算子可以正确读取这类张量。问题在于：

- 有些算子要求特定布局，需要先拷贝整理。
- 有些算子支持非连续布局，但执行成本不同。
- 本次编译报错发生在 Python 对连续性作判断时，不是仅凭“非连续”三个字就能断定算子不能算。

补充边界：空维度和大小为 1 的维度有特殊规则。例如 `[1,F]` 的行 stride 即使是 `2F`，也可能被判定为连续，因为根本没有第二行需要跨越。

## 四、重点三：运行时知道 R，为什么编译时还“不知道”？

### 反复提出的问题

- 拼接都已经完成了，shape 中的 R 不就知道了吗？
- 二维已经确定，为什么连续性不能直接是 True/False？
- 什么叫“数据依赖的符号形状”？
- 为什么基线没问题，融合后才暴露？

### 1. 区分三个概念

| 概念 | 示例 |
| --- | --- |
| 张量维数 `ndim` | `[R,F]` 是二维张量 |
| 某一维的长度 | R 可能是 0、1、2、100…… |
| 某一维的索引 | 第 0 维是行，第 1 维是列 |

“二维已经确定”并不代表“行数 R 已知”。R=0 是二维空张量，不是零维标量。

### 2. 编译可能处理“占位张量”

真实执行时，设备确实会产生一个具体 R。但是 AOTAutograd 等构图阶段可能使用 FakeTensor，只传播 shape、stride、dtype 等信息，不实际执行路由。

例如：

```text
真实一次运行：shape=[17,64]
编译时表示：  shape=[u0,64]
```

`u0` 表示一个符号整数。若它来自 token 分发等运行时数据，并且没有可用的具体值提示，就属于本次讨论的 unbacked symbolic size。

`cat` 只确定两路最后一维从 F 变成 2F；不会把原本未知的 R 自动变成常量。

### 3. 连续性为什么可能依赖 R？

假设 shape 为 `[R,64]`，stride 为 `(128,1)`：

| R 的值 | 普通连续性判断 |
| --- | --- |
| R=0 | 空张量，可能按规则视为连续 |
| R=1 | 只有一行，行 stride 不影响连续性 |
| R>1 | 行间有间隙，非连续 |

所以，在编译器不能证明 R 属于哪一类时，`is_contiguous()` 不一定能转为确定的 Python 布尔值。

本会话中的回归验证曾在恢复旧判断后出现：

```text
GuardOnDataDependentSymNode
Eq(u0, 1) | Eq(64*u0, 0)
```

表达式对应“单行或空张量”等边界。编译器不能随意假设这些条件为真或假。

### 4. 为什么融合后暴露，基线没有？

专家 token 数不确定是已有背景条件。融合路径新增的激活拼接及其反向切片，让梯度呈现了不同布局；“符号行数 + 带间隙梯度 + Python 连续性分支”组合在一起，才触发了问题。

基线没有报错，不能单独证明它的 R 静态，也不能证明整个问题必然是 CANN 或 PTA 算错。

本次处理是去掉量化分支中的 `or tensor.is_contiguous()`，用户反馈可跑通；CPU 符号布局回归也支持这一定位。它并不等于已经证明所有 NPU 数值和所有动态 shape 都正确。

## 五、重点四：为什么是 AOTAutograd / 反向阶段报错？

### 1. 前向布局和反向布局可以不同

前向的 gate、up 可以各自是紧凑张量，cat 后 packed 也可以连续；反向经过切片，传给两路矩阵乘的梯度却可能是带间隙的视图。

所以前向未遇到同样的输入布局，并不奇怪。

### 2. “追踪反向图”不等于“已经在设备上执行反向算子”

简化分工：

```text
TorchDynamo：从 Python 执行中捕获图
AOTAutograd：追踪、整理并划分前向与反向图
Inductor 等后端：进一步优化图并生成执行代码
```

本次符号连续性判断可以在追踪反向图时失败，实际反向 kernel 还没运行。

也不能仅凭栈里出现 AOTAutograd 就断言失败一定发生在用户显式调用 `loss.backward()` 后；联合构图也可能在首次前向调用期间触发。

### 3. eager、aot_eager、inductor

| 名称 | 应如何理解 |
| --- | --- |
| 普通 eager | 直接按 Python 调用执行算子，未通过这里讨论的 compile 包装 |
| `torch.compile(..., backend="eager")` | 仍经过 Dynamo 捕图，再以 eager 方式执行图 |
| `torch.compile(..., backend="aot_eager")` | 还经过 AOTAutograd，之后以 eager 方式执行图 |
| `torch.compile(..., backend="inductor")` | 使用 Inductor 后端编译图 |

因此，后端名字里带 eager，不等于完全没有 `torch.compile`。

## 六、cat 与 slice 写入：实现有什么不同？

比较过两种写法：

```python
# cat
gate = grouped_mm(...)
up = grouped_mm(...)
packed = torch.cat((gate, up), dim=-1)

# 预分配 + slice 写入
packed = x.new_empty((R, 2 * F))
packed[:, :F] = grouped_mm(...)
packed[:, F:] = grouped_mm(...)
```

第二种写法仍然是先得到矩阵乘结果，再将结果拷贝到 slice；不是让矩阵乘直接写进最终 buffer。直接写入需要算子支持相应输出接口和布局，并兼容量化、autograd 等路径。

两者的分配、拷贝次数、峰值存活张量及反向图可能不同；slice 原地赋值还可能引入额外的梯度处理。不能只根据源码少了一个 `cat` 就推断更快。

**本会话的实际结论：用户测得 slice 方案劣化，已恢复 cat。** `is_contiguous()` 问题已另行处理，不再作为这两种写法优劣的依据。

## 七、量化模块继承、Config 和 override

### 1. 为什么不用单独导入父类的 forward？

```python
class QuantizedExperts(AscGroupedExperts):
    def __init__(self, config):
        super().__init__(config)
        quantize_(self, ...)
```

子类没有定义 `forward` 时，Python 会沿继承关系使用 `AscGroupedExperts.forward`。导入父类后，其方法已经通过继承可访问，不需要再单独“导入 forward”。

`super().__init__()` 负责调用父类初始化，不是负责“复制 forward”。`quantize_()` 在这里安装量化参数包装，前向仍由继承的方法执行。

### 2. 重复转换判断的作用

```python
if (
    isinstance(config, NpuQuantizedAscGroupedExpertsModule.Config)
    or parent_cls in _npu_quantized_module_cache.values()
):
    continue
```

两部分分别识别固定的融合量化专家类和工厂动态生成的量化类，避免重复包装。

注意：跳过重复对象不意味着整个 converter 在任何配置下都允许“零项转换”；`require_match=True` 时仍可能因为没有新匹配而报错。测试重复转换时需正确设置该选项。

### 3. 为什么配置了 override，转换器还会看到 GroupedExperts？

本次核对的 `TrainerEx` 顺序是：

```text
初始 GroupedExperts.Config
    ↓ 量化配置转换
NpuQuantizedAscGroupedExpertsModule.Config
    ↓ apply_overrides：识别为 Asc 配置子类，保留
    ↓ build()
真正的量化专家实例
```

`override.imports` 指定要应用的工厂，不代表写进参数时就已经替换了模型。

## 八、`_split_k_x()`：不是改变矩阵数值，而是整理布局

该函数服务于 grouped matmul 的 2D×2D 权重梯度路径。split-K 在这里沿归约维的分组边界计算不同专家的权重梯度。

```python
if x.stride(-2) == 1 and x.stride(-1) == x.size(-2):
    return x
return x.transpose(-1, -2).contiguous().transpose(-1, -2)
```

以普通连续 `[2,3]` 为例：

| 步骤 | shape | stride |
| --- | --- | --- |
| 原始 | `[2,3]` | `(3,1)` |
| 第一次转置 | `[3,2]` | `(1,3)` |
| 连续化拷贝 | `[3,2]` | `(2,1)` |
| 再次转置 | `[2,3]` | `(1,2)` |

最终形状和数值不变，存储布局变为目标转置布局；它通常不属于普通行连续张量。

**尚未完全验证的边界：** singleton/空维度下 `.contiguous()` 可能不拷贝，所以当前实现不保证所有形状都得到精确 `(1,M)`。会话只确认了本地 CANN 源码中的严格 stride 检查，尚未在用户实际版本完成硬件复现，不应记成“已修复”。

## 九、iteration、micro-batch、optimizer step 和 profiler step

这是另一组专门询问含义的术语。

假设梯度累积次数为 4：

```text
Optimizer.zero_grad()
  micro-batch 1：FWD + BWD
  micro-batch 2：FWD + BWD
  micro-batch 3：FWD + BWD
  micro-batch 4：FWD + BWD
Optimizer.step()
ProfilerStep（若代码在此调用 profiler.step()）
```

| 术语                      | 含义                                                |
| ----------------------- | ------------------------------------------------- |
| FWD                     | 前向计算                                              |
| BWD                     | 反向传播，计算梯度                                         |
| micro-batch             | 一次前向/反向处理的小批次；多个小批次可累积梯度                          |
| 梯度累积                    | 多次 BWD 累积到参数梯度，再执行一次参数更新                          |
| `Optimizer.zero_grad()` | 清除上次梯度，具体可以置零或设为 None                             |
| `Optimizer.step()`      | 使用当前梯度和优化器状态更新参数                                  |
| optimizer step          | 一次参数更新对应的训练进度单位                                   |
| iteration               | 泛称“一轮循环”，必须看具体循环边界，不能固定等同于 micro-batch            |
| `ProfilerStep`          | profiler 根据代码中的 `profiler.step()` 划出的采集边界         |
| 反向重计算                   | activation checkpointing 为节省激活存储，在反向过程中重新执行部分前向计算 |

本次核对的 Trainer 中 `profiler.step()` 位于外层训练 step；不能把其他日志或自定义标记里的 iteration 一律理解成同一粒度。

## 十、如何从很长的错误栈定位问题？

用户特别问过：是否因为文件在本地 torchtitan-npu 目录，而其他文件在 `/usr/local`，才锁定它？

答案是：**路径可以帮助判断代码归属，真正依据是异常类型、抛错位置和调用链。** 框架或安装依赖中的问题同样可能是根因。

推荐阅读顺序：

1. 找最末端真正的异常：例如 `GuardOnDataDependentSymNode` 或主动抛出的 `RuntimeError`。
2. 找触发它的具体语句：例如 `tensor.is_contiguous()` 或优化器 `state_dict()`。
3. 往上追业务调用者：是谁把这个 tensor、配置或状态对象传进来？
4. 确定阶段：前向、反向构图、运行时 kernel、checkpoint 保存还是加载。
5. 核对 shape/stride、配置、版本和执行顺序，提出能被验证的解释。
6. 通过最小改动、独立预期或旧代码对照验证；不能把“不再报错”直接当作数值与性能都正确。

分布式日志中的 `Root Cause (first observed failure)` 是首先观察到的 worker 失败记录。`ChildFailedError`、其他 rank 退出及通信异常可能是连带结果，不一定各自是根因。

## 十二、重点五：保存权重、加载权重和完整断点续训

这部分发生了多次追问和两次实际报错，必须区分清楚。

### 1. 参数含义速查

| 参数 | 含义 |
| --- | --- |
| `checkpoint.enable` | 启用 checkpoint 功能 |
| `checkpoint.load-only` | 允许加载，禁止保存；仍可训练 |
| `checkpoint.no-load-only` | 允许按策略保存，也可以加载 |
| `checkpoint.last-save-model-only` | 最后一步只保存模型状态 |
| `checkpoint.no-last-save-model-only` | 最后一步保存完整训练状态 |
| `checkpoint.interval` | 中间 checkpoint 保存间隔；不等同于最后一步的保存策略 |
| `checkpoint.load-step 20` | 从 checkpoint 根目录中的 step-20 按恢复逻辑加载 |
| `checkpoint.initial-load-path` | 无可自动恢复 checkpoint 时，指定初始加载目录 |
| `checkpoint.initial-load-model-only` | 初始加载只取模型，不恢复优化器等训练进度 |
| `training.steps` | 当前任务累计 step 的停止值；是否从 0 开始取决于是否恢复进度 |
| `lr-scheduler.total-steps` | 学习率调度计划的总长度，不决定训练何时停止 |

### 2. 本次需求对应的正确命令模板

前提：在实际仓库根目录执行；已有数据、tokenizer 和运行环境配置可用。以下命令按会话核对的版本整理，未在本地完成 NPU 全流程验证。

第一次跑 20 step，保存最后一步模型。输出根目录应是新目录；若需要从预训练权重开始，还需另设初始加载配置。

```bash
export CKPT_SAVE_LOAD_PATH="$PWD/outputs/dsv4_resume_test"

bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh \
    --training.steps 20 \
    --lr-scheduler.total-steps 50 \
    --checkpoint.enable \
    --checkpoint.no-load-only \
    --checkpoint.interval 1000 \
    --checkpoint.last-save-model-only \
    --debug.seed 42 \
    --debug.deterministic
```

`interval=1000` 避免前 20 步中途保存完整状态；正常到达最后一步仍按最后一步策略保存。仅保存模型时还可能按 `export_dtype` 转换导出 dtype，它与完整训练状态保存不同。

第二次从该权重开始新任务，再训练 50 step，不保存：

```bash
# 这个目录必须没有可自动恢复的 checkpoint。
export CKPT_SAVE_LOAD_PATH="$PWD/outputs/dsv4_weights_followup"

bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh \
    --training.steps 50 \
    --lr-scheduler.total-steps 50 \
    --checkpoint.enable \
    --checkpoint.load-only \
    --checkpoint.initial-load-path "$PWD/outputs/dsv4_resume_test/step-20" \
    --checkpoint.initial-load-model-only \
    --debug.seed 42 \
    --debug.deterministic
```

要点：**新输出根目录、旧权重的具体 step-20 目录、初始只加载模型、训练 50 步。** 若融合算子不支持确定性模式，相关报错需要单独处理；这组 checkpoint 参数不解决算子确定性问题。

Shell 写法也要区分：

```bash
# 正确：两条命令
export CKPT_SAVE_LOAD_PATH="..."
bash script.sh ...

# 正确：单次命令的环境变量
CKPT_SAVE_LOAD_PATH="..." bash script.sh ...

# 不应把它们直接连成：export CKPT_SAVE_LOAD_PATH="..." bash script.sh ...
```

## 十四、复习时最值得记住的十句话

1. **cat 的是 gate/up 激活，不是 W1/W3 权重。**
2. **shape 是逻辑大小，stride 是存储寻址步长。**
3. **可以跳着读，不代表普通连续；非连续也不代表不能算。**
4. **张量是二维，不代表它的行数 R 在编译时已知。**
5. **R=0、R=1 的特殊规则会让连续性判断依赖符号值。**
6. **反向构图失败和设备反向 kernel 失败是两个阶段。**
7. **iteration 和 ProfilerStep 的含义必须看循环及标记位置。**
8. **配置了 override 不等于它已经执行；本次量化转换在它之前。**
9. **load-only 是不保存，不是不训练，也不等于只加载模型。**
10. **仅加载权重会重置训练进度；旧输出目录的自动恢复可能覆盖 initial-load 配置。**
