# 修复 DeepSeek-V4 HcHead 在张量并行 + torch.compile 下的编译崩溃

## 解决什么问题

TP 下开启 `torch.compile` 训练 DeepSeek-V4,会在编译最后一层的 `HcHead` 时崩溃:不带融合算子时报 `unsupported operand type(s) for *: 'DTensor' and 'DTensor'`,带融合且有动态序列维时报 `unhashable type: non-nested SymInt`。根因是 `HcHead` 在 TP 下本是"全 Replicate 的冗余计算",却让 DTensor 子类一路进入编译图,而 inductor/AOTAutograd 对这条 DTensor 路径支持不全;eager 有 dispatch 兜底所以正常,只有开 compile 才暴露。

## 怎么修改的

让 `HcHead.forward` 全程在普通张量上运算:把 `HcHeadParallelStyle` 的 `use_local_input` 由 `False` 改为 `True`(输入 all-gather 成 Replicate 后转成本地张量再传入),并在 forward 内对三个 Replicate 参数 `to_local()`;出口不变,仍由并行风格包回 `Shard(1)`。对 Replicate 张量而言 `to_local/from_local` 无损且无通信,因此数值、梯度、通信都与原来等价;参数在模块层面仍是 DTensor,FSDP 与 checkpoint 不受影响。

## 测试

新增 `deepseek_v4_tp_compile` smoke test(TP=2 + `--compile.enable`),在 2 die 上按 `.ci/smoke_test.sh` 方式验证:loss/grad_norm 正常、Training completed、不再 `BackendCompilerFailed`,约 48s 跑完。
