---
title: NPU 训练适配所需的 Linux 与调试基础
type: concept
created: 2026-07-26
updated: 2026-07-26
tags: [linux, 调试, gdb, pdb, perf, strace, core-dump, npu]
sources:
  - https://sourceware.org/gdb/current/onlinedocs/gdb/
  - https://man7.org/linux/man-pages/man5/core.5.html
  - https://docs.python.org/3/library/pdb.html
  - https://man7.org/linux/man-pages/man1/perf-top.1.html
  - https://man7.org/linux/man-pages/man1/strace.1.html
  - https://man7.org/linux/man-pages/man1/dmesg.1.html
---

**Linux 调试的核心不是记住命令，而是先判断故障发生在哪一层，再用代价最低的工具取得能证伪猜想的证据。**

NPU 训练适配横跨 Python、PyTorch C++、后端运行时、驱动和设备，因此同一个“训练挂了”可能分别表现为 Python 异常、进程 `SIGSEGV`、系统调用卡住、主机 CPU 热点、驱动报错或设备异步错误。工具的分工是：

| 现象 | 第一工具 | 回答的问题 |
|---|---|---|
| Python 抛异常或逻辑不对 | `pdb` | 哪一行、哪些 Python 对象和控制流导致异常 |
| 进程 `Segmentation fault (core dumped)` | `gdb` + core dump | 哪个原生线程、哪条 C/C++ 调用链崩了 |
| 进程启动失败、找不到文件、权限异常、疑似阻塞 | `strace` | 进程正在进行哪些系统调用，返回值是什么 |
| CPU 满、进程没崩但很慢 | `perf top` | CPU 时间主要消耗在哪些符号上 |
| 疑似 OOM killer、驱动、IOMMU、PCIe 或内核问题 | `dmesg` / `journalctl -k` | 内核在故障时记录了什么 |
| NPU 算子异步失败 | 后端日志 + 强制同步 + 上述工具 | 真正失败的算子是否早于 Python 报错位置 |

对应文档：[GDB 官方手册](https://sourceware.org/gdb/current/onlinedocs/gdb/) · [Python `pdb`](https://docs.python.org/3/library/pdb.html) · [`strace(1)`](https://man7.org/linux/man-pages/man1/strace.1.html) · [`perf-top(1)`](https://man7.org/linux/man-pages/man1/perf-top.1.html) · [`dmesg(1)`](https://man7.org/linux/man-pages/man1/dmesg.1.html)

---

## 一、先做故障分层

### 1. Python 异常不等于 Python 根因

看到 traceback 时先读最末行的异常类型和消息，再从最靠近业务代码的栈帧向下看。以下情况尤其可能来自原生层：

- 异常消息包含算子名、stream、runtime、driver、device、HCCL 等信息。
- 报错位置是 `.cpu()`、`.item()`、`print(tensor)` 或显式同步，而真正失败的设备算子更早。
- 进程没有 Python traceback，直接退出并显示 `Segmentation fault`、`Aborted` 或负的 return code。
- 多进程训练只有一个 rank 先报错，其他 rank 随后超时或被 launcher 终止。

设备执行通常是异步的：Python 线程完成“下发”不代表设备已经完成计算。排查时临时在可疑区间插入同步，把“晚报错”压缩成“就近报错”；同步会改变性能和时序，因此只能作为定位手段，不能留在性能路径中。

### 2. 常见退出形式

| 表现 | 常见含义 | 首要动作 |
|---|---|---|
| 正常 Python traceback | Python 或被转换成 Python 异常的后端错误 | 保存完整 traceback，进 `pdb` 或最小复现 |
| `Segmentation fault`，shell 状态常见为 139 | 通常是进程收到 `SIGSEGV` | 查 core dump，用 GDB |
| `Aborted`，shell 状态常见为 134 | 通常是 `SIGABRT`，可能由断言、运行库主动终止触发 | 查 stderr、core、GDB |
| `Killed`，无业务栈 | 可能是 OOM killer 或外部 `SIGKILL` | 立刻看 `dmesg` / cgroup 限额 |
| 一直不退出 | 死锁、通信不匹配、阻塞 I/O、设备调用未返回 | `strace -p`、GDB attach、通信日志 |
| 只有性能下降 | host bound、锁竞争、频繁系统调用、CPU 热点 | `perf top`，再用 profiler |

退出状态只能给方向，不能独立证明根因。信号与 core dump 的系统行为见 [`signal(7)`](https://man7.org/linux/man-pages/man7/signal.7.html) 和 [`core(5)`](https://man7.org/linux/man-pages/man5/core.5.html)。

### 3. 第一时间保存现场

在重跑、重启服务或清理日志前记录：

```bash
date --iso-8601=seconds
hostname
uname -a
id
pwd
ps -ef
ulimit -a
cat /proc/sys/kernel/core_pattern
```

训练任务还要保存：

- 完整启动命令、环境变量和配置文件。
- 所有 rank 的 stdout/stderr，而不是只保存 rank 0。
- 软件包、驱动、固件和后端版本。
- rank、主机、设备的映射。
- 最后一个成功 step 和第一个失败 step。
- 是否固定随机种子、能否稳定复现、单卡是否复现。

不要一开始就“多改几个开关看看”。一次改变多个变量会破坏因果证据。

---

## 二、GDB 与 core dump：原生崩溃的主线

### 1. core dump 是什么

core dump 是进程终止时的部分内存映像和寄存器状态，用于事后恢复“崩溃瞬间”。它通常不包含所有文件映射，分析时还需要崩溃时的可执行文件、动态库和调试符号。系统可能把 core 交给 `systemd-coredump`，所以当前目录里没有 `core` 文件不代表没有生成。

对应文档：[Linux `core(5)`](https://man7.org/linux/man-pages/man5/core.5.html) · [GDB 加载可执行文件与 core](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Files.html)

### 2. 让系统能够产生 core

先做只读检查：

```bash
ulimit -c
cat /proc/sys/kernel/core_pattern
cat /proc/self/coredump_filter
```

当前 shell 临时解除 core 大小限制：

```bash
ulimit -c unlimited
```

然后从同一个 shell 启动训练。`ulimit` 是 shell 及其子进程的属性，在另一个终端执行不会改变已经运行的训练进程。

若系统使用 systemd：

```bash
coredumpctl list
coredumpctl info <PID-or-executable>
coredumpctl debug <PID-or-executable>
```

若需导出：

```bash
coredumpctl dump <PID-or-executable> --output=core.dump
```

core 可能很大并包含输入、密钥、路径和业务数据，不要直接上传到公共 issue。core 的生成条件、`core_pattern`、`coredump_filter` 和 systemd 行为见 [`core(5)`](https://man7.org/linux/man-pages/man5/core.5.html)；`coredumpctl` 见 [`coredumpctl(1)`](https://www.freedesktop.org/software/systemd/man/latest/coredumpctl.html)。

### 3. 调试符号决定能看到多少

理想构建至少包含 `-g`；高优化构建中局部变量可能显示为 `<optimized out>`，函数也可能被内联。常用构建取舍：

```text
-g -O0              最容易单步，但行为和性能偏离发布构建
-g -Og              保留较好可调试性，同时做有限优化
-g -O2/-O3          最接近线上，但变量、栈和执行顺序更难解释
-fno-omit-frame-pointer  通常更利于采样和回溯
```

分析线上 core 时，必须尽量使用与崩溃进程**完全匹配**的二进制和 `.so`。同名但重新编译过的库可能产生错误符号或错误行号。GDB 对优化代码的限制见 [Debugging Optimized Code](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Optimized-Code.html)。

### 4. 打开 core 后的最小命令集

```bash
gdb /path/to/python /path/to/core
```

如果 core 来自某个独立 C++ 程序，把 `/path/to/python` 换成原可执行文件。进入 GDB 后：

```gdb
set pagination off
info files
info sharedlibrary
info threads
thread apply all bt
thread apply all bt full
```

`bt` 是调用栈，`bt full` 还尝试显示局部变量；多线程程序必须看所有线程，不能只看当前线程。对应文档：[GDB Backtrace](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Backtrace.html)。

定位可疑线程和栈帧：

```gdb
thread <thread-number>
frame 0
up
down
info args
info locals
p variable
p/x variable
ptype variable
```

查看寄存器、指令和内存：

```gdb
info registers
x/i $pc
disassemble /m
x/16gx <address>
```

常见观察：

- `$pc` 所在指令访问了哪个地址。
- 指针是 `0x0`、明显的毒值、已释放地址，还是合法但越界。
- 崩溃线程之外，其他线程是否都在等待同一把锁或同一次通信。
- 调用栈是否落在后端、驱动用户态库、PyTorch dispatcher、ATen kernel 或 Python 扩展边界。

### 5. 活进程 attach

对“卡住但未崩”的进程：

```bash
gdb -p <PID>
```

进入后先：

```gdb
set pagination off
thread apply all bt
```

若所有工作线程都等待相同 futex，可能是锁或条件变量等待；若一个线程停在设备运行时调用，其他线程都在 join，优先追那个调用。detach 后再让进程继续：

```gdb
detach
quit
```

attach 会暂停目标进程，线上使用前要确认影响。对应文档：[Debugging an Already-running Process](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Attach.html)。

### 6. Python 与 C++ 混合栈

PyTorch 进程的原生入口通常是 Python。普通 `bt` 能看到 C/C++ 栈；若安装了与 CPython 匹配的 GDB 辅助脚本，还可尝试：

```gdb
py-bt
py-list
py-locals
```

这些命令是否可用取决于发行版是否提供 Python debug helpers。不要因为 `py-bt` 不可用就认定 core 无法分析；原生 `thread apply all bt` 仍然是主证据。CPython 的 GDB 支持见 [Debugging C API extensions and CPython Internals with GDB](https://docs.python.org/3/howto/gdb_helpers.html)。

### 7. 一份可分享的 GDB 批处理输出

```bash
gdb -q -batch \
  -ex "set pagination off" \
  -ex "info sharedlibrary" \
  -ex "info threads" \
  -ex "thread apply all bt full" \
  /path/to/python /path/to/core > gdb-report.txt 2>&1
```

分享前检查输出中是否含路径、输入内容、访问令牌或其他敏感数据。

---

## 三、PDB：Python 控制流和对象状态

### 1. 三种最常用入口

从程序一开始运行：

```bash
python -m pdb train.py --config config.yaml
```

在代码中停下：

```python
breakpoint()
```

捕获异常后进入事后调试：

```python
import pdb

try:
    run_training()
except Exception:
    pdb.post_mortem()
```

Python 3.14 起还有远程 attach 能力，但生产环境是否允许取决于版本和安全策略；基础用法以 [`pdb` 官方文档](https://docs.python.org/3/library/pdb.html)为准。

### 2. 必会命令

| 命令 | 作用 |
|---|---|
| `l` / `ll` | 看当前附近代码 / 整个函数 |
| `n` | 执行当前行，停在同一函数下一行 |
| `s` | 进入函数 |
| `r` | 运行到当前函数返回 |
| `c` | 继续到下一个断点 |
| `u` / `d` | 调到上一层 / 下一层栈帧 |
| `w` / `bt` | 查看 Python 调用栈 |
| `p expr` / `pp expr` | 求值并打印表达式 |
| `a` | 查看当前函数参数 |
| `b file.py:123` | 设置断点 |
| `b 123, condition` | 条件断点 |
| `disable` / `enable` / `clear` | 管理断点 |
| `q` | 退出 |

### 3. 训练代码中优先检查什么

```python
p tensor.shape
p tensor.dtype
p tensor.device
p tensor.stride()
p tensor.is_contiguous()
p tensor.requires_grad
p tensor.grad_fn
p torch.isfinite(tensor).all()
p tensor.detach().float().abs().max()
```

对大 tensor 不要直接 `p tensor`：打印可能触发设备同步、D2H 传输并产生海量输出。优先打印 shape、dtype、device、有限性和小范围切片。

### 4. PDB 的边界

PDB 不能解释原生 `SIGSEGV`，也无法单步设备 kernel；Python 进程直接崩溃时应切到 core + GDB。PDB 还会改变线程时序，不能用“加了断点以后不复现”证明问题已消失。

---

## 四、strace：看清进程和内核之间发生了什么

`strace` 跟踪系统调用及其返回值，适合查文件、网络、进程、信号、权限和阻塞问题，不适合直接分析 Python 算法或设备 kernel 内部逻辑。完整选项见 [`strace(1)`](https://man7.org/linux/man-pages/man1/strace.1.html)。

### 1. 一套实用默认参数

```bash
strace -f -tt -T -s 256 -yy -o strace.log \
  python train.py
```

- `-f`：跟踪子进程和线程。
- `-tt`：打印高精度时间。
- `-T`：打印每个系统调用耗时。
- `-s 256`：扩大字符串截断长度。
- `-yy`：尽量解析文件描述符指向的资源。
- `-o`：写文件，避免与训练日志混在一起。

### 2. attach 到卡住的进程

```bash
strace -f -tt -T -p <PID>
```

典型模式：

- 长时间停在 `futex(...)`：线程在等锁或条件变量，但还需结合所有线程栈判断谁持锁。
- 长时间停在 `poll` / `epoll_wait`：可能正常等待 I/O，也可能上游永远不会产生事件。
- `connect(...)` 超时或拒绝：通信地址、路由、防火墙或服务状态问题。
- 反复 `openat(...)= -1 ENOENT`：可能是动态库/配置搜索，也可能只是正常探测。
- `read(...)` 长时间不返回：管道、socket、设备文件或数据输入阻塞。

### 3. 用过滤降低噪声

```bash
strace -f -e trace=%file -o files.log python train.py
strace -f -e trace=%network -o network.log python train.py
strace -f -e trace=%process,%signal -o process.log python train.py
strace -f -e trace=openat,read,write,ioctl,futex -o focused.log python train.py
```

查动态库搜索：

```bash
strace -f -e trace=openat,newfstatat,access \
  python -c "import torch_npu"
```

### 4. 不要误读

- `ENOENT` 经常是程序按候选路径搜索文件的正常过程，只有最终所需文件始终没找到才是问题。
- 一个慢系统调用不一定是根因，可能是它在等待设备或其他进程。
- `strace` 会带来显著开销，尤其 `-f` 跟踪多进程训练时；先缩小到单进程最小复现。
- 权限策略（如 Yama `ptrace_scope`）可能禁止 attach，不要绕过安全策略。

---

## 五、perf top：主机 CPU 热点的快速体检

`perf top` 实时采样 CPU 性能计数器并按符号聚合，回答“CPU 正忙在哪里”。它看到的是 host 侧，不会直接给出 NPU kernel 的设备耗时。对应文档：[`perf-top(1)`](https://man7.org/linux/man-pages/man1/perf-top.1.html)。

### 1. 常用方式

系统范围：

```bash
perf top
```

只看目标进程：

```bash
perf top -p <PID>
```

显示调用图：

```bash
perf top -p <PID> -g
```

按某个事件采样：

```bash
perf top -p <PID> -e cycles
```

先用 `perf list` 查看当前机器支持的事件。生产环境可能限制 `perf_event_paranoid`，应遵循系统权限策略。

### 2. 如何解释热点

| 热点位置 | 可能方向 |
|---|---|
| Python 解释器、对象分配、字典查找 | Python 控制流或小算子下发过密 |
| `memcpy` / `memmove` | host 数据搬运、布局转换或序列化 |
| 锁、futex 相关符号 | 锁竞争或线程调度 |
| 后端 runtime 的 launch / synchronize | 下发过密或频繁同步 |
| 数据解码、tokenizer、DataLoader | 输入流水线 host bound |
| 内核态网络栈 | 分布式通信或数据读取 |

“符号占比高”只说明采样期间 CPU 在那里，不等于优化它必然改善端到端性能。必须结合 wall-clock 分解和设备 timeline。

### 3. 何时从 `perf top` 升级

热点稳定后，用：

```bash
perf record -F 99 -g -p <PID> -- sleep 30
perf report
```

`perf top` 用于快速判断方向；`perf record/report` 用于保存可复查证据。采样数据接口的背景见 [Linux perf ring buffer 文档](https://www.kernel.org/doc/html/latest/userspace-api/perf_ring_buffer.html)，命令入口以本机 `man perf` 为准。

---

## 六、dmesg：内核、驱动与 OOM 的证据

`dmesg` 读取内核 ring buffer。它适合查内核和驱动事件，不是普通应用日志。对应文档：[`dmesg(1)`](https://man7.org/linux/man-pages/man1/dmesg.1.html)。

### 1. 常用查看方式

```bash
dmesg --human
dmesg --ctime
dmesg --level=emerg,alert,crit,err,warn
dmesg --follow
```

按关键词筛选：

```bash
dmesg --ctime | grep -Ei 'oom|killed process|segfault|npu|iommu|pcie|error|fault'
```

使用 systemd 的机器也可：

```bash
journalctl -k --since "10 minutes ago"
journalctl -k -p warning
```

### 2. 常见线索

- `Out of memory` / `Killed process`：主机内存或 cgroup OOM，不是设备显存 OOM。
- `segfault at ... ip ... in libxxx.so`：内核记录了崩溃地址和所在映射，可与 core/GDB 对照。
- IOMMU、PCIe、驱动 reset、设备掉线：进入驱动和硬件链路排查。
- 文件系统、网络接口错误：可能解释 checkpoint、数据或分布式异常。

### 3. 时间戳陷阱

`dmesg` 的原始时间通常是开机后的单调时间，不一定直接对应业务日志的墙钟时间；`--ctime` 的转换在休眠/恢复等场景也可能不精确。排障记录应同时保存训练日志时间、`date` 和内核日志上下文。

读取内核日志可能受 `dmesg_restrict` 限制。没有权限时请求运维提供指定时间窗日志，不要默认使用提权绕过。

---

## 七、适配岗的标准排障流程

### 阶段 1：复现并分类

1. 保存命令、环境、版本、全 rank 日志。
2. 判断是 Python 异常、原生崩溃、hang、主机 OOM、性能问题还是设备异步错误。
3. 缩成单卡、小 shape、少层数、少 step 的最小复现。
4. 每轮只改变一个变量。

### 阶段 2：按层取证

```text
Python traceback
    ↓ pdb / 插入有限的断言
Python ↔ C++ 边界
    ↓ core + GDB / 原生日志
系统调用与进程状态
    ↓ strace
host CPU
    ↓ perf top / perf record
内核与驱动
    ↓ dmesg / journalctl -k
设备执行
    ↓ 后端日志、强制同步、设备 profiler
```

### 阶段 3：形成可证伪假设

坏假设：“可能是 NPU 有问题。”

可检验假设：“第 37 个 `aten::index_put_` 下发后，下一次同步返回设备错误；相同输入在 CPU 正常；关闭该原地更新后错误消失。”

每个结论至少包含：

- 观察到了什么。
- 由哪个命令或日志得到。
- 排除了什么。
- 下一步实验如何区分剩余原因。

### 阶段 4：修复后验证

- 最小复现不再失败。
- 原始规模与多卡配置不再失败。
- 没有通过吞掉异常、关闭校验或永久同步来“修复”。
- 对精度、性能和资源占用做回归。
- 保留一份不含敏感信息的复现和排障记录。

---

## 八、常见误区

1. **只看 rank 0。** 分布式任务的第一现场经常在其他 rank，rank 0 只是随后超时。
2. **把最后一行当根因。** `.item()` 处报错可能来自更早的异步算子。
3. **没有匹配符号就分析 core。** 错版本二进制会制造误导性栈帧。
4. **用 `strace` 查设备 kernel 性能。** 它只看系统调用。
5. **把 `perf top` 百分比当端到端占比。** 它只描述采样 CPU 时间。
6. **在故障后很久才看 dmesg。** ring buffer 会覆盖，时间关联也会变差。
7. **直接打印大 tensor。** 这会同步设备并扰动现场。
8. **调试开关永久留在训练路径。** 同步、详细日志、sanitizer 都会改变性能甚至时序。

---

## 九、速查表

```bash
# core 是否开启
ulimit -c
cat /proc/sys/kernel/core_pattern

# systemd core
coredumpctl list
coredumpctl debug <PID>

# core 的所有线程栈
gdb -q -batch \
  -ex "set pagination off" \
  -ex "thread apply all bt full" \
  /path/to/executable /path/to/core

# Python 调试
python -m pdb train.py

# 卡住进程的系统调用
strace -f -tt -T -p <PID>

# 目标进程 CPU 热点
perf top -p <PID> -g

# 最近内核告警
dmesg --ctime --level=emerg,alert,crit,err,warn
journalctl -k --since "10 minutes ago"
```

---

## 十、自测与练习

### 自测

- 为什么 `.item()` 报设备错误时，`.item()` 不一定是根因？
- shell 显示 `Killed` 且没有 traceback，先查什么？
- core 文件存在但所有变量都是 `<optimized out>`，可能缺什么？
- `strace` 中大量 `ENOENT` 为什么不能直接判定为故障？
- `perf top` 显示 runtime launch 很热，如何证明端到端确实 host bound？
- 多卡任务 hang 时，为什么必须同时保存所有 rank 的日志和线程栈？

### 建议练习

1. 写一个空指针解引用的小 C++ 程序，开启 core，用 GDB 找到崩溃行、参数和寄存器。
2. 用 `pdb` 调试一个 shape 传播错误，练习条件断点和上下栈帧。
3. 用 `strace` 对一个 `import torch` 过程只过滤文件系统调用，找出 `.so` 搜索路径。
4. 对一个 CPU 忙循环运行 `perf top -p`，再用 `perf record/report` 保存报告。
5. 在内存受限的容器中复现一次 host OOM，比较应用日志和内核日志的差异。

---

## 相关

[[npu-training-adaptation-learning-path]] · [[cpp-reading-for-pytorch-backends]] · [[python-advanced-mechanisms-for-pytorch]] · [[deep-learning-training-numerics]] · [[floating-point-error-analysis]] · [[npu-precision-debugging]]
