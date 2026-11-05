---
title: Linux 常用命令与工作入门
type: concept
created: 2026-11-04
updated: 2026-11-04
tags: [linux, bash, shell, 命令行, 入门]
sources:
  - https://www.gnu.org/software/bash/manual/bash.html
  - https://www.gnu.org/software/coreutils/manual/
  - https://www.gnu.org/software/findutils/manual/html_mono/find.html
  - https://www.gnu.org/software/grep/manual/grep.html
  - https://www.gnu.org/software/tar/manual/tar.html
  - https://man.openbsd.org/ssh
  - https://man.openbsd.org/scp
  - https://www.freedesktop.org/software/systemd/man/latest/systemctl.html
---

Linux 命令行通过 Shell 接收命令：先确认当前位置和操作对象，再查看、修改文件或运行程序，遇到不熟悉的选项就查手册。

这份笔记面向开始使用 Linux 工作的人，示例以 **Bash 和常见 GNU/Linux 工具**为准；不同发行版、Shell 或工具版本的选项可能不同。`#` 后面的文字是说明，不用输入。`<用户名>`、`<服务器>`、`<PID>` 等是需要替换的占位符，不要连尖括号一起输入。

## 1. 打开终端：先认识命令

```bash
pwd                 # 当前所在目录
whoami              # 当前用户名
hostname            # 当前主机名
uname -a            # 内核及机器信息
cat /etc/os-release # 查看发行版信息
```

命令一般由“命令名、选项、参数”组成，例如 `ls -l /tmp` 中，`ls` 是命令，`-l` 是选项，`/tmp` 是要查看的路径。选项的含义由具体命令决定，同一个 `-f` 不一定总是同一含义。

```bash
man ls             # 阅读手册；按 q 退出
ls --help          # 查看简要帮助
command -v python3 # 查找命令；没有输出通常表示当前 PATH 中找不到
```

常用快捷键：**Tab** 补全，**↑ / ↓** 浏览历史，**Ctrl+C** 请求中断前台命令，**Ctrl+L** 清屏，**Ctrl+D** 在空命令行上结束当前 Shell 输入。交互式程序对快捷键可能有自己的处理方式。

## 2. 路径与目录

`/` 是根目录；`~` 是当前用户的家目录；`.` 是当前目录；`..` 是上一级目录。以 `/` 开头的是绝对路径，其他路径一般相对于当前目录。

```bash
pwd                 # 我在哪里
ls                  # 列出当前目录
ls -lah             # 包括隐藏文件，显示详细信息和易读大小
cd /tmp             # 进入绝对路径
cd ~/work           # 进入家目录下的 work
cd ..               # 回到上一级
cd                  # 回到家目录
cd -                # 返回上一次所在目录
```

Linux 文件名通常区分大小写。带空格的路径用引号包起来，例如 `cd "my project"`。以 `.` 开头的文件通常由 `ls -a` 显示。

## 3. 创建、复制、移动和删除

先在家目录建立练习区，下面的相对路径示例都在这里执行：

```bash
mkdir -p ~/linux-practice
cd ~/linux-practice
mkdir -p project/logs          # 创建目录，必要时连同父目录一起创建
touch project/notes.txt        # 文件不存在时创建；存在时更新时间戳
cp -i project/notes.txt project/notes.bak  # 复制；覆盖前询问
cp -ri project project-copy    # 递归复制目录；覆盖前询问
mv -i project/notes.bak project/notes-old.txt # 移动或重命名；覆盖前询问
```

删除通常不经过桌面回收站，先用 `pwd` 和 `ls` 确认路径：

```bash
rm -i project/notes-old.txt    # 删除文件前询问
rmdir project/logs             # 只删除空目录
rm -ri project-copy            # 递归删除并逐项询问
```

`rm -r` 会递归删除目录内容，`-f` 会跳过大部分提示；不要对不确定的路径使用 `rm -rf`。通配符如 `*.log` 由 Shell 展开，删除前先用 `ls -- *.log` 检查目标。文件名以 `-` 开头时可用 `rm -- -filename` 避免被当成选项。复制、移动也可能覆盖目标；`-i` 会询问，但脚本和不同实现的行为仍应先查手册。

## 4. 查看、编辑和查找文本

```bash
cat project/notes.txt             # 输出短文件全部内容
less project/notes.txt            # 分页浏览；/关键词 搜索，n 下一个，q 退出
head -n 20 project/notes.txt      # 前 20 行
tail -n 20 project/notes.txt      # 最后 20 行
tail -f project/logs/train.log    # 持续跟踪追加的日志；Ctrl+C 结束
```

简单编辑可用 `nano project/notes.txt`：输入文字后按 `Ctrl+O` 保存、回车确认，`Ctrl+X` 退出。`nano` 不一定预装；也可以使用自己熟悉的编辑器。

```bash
grep -n 'error' project/notes.txt        # 查匹配行并显示行号
grep -in 'error' project/notes.txt       # 忽略大小写
grep -RIn 'TODO' project                 # 递归搜索目录内文本
find project -type f -name '*.txt'       # 按文件名查找普通文件
find project -type d -name 'logs'        # 查找目录
```

`grep` 默认把搜索词视为正则表达式；查找固定文本可用 `grep -F`。`find` 的 `*.txt` 加引号，是为了交给 `find` 匹配，而不是让当前 Shell 先展开。

## 5. 把命令组合起来

```bash
printf 'hello\nworld\n' > project/notes.txt  # 写入；如果文件存在则覆盖
printf 'another line\n' >> project/notes.txt # 追加
cat project/notes.txt | grep -n 'hello'      # 管道：前者标准输出交给后者
```

`>` 会覆盖目标文件，`>>` 会追加；确认路径后再执行。管道 `|` 默认传递标准输出，错误信息默认仍显示在终端。需要将标准输出和错误都记录到同一文件时：

```bash
python3 train.py > train.log 2>&1  # 覆盖日志；命令结束前可能仍在运行
```

`train.py` 是示例程序，执行前需替换成实际程序。`2>&1` 让标准错误指向此时标准输出所指的位置，重定向顺序有意义。更多语法见 [Bash 重定向](https://www.gnu.org/software/bash/manual/html_node/Redirections.html)。

```bash
echo "$HOME"              # 双引号中变量会展开
printf '%s\n' '$HOME'     # 单引号中的内容按字面输出
```

## 6. 权限与用户

`ls -l` 的权限字段如 `-rw-r--r--`：第一个字符表示文件类型，后面每三位分别表示所有者、所属组、其他用户的权限；`r` 读、`w` 写、`x` 执行。对目录，`x` 表示可以搜索或进入该目录，具体访问还取决于上级目录权限等。

```bash
id                        # 用户 ID 和所属组
ls -l project/notes.txt
chmod u+x script.sh       # 给文件所有者增加执行权限
chmod 644 project/notes.txt  # 所有者可读写，组和其他用户可读
```

`sudo` 以管理员等其他用户身份执行一条命令，权限由系统配置决定。操作系统文件、安装软件或改权限前先确认命令和目标；共享服务器遵守管理员的使用规则。不要为解决权限错误随意使用 `chmod -R 777`。

## 7. 进程、任务和资源

```bash
ps -u "$USER" -f       # 查看当前用户的进程
ps -ef                 # 查看系统进程
pgrep -af 'train.py'   # 按完整命令行搜索，并显示匹配结果
top                    # 实时查看进程；按 q 退出
```

找到并确认目标 PID 后，优先请求正常退出：

```bash
kill <PID>             # 默认发送 SIGTERM，请求进程退出
```

进程可能忽略或处理该信号；必要时先检查任务及其日志。`kill -9 <PID>` 发送无法被捕获的 SIGKILL，程序无法正常清理，仅在确认需要时使用，不要对不属于自己的任务操作。

```bash
jobs -l                # 查看当前 Shell 启动的后台任务
```

命令后加 `&` 可在当前 Shell 中放到后台；`Ctrl+Z` 通常暂停前台任务，`bg` 继续后台运行，`fg` 将任务带回前台。后台运行不保证断开 SSH 后任务继续：长期任务应使用服务器提供的任务调度器或 `tmux`，并确认本地配置。

```bash
uptime                 # 运行时间和平均负载
free -h                # 内存使用情况
df -h                  # 已挂载文件系统的空间
du -sh project         # 查看目录占用总量
```

`df` 查文件系统剩余空间，`du` 统计指定路径下的占用；结果可能因删除但仍被进程打开的文件等原因不同。

## 8. 网络与远程工作

```bash
ip addr                          # 查看网络接口与地址
ip route                         # 查看路由
ss -lnt                          # 查看本机监听的 TCP 端口
ssh <用户名>@<服务器>              # 连接远程机器
ssh -p 2222 <用户名>@<服务器>      # 指定 SSH 端口
```

首次连接时应通过可信渠道核对服务器主机密钥指纹，再决定是否接受。登录后先用 `hostname`、`whoami`、`pwd` 确认当前机器、用户和位置；`exit` 退出远程 Shell。

下面的 `scp` 命令**在本地终端**执行，示例中的路径和用户名需要替换：

```bash
scp local.txt <用户名>@<服务器>:~/   # 本地上传到远端家目录
scp <用户名>@<服务器>:~/remote.txt . # 下载到本地当前目录
scp -r local-dir <用户名>@<服务器>:~/ # 递归复制目录
scp -P 2222 local.txt <用户名>@<服务器>:~/ # scp 的端口选项是大写 P
```

确认目的文件名，避免覆盖；远端路径也可能受服务器权限或配置限制。若使用集群，登录节点和计算节点的用途由集群规定。

## 9. 压缩、环境变量和软件

```bash
tar -czf project.tar.gz project  # 创建 gzip 压缩包
tar -tzf project.tar.gz          # 先列出压缩包内容
tar -xzf project.tar.gz          # 解压到当前目录
```

解压前先检查文件列表、来源和目标目录，避免覆盖已有文件。`tar` 的选项和支持的压缩格式取决于实现。

```bash
echo "$SHELL"         # 登录 Shell 的配置值，不一定是当前运行的 Shell
echo "$PATH"          # 查找可执行文件的目录列表
export MY_WORK_MODE=dev  # 给当前 Shell 及后续子进程设置环境变量
python3 --version
```

`export` 的设置只影响当前 Shell 及其之后启动的子进程；新终端不会自动继承这次临时修改。不要在公开日志、截图或笔记中暴露令牌、密码等敏感环境变量。

软件安装取决于发行版和权限。Ubuntu / Debian 使用 APT，例如先 `apt search <包名>` 搜索，确认来源与包名后可用 `sudo apt install <包名>` 安装；其他发行版可能使用 `dnf` 等工具。在共享机器上先查管理员文档。Python 项目通常优先使用项目指定的虚拟环境及依赖文件，相关格式见 [[pip-requirements-file-format]]。

如果系统使用 systemd，可以只读检查服务和日志：

```bash
systemctl status <服务名>          # 查看服务状态；按 q 退出
journalctl -u <服务名> -n 50     # 查看该服务最近 50 行日志
```

服务名称、systemd 是否可用及日志权限因机器而异，尤其在容器中不一定适用。

## 10. 一次可以照做的练习

以下练习只在家目录的 `linux-practice` 中操作；如果该目录已有重要内容，先换一个新的目录名。练习中的 `>` 会覆盖 `demo.txt`。

```bash
mkdir -p ~/linux-practice/demo
cd ~/linux-practice/demo
pwd
printf 'hello linux\nerror: example\n' > demo.txt
cat demo.txt
grep -n 'error' demo.txt
cp -i demo.txt demo-copy.txt
ls -lh
mv -i demo-copy.txt demo-renamed.txt
find . -type f -name '*.txt'
rm -i demo-renamed.txt
```

完成后想一想：`pwd` 输出是什么？`grep` 找到了第几行？为什么 `rm` 会询问？回到其他目录后，`demo.txt` 的绝对路径是什么？

## 11. 按任务快速回查

- 找位置：`pwd`、`ls -lah`、`cd`。
- 操作文件：`mkdir`、`cp -i`、`mv -i`、`rm -i`。
- 查文件和日志：`less`、`tail -f`、`grep -n`、`find`。
- 看任务和资源：`ps`、`pgrep`、`top`、`free -h`、`df -h`、`du -sh`。
- 远程工作：`ssh` 登录，`scp` 传文件，先确认本地和远端路径。
- 程序崩溃、卡住或运行缓慢：继续阅读 [[linux-debugging-for-npu-adaptation]]。

常见问题优先查看本机 `man <命令>`；手册缺失时可查 [GNU Coreutils](https://www.gnu.org/software/coreutils/manual/)、[Bash 手册](https://www.gnu.org/software/bash/manual/bash.html)、[GNU Findutils](https://www.gnu.org/software/findutils/manual/html_mono/find.html) 和 [OpenSSH 手册](https://man.openbsd.org/ssh)。
