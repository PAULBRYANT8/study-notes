---
title: pip requirements 文件格式速查
type: concept
created: 2026-08-12
updated: 2026-08-12
tags: [pip, Python, requirements, packaging]
sources:
  - https://pip.pypa.io/en/stable/reference/requirements-file-format/
  - "[[raw/2026-08-12-pip-requirements-file-format]]"
---

pip requirements 文件是供 `pip install` 读取的“待安装项目或安装参数”清单；`requirements.txt` 只是常见文件名，不是格式要求。

本文把 [pip 官方 Requirements File Format 页面](https://pip.pypa.io/en/stable/reference/requirements-file-format/)（页面标示 v26.2.1，访问日期 2026-08-12）整理成查阅版。**官方事实**来自该页面；**实践建议**只表示编辑和审查时的工作习惯。原始事实摘录见 [[raw/2026-08-12-pip-requirements-file-format]]。

> **边界（官方事实）**：基本格式相对稳定，但完整语法和 pip 的内部细节、命令行选项紧密相关，主要供 pip 自身消费；其他工具不一定完整兼容。

## 30 秒速查

| 目的 | 写法 | 备注 |
|---|---|---|
| 按名称安装 | `requests` | 普通 requirement |
| 指定版本 | `docopt == 0.6.1` | 页面示例 |
| 版本约束 | `pkg>=1.2,<2` | 具体说明符规则见 Requirement Specifiers |
| 环境标记 | `requests >= 2.8.1 ; python_version < "2.7"` | 分号后写 marker |
| 命名直接 URL | `urllib3 @ https://host/pkg.zip` | 项目名与 URL 用 `@` 连接 |
| 远程归档 | `https://host/pkg.whl` | wheel 或其他归档 URL |
| 本地归档 | `./downloads/pkg.whl` | 本地 archive 路径 |
| 本地项目 | `./project` | 项目路径 |
| editable 项目 | `-e ./project` | `-e` 即 `--editable` |
| 引用其他 requirements | `-r other-requirements.txt` | 继续读取另一个文件 |
| 引用 constraints | `-c constraints.txt` | 传入 constraints 文件 |
| 全局安装选项 | `--no-index` | 独占一行 |
| 单项选项 | `--hash` / `--config-settings` | 页面列为单项选项；具体写法见当前 pip 参考 |
| 环境变量 | `https://${HOST}/simple` | 只支持 `${NAME}` |
| 注释 | `# comment` / `pkg  # comment` | 续行处理后剥离 |

## 每行可以写什么

官方页面列出五种基本形式：

```text
[[--option]...]
<requirement specifier>
<archive url/path>
[-e] <local project path>
[-e] <vcs project url>
```

尖括号是占位记号，不要原样复制。每行表示一个待安装项目，或一个传给 `pip install` 的参数。

### 1. Requirement specifier

可以写纯名称，也可以附带版本说明符、extras、环境标记等；页面示例包括：

```text
pytest
pytest-cov
beautifulsoup4
docopt == 0.6.1
requests [security] >= 2.8.1, == 2.8.* ; python_version < "2.7"
```

这篇笔记只负责 requirements 文件中的位置和写法；完整的版本说明符语法应查 pip 的 [Requirement Specifiers](https://pip.pypa.io/en/stable/reference/requirement-specifiers/) 页面。

### 2. URL、归档和路径

命名直接 URL：

```text
urllib3 @ https://github.com/urllib3/urllib3/archive/refs/tags/1.26.8.zip
```

直接写归档 URL 或本地路径：

```text
https://example.org/packages/pkg-1.0.0-py3-none-any.whl
./downloads/pkg-1.9.2-cp311-none-manylinux_x86_64.whl
```

### 3. 本地项目和 VCS 项目

```text
./project
-e ./project
git+https://example.org/team/project.git
-e git+https://example.org/team/project.git
```

这里的 `-e` 表示 editable。**实践建议：**VCS 项目的具体协议和可用写法应以当前 pip 版本的 VCS 支持文档为准；本页只列出这种行结构。

## 文件引用：`-r` 与 `-c`

```text
-r other-requirements.txt
-c constraints.txt
```

- `-r` / `--requirement`：引用另一个 requirements 文件。
- `-c` / `--constraint`：引用 constraints 文件。

**实践建议：**把“要安装的项目”与“额外传给 pip 的约束”分开放，文件名和相对路径保持清楚；不要让文件之间循环引用。

## 编码、续行和注释

### 编码

默认编码是 UTF-8。若需要其他编码，可用 PEP 263 风格注释声明，例如：

```text
# -*- coding: latin-1 -*-
```

### 续行

以未转义反斜杠 `\\` 结尾的行会续接下一行，紧随其后的换行符等效于被忽略：

```text
SomeProject==1.2.3 \
  --config-settings=build-option=fast
```

**实践建议：**让反斜杠成为物理行最后一个字符，不要在其后追加空格或注释。

### 注释

- 以 `#` 开头的行是注释。
- 空白后出现 `#` 时，`#` 及其余内容也视为注释。
- pip 先处理续行，再剥离注释。

因此，解释性注释最好独占一行，避免和续行混在一起。

## 支持的 pip 选项

requirements 文件只支持 pip install 的一部分选项；完整清单以官方页面为准。

### 全局选项

全局选项影响整个 `pip install` 运行，必须各自写在独立行：

| 选项 | 页面列出的名称 |
|---|---|
| 包索引 | `-i`, `--index-url` |
| 附加索引 | `--extra-index-url` |
| 禁用索引 | `--no-index` |
| 引用约束 | `-c`, `--constraint` |
| 引用 requirements | `-r`, `--requirement` |
| editable | `-e`, `--editable` |
| 查找归档 | `-f`, `--find-links` |
| 禁止二进制 | `--no-binary` |
| 仅二进制 | `--only-binary` |
| 偏好二进制 | `--prefer-binary` |
| 要求 hash | `--require-hashes` |
| 不要求 hash | `--no-require-hashes` |
| 预发布版本 | `--pre` |
| 所有发行版 | `--all-releases` |
| 仅正式版 | `--only-final` |
| 信任主机 | `--trusted-host` |
| 实验特性 | `--use-feature` |

示例：

```text
--pre
--no-index
--find-links /my/local/archives
--find-links https://archives.example.org/
```

### 单项选项

页面列出的 per-requirement 选项是：

- `--config-settings`
- `--hash`（用于 hash-checking mode）

它们可以附着在单个 requirement 上，通常配合反斜杠续行：

```text
SomeProject==1.2.3 \
  --hash=sha256:<digest>
```

## 环境变量

pip 支持在 requirements 文件中使用环境变量；变量名必须使用带大括号的大写 POSIX 形式：

```text
private-pkg @ https://${PKG_HOST}/private-pkg.whl
--extra-index-url https://${PIP_USER}:${PIP_TOKEN}@packages.example.org/simple
```

pip 在运行时从主机环境查找变量。以下形式不受支持：

```text
$NAME
%NAME%
```

**实践建议：**token、密码等敏感值只放在运行环境的 secret/environment 中，不要把实际值提交到 requirements 文件。

## 可复制的最小模板

```text
# 全局选项：各自独占一行
--index-url https://pypi.org/simple
# --extra-index-url https://packages.example.org/simple

# 普通需求和版本约束
requests
docopt == 0.6.1
requests [security] >= 2.8.1, == 2.8.* ; python_version < "2.7"

# 命名直接 URL
urllib3 @ https://example.org/urllib3-1.26.8.zip

# 本地归档和本地项目
./downloads/pkg-1.0.0-py3-none-any.whl
# -e ./project

# 组合文件
-r base-requirements.txt
-c constraints.txt

# 运行时环境变量（仅支持 ${NAME}）
private-pkg @ https://${PKG_HOST}/private-pkg.whl
```

## 查阅前检查清单

- [ ] 文件名不必叫 `requirements.txt`，但团队应统一命名。
- [ ] 每个非注释行都能归入页面列出的五种结构之一。
- [ ] 全局选项各占一行；单项选项明确附着在哪个 requirement 上。
- [ ] 续行反斜杠是物理行最后一个字符。
- [ ] 需要非 UTF-8 时写 PEP 263 编码声明，否则统一使用 UTF-8。
- [ ] 环境变量使用 `${NAME}`，没有误写成 `$NAME` 或 `%NAME%`。
- [ ] `-r` 与 `-c` 的路径或 URL 已按当前 pip 行为验证，且没有循环引用（实践检查）。
- [ ] 需要跨工具消费时，先确认目标工具是否支持 pip 的完整语法。
- [ ] 含真实 token、密码或私有地址的行不直接提交。

## 相关笔记

- [[python-advanced-mechanisms-for-pytorch]] — Python import、扩展加载和环境依赖排查。
- [[npu-training-adaptation-learning-path]] — 把 Python/环境基础放进 NPU 适配学习路线。
- [[raw/2026-08-12-pip-requirements-file-format]] — 官方事实摘录。

## 来源

- [pip Requirements File Format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [[raw/2026-08-12-pip-requirements-file-format]]
