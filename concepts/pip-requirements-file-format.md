---
title: pip requirements 文件格式
type: concept
created: 2026-08-13
updated: 2026-08-13
tags: [pip, python, requirements-file, 依赖管理, 包管理]
sources:
  - https://pip.pypa.io/en/stable/reference/requirements-file-format/
  - https://github.com/pypa/pip/blob/634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf/docs/html/reference/requirements-file-format.md
  - https://github.com/pypa/pip/releases/tag/26.2.1
  - "[[raw/2026-08-12-pip-requirements-file-format]]"
---

# pip requirements 文件格式

pip requirements 文件是 `pip install` 的逐行输入清单，不是通用依赖标准，也不要求文件名必须是 `requirements.txt`。

它的基础格式相对稳定，但完整语法与 pip 的命令行选项和内部行为紧密相关，主要供 pip 使用；其他工具即使接受 requirements 文件，也不应默认支持 pip 的全部语法。

## 30 秒速查

| 官方支持的行结构 | 示例 | 定位 |
|---|---|---|
| pip 全局选项 | `--index-url https://pypi.org/simple` | 控制本次 `pip install`；必须单独占一个逻辑行 |
| requirement specifier | `requests>=2.32,<3` | 用名称、版本约束、extra 或环境标记描述待安装项 |
| 归档 URL 或路径 | `./dist/demo_package-1.0-py3-none-any.whl` | 指向 wheel、源码归档等安装产物 |
| editable 本地项目 | `-e ./local-project` | 以 editable 模式安装本地项目 |
| editable VCS URL | `-e "project @ git+https://github.com/example/project.git"` | 以现代 direct URL 写法 editable 安装版本库项目 |

`-r` 用来引用另一份 requirements 文件，`-c` 用来引用 constraints 文件；它们是文件组织指令，不是普通安装项。

## 1. 最常用写法

```text
# 不限定版本
requests

# 版本范围
requests>=2.32,<3

# 环境标记
importlib-metadata>=7 ; python_version < "3.10"

# 直接 URL
demo-package @ https://packages.example.invalid/files/demo_package-1.0-py3-none-any.whl

# 本地 wheel
./dist/demo_package-1.0-py3-none-any.whl

# editable 本地项目
-e ./local-project

# editable VCS direct URL
-e "project @ git+https://github.com/example/project.git"
```

这些地址和项目名用于展示语法；`example.invalid` 是占位域名，使用时应替换为实际制品地址。

## 2. 文件组织

```text
# 引用另一份 requirements 文件
-r requirements/base.txt
-r requirements/dev.txt

# 引用 constraints 文件
-c constraints/python311.txt
```

- `-r` / `--requirement`：把另一份 requirements 文件纳入输入。
- `-c` / `--constraint`：引用 constraints 文件。本笔记只说明这一引用关系，不展开 constraints 的独立语义。

相对路径需要特别检查可移植性。本笔记依据的官方摘录没有为所有路径类型统一声明解析基准，不应据此编造规则。

> **实践建议（不是格式要求）**：统一从项目根目录执行安装，并在 CI 中显式固定工作目录；同时在实际环境中验证嵌套 `-r`、`-c`、本地项目和本地产物路径。

## 3. 语法规则

### 编码

- **官方规则**：默认编码是 UTF-8。
- **官方规则**：可以用 PEP 263 风格的编码声明指定其他编码。

```text
# -*- coding: utf-8 -*-
requests
```

### 续行

物理行末尾未转义的反斜杠 `\` 会续接下一物理行；对应换行被忽略。

```text
requests>=2.32,\
<3
```

### 注释与处理顺序

```text
# 整行注释
requests>=2.32  # 空白后的行尾注释
```

- 以 `#` 开头的行会被忽略。
- 空白后出现的 `#` 会使该 `#` 及其后的行内内容成为注释。
- pip 先处理反斜杠续行，再剥离注释。
- 不能把规则简化成“遇到任何 `#` 都截断”；是否是注释取决于位置，URL 自身可能带有 fragment。

## 4. pip 选项

官方页面把下面一组列为 requirements 文件支持的全局选项。它们控制整个 `pip install` 输入过程，并且必须各自单独占一个逻辑行；不要把全局选项附在普通 requirement 行尾。`-r`、`-c` 和 `-e` 所在行分别表示文件引用、constraints 引用和 editable 安装目标。

| 全局选项 | 用途速记 |
|---|---|
| `-i` / `--index-url` | 设置主包索引 |
| `--extra-index-url` | 增加额外包索引 |
| `--no-index` | 不使用包索引 |
| `-c` / `--constraint` | 引用 constraints 文件 |
| `-r` / `--requirement` | 引用另一份 requirements 文件 |
| `-e` / `--editable` | 声明 editable 安装目标 |
| `-f` / `--find-links` | 增加查找安装文件的位置 |
| `--no-binary` | 禁止指定项目使用二进制包 |
| `--only-binary` | 要求指定项目使用二进制包 |
| `--prefer-binary` | 优先选择二进制包 |
| `--require-hashes` | 要求 hash 校验 |
| `--no-require-hashes` | 不要求 hash 校验 |
| `--pre` | 允许预发布版本 |
| `--all-releases` | 考虑所有 release |
| `--only-final` | 只考虑 final release |
| `--trusted-host` | 将指定主机标记为可信 |
| `--use-feature` | 启用指定功能 |

单项 requirement 选项只作用于它所在的那个 requirement，而不是整个安装过程：

| 单项选项 | 作用域 |
|---|---|
| `--config-settings` | 当前 requirement 的构建配置 |
| `--hash` | 当前 requirement 的文件 hash 校验值 |

## 5. 环境变量

requirements 文件只支持 `${UPPERCASE_NAME}` 这种带花括号的大写名称形式；`$NAME` 和 `%NAME%` 都不支持。变量值不是保存在文件中，而是由 pip 运行时从宿主环境查找。

下面是一个仅含占位信息的组合模板：

```text
# index 配置；PRIVATE_INDEX_TOKEN 由运行环境注入
--index-url https://pypi.org/simple
--extra-index-url https://token:${PRIVATE_INDEX_TOKEN}@packages.example.invalid/simple

# 拆分 requirements
-r requirements/base.txt
-r requirements/dev.txt

# 环境标记
importlib-metadata>=7 ; python_version < "3.10"

# 直接 URL
demo-package @ https://packages.example.invalid/files/demo_package-1.0-py3-none-any.whl
```

模板中的 `${PRIVATE_INDEX_TOKEN}` 不是实际凭据；不要把真实 token、密码或包含它们的展开结果提交到仓库。

> **实践建议（不是格式要求）**：凭据若含 URL 保留字符，应按索引服务要求先做 percent-encoding，并使用该服务要求的用户名形式。

## 6. 常见坑与检查清单

### 官方规则

- [ ] 文件是传给 `pip install` 的逐行输入，完整语法主要供 pip 使用，而不是跨工具通用的依赖标准。
- [ ] 文件名不限于 `requirements.txt`；命令通过 `-r <文件>` 指定输入文件。
- [ ] 全局选项单独占一个逻辑行，没有挂在普通 requirement 后面。
- [ ] `-r` 与 `-c` 被当作文件引用指令，而不是待安装的包。
- [ ] 非 UTF-8 文件带有合适的 PEP 263 编码声明。
- [ ] 续行先合并，注释后剥离；没有把 URL fragment 中的每个 `#` 都当作注释起点。
- [ ] 环境变量只写成 `${UPPERCASE_NAME}`，没有使用 `$NAME` 或 `%NAME%`。

### 实践建议（不是官方格式要求）

- [ ] 根据交付目标决定是否固定精确版本；格式本身不强制锁死版本。
- [ ] 需要完整性或可重复性保障时评估 `--hash` / `--require-hashes`；hash 不是每份 requirements 文件的必填项。
- [ ] 按 base、dev、test 等用途拆分文件时，保持引用关系简单并实际测试组合结果；拆分不是格式要求。
- [ ] 相对路径统一从项目根目录验证，CI 显式固定工作目录。
- [ ] 敏感值由 CI secret 或宿主环境注入，不把真实凭据写入 requirements；同时留意命令输出、错误信息和代理日志可能暴露展开后的 URL。

## 相关笔记

- [[raw/2026-08-12-pip-requirements-file-format]]

## 来源

- [pip Requirements File Format（stable）](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [pip v26.2.1 文档固定 commit](https://github.com/pypa/pip/blob/634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf/docs/html/reference/requirements-file-format.md)
- [pip 26.2.1 release/tag](https://github.com/pypa/pip/releases/tag/26.2.1)
