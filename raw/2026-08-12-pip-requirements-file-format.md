---
title: pip Requirements File Format 官方资料摘录
type: raw
created: 2026-08-12
updated: 2026-08-12
tags: [pip, python, requirements-file, 包管理]
sources:
  - https://pip.pypa.io/en/stable/reference/requirements-file-format/
  - https://github.com/pypa/pip/blob/634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf/docs/html/reference/requirements-file-format.md
  - https://github.com/pypa/pip/releases/tag/26.2.1
---

# pip Requirements File Format 官方资料摘录

pip requirements 文件是在执行 `pip install` 时列出待安装项的清单；`requirements.txt` 只是这类文件的惯用名称，并非格式要求。

## 来源信息

- stable 文档：https://pip.pypa.io/en/stable/reference/requirements-file-format/
- v26.2.1 固定文档：https://github.com/pypa/pip/blob/634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf/docs/html/reference/requirements-file-format.md
- release/tag：https://github.com/pypa/pip/releases/tag/26.2.1（该 tag 指向 commit `634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf`）
- 访问日期：2026-08-12
- 页面版本：pip documentation v26.2.1

## 用途与兼容性边界

- requirements 文件的基本格式相对稳定且可移植。
- 完整语法与 pip 的内部细节和命令行选项紧密绑定，主要供 pip 消费；其他工具使用这一格式时需要考虑该边界。

## 每行支持的结构

requirements 文件的每个逻辑行表示一个待安装项或传给 `pip install` 的参数，支持以下形式：

- `[[--option]...]`，表示零个或多个受支持选项
- requirement specifier
- archive URL 或路径
- `[-e]` 本地项目路径
- `[-e]` VCS 项目 URL

## 编码、续行和注释

- 默认编码是 `UTF-8`；可以用 PEP 263 风格的编码注释指定其他编码。
- 行尾未转义的反斜杠 `\` 表示续行，其后的换行会被忽略。
- 以 `#` 开头的行会被当作注释忽略；空白后出现的 `#` 会使其自身及该行余下内容被视为注释。
- pip 先处理续行，再剥离注释。

## 支持的 pip install 选项

requirements 文件只支持官方页面列出的部分 `pip install` 选项。

### 全局选项

全局选项作用于整个 `pip install` 过程，并且必须各自单独占一行：

- `-i` / `--index-url`
- `--extra-index-url`
- `--no-index`
- `-c` / `--constraint`
- `-r` / `--requirement`
- `-e` / `--editable`
- `-f` / `--find-links`
- `--no-binary`
- `--only-binary`
- `--prefer-binary`
- `--require-hashes`
- `--no-require-hashes`
- `--pre`
- `--all-releases`
- `--only-final`
- `--trusted-host`
- `--use-feature`

### 单项 requirement 选项

可以应用于单个 requirement 的选项是：

- `--config-settings`
- `--hash`

## 引用其他文件

- `-r` / `--requirement` 用于引用另一个 requirements 文件。
- `-c` / `--constraint` 用于引用 constraints 文件。

## 环境变量

requirements 文件中的环境变量只支持 `${UPPERCASE_NAME}` 这种带花括号的大写名称 POSIX 格式。pip 会在运行时查找主机上的对应环境变量；不支持 `$NAME` 和 `%NAME%`。
