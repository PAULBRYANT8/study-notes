---
title: pip Requirements 文件格式事实摘录
type: raw
created: 2026-08-12
updated: 2026-08-12
tags: [pip, Python, requirements, packaging]
sources:
  - https://pip.pypa.io/en/stable/reference/requirements-file-format/
---

本文摘录 pip 官方《Requirements File Format》页面（文档版本 v26.2.1，访问日期 2026-08-12）的格式事实；完整语法主要供 pip 自身消费。

## 用途与命名

Requirements 文件是 `pip install` 使用的待安装项目清单，也可以在每行写安装参数。它们通常被称为 `requirements.txt` 文件，但文件名并非格式要求。

## 每行支持的五类结构

每行表示一个待安装项目或 `pip install` 参数，支持以下五种形式：

1. `[[--option]...]`：选项行。
2. `<requirement specifier>`：需求说明符。
3. `<archive url/path>`：归档 URL 或路径。
4. `[-e] <local project path>`：本地项目路径，可选 `-e`（editable）。
5. `[-e] <vcs project url>`：版本控制系统项目 URL，可选 `-e`。

## 编码、续行与注释

- 默认编码是 UTF-8；可用 PEP 263 风格注释（例如 `# -*- coding: <encoding name> -*-`）声明其他编码。
- 以未转义反斜杠 `\` 结尾的行会续接下一行，紧随其后的换行符等效于被忽略。
- `#` 开头的行是注释并被忽略；空白后出现的 `#` 及其余内容也视为注释。
- pip 先处理续行，再剥离注释；因此注释处理顺序不能与续行互换。

## 选项

### 全局选项

影响整个 `pip install` 运行的选项必须各自写在独立行。页面列出的支持项包括：

`-i`/`--index-url`、`--extra-index-url`、`--no-index`、`-c`/`--constraint`、`-r`/`--requirement`、`-e`/`--editable`、`-f`/`--find-links`、`--no-binary`、`--only-binary`、`--prefer-binary`、`--require-hashes`、`--no-require-hashes`、`--pre`、`--all-releases`、`--only-final`、`--trusted-host`、`--use-feature`。

### 单项（per-requirement）选项

可作用于单个需求的选项是 `--config-settings` 和 `--hash`（用于 hash-checking mode）。页面注明这类选项自 pip 7.0 起加入。

## 引用其他文件

- `-r more_requirements.txt`（或 `--requirement`）可引用另一个 requirements 文件。
- `-c some_constraints.txt`（或 `--constraint`）可引用 constraints 文件。

## 环境变量展开

pip 支持在 requirements 文件中展开环境变量（页面注明自 pip 10.0 起支持）。变量名必须使用 POSIX 形式并以大写名称置于花括号中，例如 `${NAME}`（页面示例为 `${API_TOKEN}`）；pip 在运行时从主机环境查找对应变量。`$NAME` 和 `%NAME%` 等其他展开语法不受支持。
