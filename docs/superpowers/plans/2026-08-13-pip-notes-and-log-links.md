# pip Requirements 笔记与可移植日志链接实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 pip Requirements File Format 中文速查及来源笔记，把 `log.md` 的内部文档名改成 Obsidian 与普通 Markdown 均可点击的相对链接，并删除两份已完成使命的设计文档。

**Architecture:** 新增一份只承载官方事实的 `raw/` 摘录和一份面向查阅的 `concepts/` 速查，再以最小范围更新 `index.md` 与 `log.md`。`log.md` 位于库根目录，因此每个链接都写成从库根出发的 Markdown 相对路径；其他笔记仍沿用现有 Wikilink。当前工作区中的无关修改不纳入本任务。

**Tech Stack:** Markdown、Obsidian 内部链接、pip 官方文档 v26.2.1、Git、`rg` 与 shell 校验。

---

### Task 1: 创建 pip 官方来源摘录

**Files:**
- Create: `raw/2026-08-12-pip-requirements-file-format.md`

- [ ] **Step 1: 记录实施前基线**

Run:

```bash
test ! -e raw/2026-08-12-pip-requirements-file-format.md
```

Expected: exit 0，证明目标文件当前确实不存在。

- [ ] **Step 2: 写入 frontmatter、来源信息和官方事实**

文档必须使用 `type: raw`，记录官方 URL、访问日期 `2026-08-12` 和页面版本 `v26.2.1`，并按以下结构整理：

```markdown
# pip Requirements File Format 官方资料摘录

## 来源信息
## 用途与兼容性边界
## 每行支持的结构
## 编码、续行和注释
## 支持的 pip install 选项
### 全局选项
### 单项 requirement 选项
## 引用其他文件
## 环境变量
```

事实范围固定为：`requirements.txt` 只是惯用名；完整语法主要供 pip 消费；每行支持全局选项、requirement specifier、归档 URL/路径、editable 本地项目和 editable VCS URL；默认 UTF-8，可用 PEP 263 声明覆盖；未转义反斜杠续行先于注释剥离；全局选项单独占行；单项选项为 `--config-settings` 和 `--hash`；文件引用使用 `-r`/`-c`；环境变量只支持 `${UPPERCASE_NAME}`，不支持 `$NAME` 和 `%NAME%`。

- [ ] **Step 3: 校验来源摘录**

Run:

```bash
rg -n 'pip.pypa.io|2026-08-12|v26.2.1|UTF-8|--config-settings|--hash|\$\{UPPERCASE_NAME\}|\$NAME|%NAME%' raw/2026-08-12-pip-requirements-file-format.md
```

Expected: 所有来源、版本和关键规则均有命中。

### Task 2: 创建中文语法速查

**Files:**
- Create: `concepts/pip-requirements-file-format.md`

- [ ] **Step 1: 写入 frontmatter 和结论**

使用 `type: concept`、日期 `2026-08-13`，`sources` 同时包含官方 URL 和 `"[[raw/2026-08-12-pip-requirements-file-format]]"`。首句明确：requirements 文件是 pip install 的逐行输入清单，不等同于通用依赖标准，也不要求文件名必须是 `requirements.txt`。

- [ ] **Step 2: 写入 30 秒速查和最常用语法**

速查表覆盖以下五类行结构，并给出可复制示例：

```text
requests
requests>=2.32,<3
importlib-metadata>=7 ; python_version < "3.10"
urllib3 @ https://github.com/urllib3/urllib3/archive/refs/tags/1.26.8.zip
./dist/example_pkg-1.0.0-py3-none-any.whl
-e ./local-project
-e git+https://github.com/example/project.git#egg=project
```

- [ ] **Step 3: 写入文件组织、解析规则和选项作用域**

必须解释 `-r`、`-c`、相对路径，UTF-8/PEP 263、反斜杠续行与注释处理顺序；用两个表分别列出官方页面的全局选项与单项选项，并说明全局选项必须单独占行。

- [ ] **Step 4: 写入环境变量、模板、常见坑和检查清单**

模板至少包含 index 配置、拆分 requirements 文件、环境标记、直接 URL 和 `${PRIVATE_INDEX_TOKEN}`。明确 `$NAME`、`%NAME%` 不展开；敏感值只放环境变量；“固定版本、使用 hash、拆分文件”等内容标为实践建议，而不是官方格式要求。

- [ ] **Step 5: 校验速查内容**

Run:

```bash
rg -n '30 秒速查|^-r |^-c |--index-url|--config-settings|\$\{PRIVATE_INDEX_TOKEN\}|\$NAME|%NAME%|实践建议|官方规则' concepts/pip-requirements-file-format.md
```

Expected: 关键语法、边界和事实/建议区分均有命中。

### Task 3: 更新目录和追加日志记录

**Files:**
- Modify: `index.md`
- Modify: `log.md`

- [ ] **Step 1: 在 Concepts 分区加入唯一索引项**

在 `index.md` 的 Concepts 标题下加入：

```markdown
- [[pip-requirements-file-format]] — pip requirements 每行语法、选项作用域、文件拆分、环境变量及常见坑速查
```

- [ ] **Step 2: 确认无需改动旧笔记**

Run:

```bash
rg -n -i '\b(pip|requirements\.txt|requirements file|依赖安装|虚拟环境|venv)\b' --glob '*.md' --glob '!raw/**' --glob '!docs/superpowers/**'
```

Expected: 没有现有知识笔记真正讨论 pip requirements 文件；不为无关工作记录补链接。

- [ ] **Step 3: 在 `log.md` 末尾追加实施记录**

先追加以下最终格式的记录：

```markdown
- 2026-08-13 整理 pip requirements 文件格式的官方来源摘录与中文速查，并加入内容目录 —— [2026-08-12-pip-requirements-file-format](raw/2026-08-12-pip-requirements-file-format.md)、[pip-requirements-file-format](concepts/pip-requirements-file-format.md)、[index](index.md)、[log](log.md)
```

### Task 4: 把现有日志链接转换成标准 Markdown 相对链接

**Files:**
- Modify: `log.md`

- [ ] **Step 1: 更新顶部格式说明**

将格式示例改为：

```markdown
格式：`- YYYY-MM-DD 动作 —— [文件名](相对路径.md)、[文件名](相对路径.md)`

**提到的每个文件都必须写成标准 Markdown 相对链接**，不能只写文件名或 `[[Wikilink]]`。
```

- [ ] **Step 2: 逐条映射全部 Wikilink**

根目录文件使用 `SCHEMA.md`、`index.md`、`CLAUDE.md`、`log.md`；其余目标根据实际目录写成 `raw/...md`、`entity/...md`、`concepts/...md`、`comparisons/...md`、`queries/...md` 或 `work/...md`。显示文字保持现有 Wikilink 文字不变。

- [ ] **Step 3: 检查没有残留 Wikilink**

Run:

```bash
rg -n '\[\[[^]]+\]\]' log.md
```

Expected: exit 1 且无输出。

### Task 5: 删除已实施的设计文档

**Files:**
- Delete: `docs/superpowers/specs/2026-08-12-pip-requirements-file-format-design.md`
- Delete: `docs/superpowers/specs/2026-08-13-log-markdown-links-design.md`

- [ ] **Step 1: 删除用户指定的两个文件**

使用补丁明确删除两个目标文件，不删除同目录下的其他设计，也不删除现有实施计划。

- [ ] **Step 2: 校验删除范围**

Run:

```bash
test ! -e docs/superpowers/specs/2026-08-12-pip-requirements-file-format-design.md
test ! -e docs/superpowers/specs/2026-08-13-log-markdown-links-design.md
```

Expected: 两条命令均 exit 0。

### Task 6: 全面验证并提交任务改动

**Files:**
- Verify: `raw/2026-08-12-pip-requirements-file-format.md`
- Verify: `concepts/pip-requirements-file-format.md`
- Verify: `index.md`
- Verify: `log.md`
- Verify deletion: 两份指定设计文档

- [ ] **Step 1: 验证日志中的每个目标存在**

Run:

```bash
while IFS= read -r target; do
  test -f "$target" || exit 1
done < <(perl -ne 'while (/\]\(([^)]+\.md)\)/g) { print "$1\n" }' log.md)
```

Expected: exit 0，不存在缺失目标。

- [ ] **Step 2: 验证 frontmatter、代码围栏和占位符**

Run:

```bash
for doc in raw/2026-08-12-pip-requirements-file-format.md concepts/pip-requirements-file-format.md; do
  test "$(head -n 1 "$doc")" = "---" || exit 1
  awk '/^```/{count++} END{exit count % 2}' "$doc" || exit 1
done
perl -0777 -ne 'exit 1 if /^##[^#\n]*\n\s*(?=##|\z)/m' raw/2026-08-12-pip-requirements-file-format.md concepts/pip-requirements-file-format.md
rg -n 'TODO|TBD' raw/2026-08-12-pip-requirements-file-format.md concepts/pip-requirements-file-format.md
```

Expected: `for` 和 `perl` exit 0；`rg` exit 1 且无输出。

- [ ] **Step 3: 检查索引和日志唯一性**

Run:

```bash
rg -n 'pip-requirements-file-format' index.md log.md
```

Expected: `index.md` 一条索引，`log.md` 一条新日志记录。

- [ ] **Step 4: 检查补丁格式与任务范围**

Run:

```bash
git add -- raw/2026-08-12-pip-requirements-file-format.md concepts/pip-requirements-file-format.md index.md log.md docs/superpowers/plans/2026-08-13-pip-notes-and-log-links.md docs/superpowers/specs/2026-08-13-log-markdown-links-design.md
git diff --cached --check
git diff --cached --name-status
git status --short
```

Expected: `git diff --cached --check` exit 0；暂存区只新增两份 pip 笔记和本计划、修改 `index.md`/`log.md`、删除已跟踪的日志链接设计；工作区仍保留原有无关改动。未跟踪的 pip 设计删除后不会形成 Git 删除记录。

- [ ] **Step 5: 只暂存并提交本任务文件，不推送远端**

提交信息：

```text
docs: add pip requirements notes and portable log links
```

提交前再次用 `git diff --cached --name-status` 确认不包含 `.obsidian/workspace.json`、`work/swiglu-group-接入复盘.md` 或其他既有未跟踪工作文件，然后执行：

```bash
git commit -m "docs: add pip requirements notes and portable log links"
```
