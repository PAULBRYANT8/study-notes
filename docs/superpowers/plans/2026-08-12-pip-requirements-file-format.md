# pip Requirements 文件格式速查 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 pip 官方 Requirements File Format 整理为可直接查阅的中文速查笔记，并同步到远端 `main`。

**Architecture:** 在最新 `origin/main` 的临时隔离克隆中，新增一份带来源的 `raw/` 事实摘录和一份 `concepts/` 速查；再以最小范围更新 `index.md`、`log.md` 及确有依赖上下文的旧笔记。只暂存本任务文件，提交后推送 `origin/main`，不触碰当前工作区无关修改。

**Tech Stack:** Markdown、Obsidian 内部链接、pip 官方文档；使用 `git` 做版本控制，使用 `rg` 和 shell 校验 Markdown 结构。

---

### Task 1: 建立隔离执行目录并确认基线

**Files:**
- Create: `/tmp/study-notes-pip-sync-<随机>/repo`（临时克隆，不进入仓库）

- [ ] **Step 1: 获取最新远端并克隆**

Run:
```bash
git clone --branch main --single-branch https://github.com/PAULBRYANT8/study-notes.git /tmp/study-notes-pip-sync-<随机>/repo
```
Expected: 克隆成功，工作区干净，HEAD 等于 `origin/main`。

- [ ] **Step 2: 确认基线文档和目录**

Run:
```bash
test -f CLAUDE.md && test -f SCHEMA.md && test -d raw && test -d concepts
```
Expected: exit 0；仓库规范和目标目录存在。

### Task 2: 保存官方来源摘录

**Files:**
- Create: `raw/2026-08-12-pip-requirements-file-format.md`

- [ ] **Step 1: 添加来源元数据和事实摘录**

内容必须包含 URL、访问日期、页面版本、requirements 文件用途、五类行结构、编码/续行/注释、支持选项、`-r`/`-c` 文件引用和 `${NAME}` 环境变量语法。

- [ ] **Step 2: 检查 raw 笔记**

Run:
```bash
rg -n 'TODO|TBD|pip.pypa.io|访问日期|环境变量' raw/2026-08-12-pip-requirements-file-format.md
```
Expected: 来源和各事实章节均有命中；无占位符。

### Task 3: 编写中文速查

**Files:**
- Create: `concepts/pip-requirements-file-format.md`

- [ ] **Step 1: 写入 frontmatter 和一句话结论**
- [ ] **Step 2: 写入 30 秒速查表和常见写法**
- [ ] **Step 3: 写入文件组织、语法规则和选项表**
- [ ] **Step 4: 写入环境变量、模板和检查清单**

必须覆盖包名、版本约束、环境标记、直接 URL、本地归档、editable/VCS、`-r`、`-c`、编码、续行、注释、全局/单项选项及 `${NAME}`；明确 `$NAME` 和 `%NAME%` 不支持，并把实践建议与官方事实分开。

- [ ] **Step 5: 自查速查笔记**

Run:
```bash
rg -n 'TODO|TBD|\$NAME|%NAME%|\$\{[A-Z_]+\}|--index-url|--config-settings|^-r |^-c |```' concepts/pip-requirements-file-format.md
```
Expected: 关键语法和限制均命中；无占位符。

### Task 4: 更新索引、交叉链接和日志

**Files:**
- Modify: `index.md`
- Modify: `log.md`
- Modify: 仅在实际讨论 Python 依赖/环境的旧笔记中补链接（先搜索确认）

- [ ] **Step 1: 将新概念加入 Concepts**
- [ ] **Step 2: 回链确有相关性的旧笔记**
- [ ] **Step 3: 追加日志，所有文件名使用 Obsidian 链接**
- [ ] **Step 4: 检查链接和重复条目**

### Task 5: 全面验证、提交并推送

**Files:**
- Commit: 本任务新增/修改文件

- [ ] **Step 1: 运行 `git diff --check` 和占位符扫描**
- [ ] **Step 2: 校验 frontmatter 与来源链接**
- [ ] **Step 3: 确认暂存区只含本任务文件**
- [ ] **Step 4: 提交 `docs: add pip requirements format cheat sheet`**
- [ ] **Step 5: 推送 `origin main` 并用 `git ls-remote` 核对远端提交哈希**
