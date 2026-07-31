# DeepSeek-V4 Q&A Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `work/deepseek-v4.md` as a durable, navigable record of the current and future DeepSeek-V4 questions and answers.

**Architecture:** Use one chronological Markdown ledger. Its top-level index preserves each complete original question and links to a permanent Obsidian block ID attached to that same question in the body; numbered entries then hold the detailed answer and a return link.

**Tech Stack:** Markdown, Obsidian wiki links, Obsidian block IDs, Git validation commands

---

### Task 1: Create the first DeepSeek-V4 question entry

**Files:**

- Create: `work/deepseek-v4.md`
- Reference: `docs/superpowers/specs/2026-07-31-deepseek-v4-qa-note-design.md`
- Reference: `work/swiglu-group-接入复盘.md`

- [x] **Step 1: Confirm the destination does not already exist**

Run:

```bash
test ! -e work/deepseek-v4.md
```

Expected: exit code 0 with no output.

- [x] **Step 2: Create the fixed document header and question index**

Create `work/deepseek-v4.md` with:

```markdown
# DeepSeek-V4 原始问答记录

本文只记录在对话中提出的 DeepSeek-V4 相关原始问题及其回答，不回填已有专题笔记。问题编号和块锚点一经创建便不再修改，方便长期引用。

## 问题索引

- [[#^q001-router|Q001：Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下]]
```

- [x] **Step 3: Add Q001 with the exact original wording and permanent block ID**

Append this entry header without changing spacing or punctuation inside the original question:

```markdown
## Q001：Router 指的是什么

- 记录日期：2026-07-31
- 主题：MoE、Router、routed score、Top-K、共享专家与路由专家

> **原问题：** Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下 ^q001-router
```

- [x] **Step 4: Add the detailed answer**

The answer must contain these concrete sections and claims:

1. “核心定义”：Router is the small learnable gating module in each MoE layer, not an expert and not an external scheduler.
2. “输入与输出”：for token hidden state \(u_t^l\), explain Router logits, normalized routing scores, Top-K expert indices, and gate weights.
3. “计算过程”：show \(z_{i,t}=(u_t^l)^T e_i^l\), \(s_{i,t}=\operatorname{Softmax}_i(z_{i,t})\), Top-K masking, and weighted expert output.
4. “数字示例”：use scores `[0.58, 0.13, 0.03, 0.26]` with Top-2 selecting experts 1 and 4.
5. “DeepSeek-V4 当前代码语境”：state that Router runs before dispatch, token reorder and GMM; routed score scales selected routed rows around the activation path and is not `w1/w2/w3`.
6. “共享专家”：state that shared experts bypass Router and do not carry routed score in the current recorded path.
7. “训练与负载均衡”：explain end-to-end learning, routing collapse, expert starvation, and the need for balance losses.
8. “与 Attention 的区别”：Attention chooses context tokens; Router chooses FFN experts.
9. “术语速查”：map logits, routing scores, Top-K indices, gate weights, dispatch and combine.
10. End with `[[#问题索引|返回问题索引]]`.

Do not assert a DeepSeek-V4 expert count or routing count unless it is established by the current repository material. When referring to formulas from DeepSeekMoE-style routing, label them as the common mechanism rather than a confirmed V4 configuration.

### Task 2: Validate navigation and content

**Files:**

- Validate: `work/deepseek-v4.md`

- [x] **Step 1: Verify the file and exact original question**

Run:

```bash
test -f work/deepseek-v4.md
rg -F -c 'Router 为每个 token 选择若干专家，并产生 routed score。这其中的Router指的是什么，请详细说明一下' work/deepseek-v4.md
```

Expected: the file check exits 0 and the count is `2`, once in the index and once in the body.

- [x] **Step 2: Verify the index target and body anchor match**

Run:

```bash
rg -n '\[\[#\^q001-router\||\^q001-router$' work/deepseek-v4.md
```

Expected: two matches using the exact identifier `q001-router`: the index link and the original-question block.

- [x] **Step 3: Scan for incomplete content and Markdown whitespace errors**

Run:

```bash
rg -n 'TO''DO|TB''D|待''补充|待''完善' work/deepseek-v4.md
rg -n '[[:blank:]]+$' work/deepseek-v4.md
```

Expected: both scans exit 1 with no matches.

- [x] **Step 4: Review the isolated diff**

Run:

```bash
sed -n '1,280p' work/deepseek-v4.md
```

Expected: the complete new Q&A note is displayed for review.

- [x] **Step 5: Commit only the implementation and plan**

Run:

```bash
git add work/deepseek-v4.md docs/superpowers/plans/2026-07-31-deepseek-v4-qa-note.md
git diff --cached --name-status
git diff --cached --check
git commit -m "docs: add DeepSeek-V4 Q&A note"
```

Expected: the staged list contains only the new Q&A note and this implementation plan, the whitespace check exits 0, and the commit succeeds.
