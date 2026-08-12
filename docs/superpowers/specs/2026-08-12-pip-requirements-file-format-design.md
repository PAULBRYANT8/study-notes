# pip requirements 文件格式速查笔记设计

## 目标

把 pip 官方《Requirements File Format》整理成一份可直接查用的中文语法速查，帮助读者快速回答“这一行能写什么、这个选项影响什么、变量如何展开以及哪些写法不受支持”。内容以官方页面为唯一事实来源，并明确区分官方语法与本笔记的使用建议。

## 来源

- 官方页面：<https://pip.pypa.io/en/stable/reference/requirements-file-format/>
- 访问日期：2026-08-12
- 页面版本：pip documentation v26.2.1（页面标示）

## 收录范围

- requirements 文件的用途、命名约定与“主要供 pip 消费”的兼容性边界。
- 每行支持的五类内容：pip 选项、requirement specifier、归档 URL/路径、本地项目路径、VCS 项目 URL。
- 包名、版本约束、环境标记、直接 URL、注释和文件引用的最小示例。
- UTF-8 默认编码、PEP 263 编码声明、反斜杠续行和注释处理顺序。
- 全局 `pip install` 选项与单个 requirement 选项的分组及代表性用法。
- `-r/--requirement` 引用其他 requirements 文件和 `-c/--constraint` 引用 constraints 文件。
- `${UPPERCASE_NAME}` 环境变量展开语法，以及 `$NAME`、`%NAME%` 不受支持这一限制。
- 面向日常编辑和审查的常见坑、模板与提交前检查清单；这些部分标注为实践建议，不冒充 pip 官方规范。

不在本次范围内：完整的 requirement specifier 规范、constraints 文件的独立语义、pip resolver 算法、安装安全专题、`pyproject.toml`/PEP 621 依赖声明，以及其他工具对 requirements 文件的扩展。

## 文件布局

1. `raw/2026-08-12-pip-requirements-file-format.md`
   - 保存来源 URL、访问日期、页面版本和按章节整理的事实摘录。
   - 不复制整页导航或大段原文；只保留可追溯的章节要点和短示例。
2. `concepts/pip-requirements-file-format.md`
   - 面向查阅者的完整中文速查。
   - 使用 frontmatter，`type: concept`，来源链接指向 raw 笔记和官方 URL。
3. `index.md`
   - 在 Concepts 分区加入 `[[pip-requirements-file-format]]` 及一句话钩子。
4. `log.md`
   - 末尾追加一条记录，提及的每个库文件均使用 Obsidian 链接。
5. 相关旧笔记
   - 扫描 Python、PyTorch、NPU 学习路线笔记；只在确实讨论环境/依赖安装的位置补入新笔记链接，不做无关改写。

## 速查笔记结构

```text
# pip requirements 文件格式
一句话结论

## 30 秒速查
每行可写什么的总表

## 1. 最常用写法
包名、版本约束、环境标记、直接 URL、本地 wheel、editable/VCS 示例

## 2. 文件组织
-r、-c、相对路径与拆分文件建议

## 3. 语法规则
编码、续行、注释及处理顺序

## 4. pip 选项
全局选项表；单项选项表；作用范围对比

## 5. 环境变量
${NAME} 示例、注入时机、不支持的展开形式

## 6. 常见坑与检查清单
可移植性、敏感信息、选项作用域、路径与续行检查

## 相关笔记
内部链接

## 来源
raw 笔记与官方 URL
```

## 设计取舍

- 推荐“表格 + 最小代码块”而不是逐段翻译，目标是降低查找成本。
- 对官方明确支持的语法给出直接结论；对安全、可复现性和团队协作建议使用“建议”措辞。
- 不把 `requirements.txt` 当作唯一合法文件名，强调文件名只是惯例。
- 不把 requirements 文件描述成通用标准：完整语法与 pip 内部实现和命令行选项耦合，其他工具不一定兼容。
- 示例中的 token 使用环境变量占位符，不写入真实凭据。

## 验收标准

- raw 与 concepts 两个文件均存在，frontmatter 字段完整且来源可追溯。
- 速查覆盖官方页面的用途、五类行结构、编码、续行、注释、全局/单项选项、文件引用和环境变量章节。
- 至少包含一个可复制的 requirements 模板，并明确 `$NAME` 与 `%NAME%` 不支持。
- `index.md` 有唯一且准确的 Concepts 条目；`log.md` 新记录中的每个文件名都写成 `[[链接]]`。
- 相关旧笔记的回链经过实际扫描；没有为不存在的文件制造错误链接，也不改动 raw 既有材料。
- Markdown 结构检查通过；不留下 `TODO`、`TBD`、占位章节或未闭合的代码围栏。
- 实施过程中保留用户现有的无关工作区修改。
