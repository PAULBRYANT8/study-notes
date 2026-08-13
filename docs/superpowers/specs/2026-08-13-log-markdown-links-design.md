# `log.md` Markdown 相对链接设计

## 目标

让 `log.md` 中提到的每个现有文档在普通 Markdown 渲染器和 Obsidian 中都能直接点击，同时保持当前显示名称和日志内容不变。

## 范围

- 将 `log.md` 中的 Obsidian Wikilink 转为相对 `log.md` 的标准 Markdown 链接。
- 更新 `log.md` 顶部的格式说明。
- 更新 `SCHEMA.md` 的链接规范和日志录入步骤，明确 `log.md` 使用标准 Markdown 相对链接；其他笔记继续使用 Wikilink。
- 不改动日志日期、动作描述、目标文档内容或其他笔记中的链接。

## 链接规则

- 根目录文件示例：`[SCHEMA](SCHEMA.md)`。
- 子目录文件示例：`[context-parallel](concepts/context-parallel.md)`。
- 显示文本沿用当前 Wikilink 文本；链接目标包含目录和 `.md` 扩展名。
- 路径中的空格及其他需要转义的字符使用 URL 编码，以兼容 Obsidian 和普通 Markdown 渲染器。

## 校验

- 检查 `log.md` 中每个 Markdown 内部链接的目标文件均存在。
- 检查日志条目中不再残留 Wikilink。
- 检查 `SCHEMA.md` 与 `log.md` 的格式说明一致。
- 运行 `git diff --check`，确认没有空白字符错误。
