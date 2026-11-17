# 双平台运行时与角色调度

## 插件根目录解析

所有共享命令中的 `<DOC_AGENT_ROOT>` 表示 Doc-Agent 插件根目录的绝对路径。每次工作流开始时只解析一次，并在后续角色交接中传递该路径。

- Claude Code：读取宿主注入的 `CLAUDE_PLUGIN_ROOT`。
- Codex 已安装插件：从当前已加载 `doc-orchestrator` Skill 的来源路径反推；先取 `skills/doc-orchestrator/SKILL.md` 所在的 `doc-orchestrator` 目录，再向上两级得到插件根目录。
- 源码仓调试：使用仓库根目录下的 `plugins/doc-agent`。

解析后必须同时确认以下文件存在，否则停止执行并报告插件安装不完整：

- `<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml`
- `<DOC_AGENT_ROOT>/rules/rules.yaml`
- `<DOC_AGENT_ROOT>/contracts/quality_report.json`

命令执行时把 `<DOC_AGENT_ROOT>` 替换为解析后的绝对路径。不要假设存在 `CODEX_PLUGIN_ROOT`，也不要硬编码 Claude Code 或 Codex 的插件缓存目录。

## 角色调度

- Claude Code：使用 Agent 工具调度插件中注册的角色。
- Codex：读取 `<DOC_AGENT_ROOT>/agents/<role>/agent.md` 的角色正文，通过宿主原生子任务/子 Agent 能力调度。
- Codex 宿主没有子任务能力时，由编排者按相同角色顺序串行执行；输入输出契约、reviewer 独立性、handoff token 和引擎门控保持不变。
- Codex 不使用 `agent.md` 中 Claude Code 专属的 `tools` frontmatter；可用工具由 Codex 宿主决定。

语义 reviewer 必须独立接收相同基线，不得看到其他 reviewer 的结果。角色之间只传递工作区路径和契约产物路径。
