# Batch Invariance Validation — orchestration

`batch-invariance-validation` 使用同一份 Python 状态机、契约和阶段指令支持不同 Agent 平台。

## 运行根目录

从已加载的 `skills/batch-invariance-validation/SKILL.md` 向上解析插件根目录，只有同时存在
`workflows/batch-invariance-validation.yaml` 和 `engine/state_machine.py` 时才接受该目录。源码检出中使用
`plugins/batch-invariance-validation`。所有引擎命令都传入解析后的绝对路径，不硬编码安装缓存目录。

## 调度规则

- orchestrator 是唯一入口，独占 TP 计划展示、用户确认、`run_config.json` 写入、状态推进和最终汇总。
- 执行态必须调用 `agents/<role>/AGENT.md` 指定的角色；暂停态和终态不得派生阶段 Agent。
- 子 Agent 只返回符合 `contracts/stage_result.schema.json` 的结果和 YAML 声明的精确 handoff。
- 每次转移前运行门禁，再调用 `engine/state_machine.py next`。禁止依据提示词中的流程描述自行选择下一状态。
- `BLOCKED` 是可恢复暂停，不消耗格式补正次数；只有结构或证据门禁失败才允许最多两次补正。

Codex 使用原生子任务能力派生角色；Claude Code 使用 Agent 工具。两端都通过 Python CLI 推进同一份 YAML，
不维护第二份 JavaScript 状态机。
