# 语义输出与工作笔记

语义文件是持久化的规划事实，不是 workflow 控制面。仓库级 goal 生成 feature model 和 coverage plan；只有存在候选根目录或计划中的示例需要资产时，才生成本地资产清单。同一个 goal 可以保留两份小型工作笔记，帮助主 Agent 在 prompt 压缩后恢复有用上下文。

允许的 `.npu-agent` 内容包括：

- 仓库功能地图和代表性覆盖计划（`feature_model.json` 和 `coverage_plan.json`）；
- 有界的本地模型或数据集清单（`local_asset_inventory.json`），不包含第二个 entry-to-asset 绑定产物；
- 当前迁移笔记（`migration_progress.md` 和 `compatibility.json`），写入目标仓库的 `.npu-agent/` 目录；
- 用户明确请求的报告。

`migration_progress.md` 记录 goal、已完成工作、当前重点、剩余工作、blocker 和重要决定。`compatibility.json` 在事实有用时，按 entry 或 group 记录带命令、日志和风险的简要结果。goal skill 提供[进度示例](../../../skills/ecosystem-code-repository-npu-migration/model-adaptation/goal/examples/migration_progress.md)和[兼容性示例](../../../skills/ecosystem-code-repository-npu-migration/model-adaptation/goal/examples/compatibility.json)；只把它们的结构作为便利参考，不是 schema。它们是供参考的工作记忆，不是崩溃恢复 checkpoint、任务身份、执行队列或支持事实来源。不要创建 queue、state、cursor、attempt、controller-context、transition、liveness 或 artifact-manifest 文件。不要从任一文件的存在或内容推断 PASS。

Subagent 委派以及任务/迁移处置仍属于直接任务上下文。主 goal Agent 是两份工作笔记的唯一写入者；subagent 和 specialist skill 直接返回事实。不要创建 subagent handoff、result、ownership 或独立的完成状态产物。

每条支持结论仍必须引用当前解释器、精确命令、退出码、相关 stdout/stderr、NPU 观察结果和实际执行的语义行为。具备这些事实时，直接命令就是充分证据。

环境、repair 和 vLLM skill 默认在当前任务中返回证据。`report-pr` 可以读取两份工作笔记，并在请求时将报告持久化到 `.npu-agent/reports/`，但它自身不维护工作笔记。

迁移知识搜索（包括明确提供的任务级外部 Markdown 根目录）和官方文档查询默认也只保留在当前任务中。不要创建 `.npu-agent/knowledge`、查询 snapshot 或第二个知识 index。外部文档影响补丁时，在直接任务证据中保留选定根目录、相对文件、版本上下文和使用的结论；网络文档影响补丁时，还要保留精确 URL 和访问日期。只有通过 `report-pr`，才能将这些事实纳入请求的报告。
