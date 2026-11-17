# 主 Agent 与 Subagent 权限

主 agent 负责仓库理解、用户范围和成功标准、规划、specialist 与 subagent 调用、选定解释器、共享 worktree 集成、NPU 资源调度、覆盖协调、最终结论以及下一步行动决策。

specialist skill 与 subagent 是正交的。skill 提供有边界的流程和知识；subagent 提供独立上下文并可以使用 specialist，但只负责明确委派给它的目标和文件。

仓库扫描、官方来源研究、授权资产清单、单个具体故障的调查或修复，以及证据稳定后的报告或文档起草，只要职责边界明确，都可以按需委派给 subagent。小型或紧密耦合的工作保留在主任务中。

只读扫描和研究可以并行。feature-model 配对文件指定一个写入者，每个源码区域指定一个写入者，共享 worktree 中最多允许一个活动 repair。环境或 vLLM 安装以及真实 NPU 执行必须串行，除非主 agent 已明确划分设备、端口和分布式资源。

每次委派都要写明绝对仓库路径、branch 或 commit、目标、读写权限、独占文件所有权，以及预期发现、证据、改动、验证和限制。执行工作还要收到选定的绝对 Python、cwd、原始命令和有界日志。

Subagent 向主任务返回直接证据。它们不选择下一仓库级阶段、不作总体支持结论、不发布改动，也不创建 handoff/result 产物。主 agent 在接受结果前重新检查 live checkout、diff、来源和决定性命令。subagent 不可用时，在主任务中直接调用同一个 specialist。

Python helper 可以探测事实、发现候选项、运行专项 smoke 或校验语义文件。它们不得推进阶段、选择重试、修改 queue/state/cursor 文件，也不得把 schema 有效性变成支持结论。

直接命令和日志是有效证据。主 agent 继续工作前不需要 controller 形态的产物。
