---
name: mindspeed-llm-fsdp2-main-control
description: 作为 MindSpeed-LLM FSDP2 优化的简单主控，在隔离 Git worktree 中按顺序调用已启用的优化 Skill，只为通过训练和验证的修改创建版本提交，并把版本、结论和子报告路径汇总到一个 Markdown 报告。适用于执行或恢复 FSDP2 融合算子等自动优化、保护原始工作区和获得单一优化报告的任务。
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM FSDP2 简单优化主控

本 Skill 只负责三件事：建立隔离工作区、调用优化 Skill、控制 Git 版本并生成一个合并报告。代码方案、训练、性能与精度判断由优化 Skill 完成。

## 边界

- 当前 Agent 始终持有主流程；不要用脚本模拟 Agent 编排。
- 完整读取并遵循被调用优化 Skill 的 `SKILL.md`，但不介入其候选方案和重试策略。
- 不重新计算或覆盖子 Skill 的验证结论。
- 子 Skill 不得提交代码、切换分支或改写历史。
- 不自动 push、merge、rebase、amend 或发布。

## 当前阶段

按顺序调用：

1. `$mindspeed-llm-fsdp2-fused-op-integration`

当前只有一个阶段，因此一次运行最多创建一个优化 commit。新增阶段时，只需在本列表追加 Skill，并说明它是否依赖前一阶段。

## 执行流程

### 1. 初始化

在修改文件前确认：

- 目标仓库、原始分支、`HEAD` 和工作区状态；
- FSDP2 后端、真实模型入口和 modeling 文件；
- 训练配置、验证门限、输出目录和允许修改路径；
- 所需优化 Skill 可用。

生成唯一 `run_id`。优先使用用户指定的报告目录；否则使用当前 Skill 集合根目录下的 `output/main-control-<run_id>/`。立即创建其中的 `fsdp2_optimization_report.md`。报告是主控唯一的状态和汇总产物；不要创建额外的状态 JSON、版本清单或阶段摘要文件。

只记录私有配置的路径，不复制密码、私钥、token 或配置内容。

### 2. 建立隔离版本

从原始 `HEAD` 创建新的调优分支和独立 Git worktree，并将该 worktree 作为所有子 Skill 唯一允许使用的本地目标仓库。

- 原始仓库干净时，原始 `HEAD` 就是 baseline，不创建空 commit。
- 原始仓库有未提交修改时，不擅自提交、stash、复制或丢弃；仍默认从原始 `HEAD` 创建 worktree。
- 不在用户原始分支上创建优化 commit。
- 创建前确认分支名和 worktree 目录尚不存在。

把原始仓库、分支、baseline、调优分支和 worktree 绝对路径写入合并报告。

### 3. 固定远端验证目标

- 不修改共享的 `remote_config.local.json` 来切换仓库。
- 子 Skill 调用 `$mindspeed-llm-remote-validation` 时，必须传入 `--repo-root <worktree_root>`。
- 远端验证完成 `plan` 后，比较 `plan.repository.local_path` 与 worktree 的规范化绝对路径。不同则结果为 `blocked`，不得同步或训练。
- 并发任务必须使用不同的 `--remote-repo`，避免覆盖远端目录。

### 4. 执行优化阶段

每个阶段只执行以下步骤：

1. 确认 worktree 干净并记录 `input_commit`。
2. 向优化 Skill 传入 worktree、modeling 文件、训练配置、门限、输出目录和允许修改路径。
3. 等待优化 Skill 完成修改、训练、验证和接受判断。
4. 读取子 Skill 的结构化结论、关键错误片段和报告路径，不复制完整日志。
5. 确认 `HEAD` 仍等于 `input_commit`，再检查 diff、未跟踪文件和修改范围。
6. 按下一节作出唯一结果并处理提交。
7. 立即更新合并报告。

阶段结果只使用：`improved`、`no_improvement`、`failed`、`blocked`。

## 简单提交规则

仅当以下条件全部满足时，结果为 `improved` 并创建一个 commit：

- 子 Skill 要求的训练和 validators 全部通过；
- 子 Skill 明确接受该修改；
- `HEAD` 未变化；
- worktree 只包含已接受且位于允许范围内的文件；
- 没有未知修改、被拒绝实验残留或意外未跟踪文件。

显式暂存已核验文件，不使用 `git add -A`。提交后把完整 SHA 写入报告，并作为下一阶段的 `input_commit`。

其他情况一律不提交：

- 已完成验证但没有保留修改：`no_improvement`；
- 执行出错或验证未完成：`failed`；
- 路径错误、环境缺失、工作区来源不明或无法形成可信结论：`blocked`。

如未提交但 worktree 仍有修改，保存子报告和 diff 路径，在报告中说明并停止。未经用户授权，不清理、恢复或丢弃这些修改；禁止 `git reset --hard`。

## 训练监控

- 训练生命周期、完整日志落盘和资源检查由优化 Skill 或远端验证 Skill 负责。
- 主控只向用户说明运行目录和 `training.log` 路径，然后等待最终结构化报告。
- 默认不按 iteration 或固定时间轮询，也不把训练日志复制到上下文或合并报告。
- 失败时只读取结构化错误和必要的相邻日志；信息不足时才扩大读取范围。
- 用户取消时，停止当前子流程，确认训练进程状态，并把取消原因和未提交修改写入合并报告。

## 唯一报告

在初始化、阶段结束、提交、失败、阻塞或取消时更新：

```text
<report_root>/fsdp2_optimization_report.md
```

报告保持简短，包含：

- `run_id`、当前/最终结果；
- 原始仓库、worktree、分支、baseline 和最终 commit；
- 模型、modeling 文件、训练配置和门限；
- 每个阶段的 Skill、结果、修改摘要、性能/精度结论和 commit；
- 子报告、验证报告、日志与 diff artifact 的路径；
- 未提交修改、失败原因和遗留问题。

训练日志和子 Skill 产物留在各自目录，主控只记录路径。恢复运行时先读本报告和当前 Git 状态，不重建独立状态机。

## 结束

- `improved`：报告最终 commit，等待用户审阅；
- `no_improvement`：报告 baseline 未变化；
- `failed` 或 `blocked`：报告原因和安全的下一步；
- 不自动合并或推送调优分支。
