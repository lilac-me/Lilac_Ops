"""Prompt contract for the repository migration goal skill."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "skills" / "goal" / "SKILL.md"
AUTHORITY = ROOT / "references" / "shared_agent_authority.md"
ARTIFACTS = ROOT / "references" / "shared_artifact_contract.md"
REPORT = ROOT / "skills" / "report-pr" / "SKILL.md"
EXAMPLES = ROOT / "skills" / "goal" / "examples"


def test_goal_is_prompt_first_and_routes_specialists() -> None:
    text = GOAL.read_text(encoding="utf-8")
    for phrase in (
        "主 agent 负责理解仓库",
        "不是固定的 transition graph",
        "直接命令和日志就是有效证据",
        "请求的仓库 workload",
        "代表性示例",
    ):
        assert phrase in text
    for skill in (
        "env-setup", "vllm-ascend-env", "local-assets", "feature-model",
        "repair", "report-pr", "docs-support",
    ):
        assert f"$ecosystem-code-repository-npu-migration:{skill}" in text
    assert "$ecosystem-code-repository-npu-migration:env-knowledge" not in text
    assert "custom-workflow" not in text
    assert "runtime/workflow" not in text
    assert len(text.splitlines()) < 220


def test_goal_uses_direct_validation_guide() -> None:
    guide = (ROOT / "skills" / "goal" / "references" / "validation_guide.md").read_text(encoding="utf-8")
    assert "import torch; torch.npu.set_device(0)" in guide
    assert "skills/feature-model/scripts/validate_feature_model_artifacts.py" in guide
    assert "controller" not in guide.lower()


def test_goal_consumes_prior_stage_context_without_a_handoff_artifact() -> None:
    text = GOAL.read_text(encoding="utf-8")
    for phrase in (
        "必须调用 `$ecosystem-code-repository-npu-migration:feature-model`",
        "将范围内每个 unit/component test、端到端示例、可执行脚本和文档启动命令各列一次",
        "通过 `features` 读取请求的功能",
        "通过 `requires_capabilities` 读取前置能力",
        "`capability_checks` 包含所有可行的迁移相关 unit/component test",
        "`representative_groups` 包含用于覆盖每个真实 equivalence group 的仓库原生示例或脚本",
        "不要执行 `deferred_entry_ids`",
        "保留 `scope_decisions`",
        "构建并校验 `.npu-agent/local_asset_inventory.json`",
        "查询每个需要的模型或数据集",
        "在真实命令中绑定其绝对路径",
        "生成 smoke 只证明 vLLM 能力",
        "重跑同一目标",
        "拆分 equivalence group",
        "covered by representative",
        "deferred entry 没有单独运行",
    ):
        assert phrase in text
    for forbidden in (
        "handoff.json",
        "evidence.json",
        "current_target.json",
        "current-target.json",
    ):
        assert forbidden not in text


def test_goal_preserves_adaptive_prompt_first_boundaries() -> None:
    text = GOAL.read_text(encoding="utf-8")
    assert "维护 `.npu-agent/migration_progress.md`" in text
    assert "`.npu-agent/compatibility.json`" in text
    assert "上下文压缩后恢复信息" in text
    assert "当前选择仍由 Agent 决定" in text
    assert "不要写入 queue、cursor 或 controller 产物" in text
    assert "不要把 JSON 数组顺序当成执行顺序" in text
    assert "要求额外 handoff 文件" in text
    assert "固定的 transition graph" in text


def test_goal_points_to_working_note_examples() -> None:
    text = GOAL.read_text(encoding="utf-8")
    assert "examples/migration_progress.md" in text
    assert "examples/compatibility.json" in text

    progress = (EXAMPLES / "migration_progress.md").read_text(encoding="utf-8")
    for heading in (
        "# 迁移进度",
        "## 目标",
        "## 已完成",
        "## 当前",
        "## 剩余",
        "## 重要决定",
        "## 最后更新",
    ):
        assert heading in progress

    compatibility = json.loads((EXAMPLES / "compatibility.json").read_text(encoding="utf-8"))
    assert compatibility["entries"]
    assert compatibility["entries"][0]["status"] == "PASS"
    assert compatibility["entries"][1]["status"] == "BLOCKED"
    assert "updated_at" in compatibility
    assert "summary" in compatibility


def test_goal_delegates_bounded_work_without_surrendering_main_authority() -> None:
    text = GOAL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for phrase in (
        "specialist skill 和 subagent 是正交的",
        "仅当 subagent 可用且有边界的任务足够大或独立",
        "主 agent 保留用户范围和成功标准",
        "绝对仓库路径、branch 或 commit",
        "选定 Python、cwd、原始命令和有界日志",
        "只读扫描和官方研究可以并行",
        "`feature_model.json` 与 `coverage_plan.json` 只指定一个写入者",
        "最多允许一个会修改共享 worktree 的活动 repair",
        "除非明确划分设备、端口和分布式资源，否则不要并行运行测试或示例",
        "重新检查 live checkout、diff、导入来源和决定性命令",
        "不要让 subagent 选择下一阶段或声称总体支持",
        "大范围源码扫描",
        "独立构建清单",
        "独立的只读 subagent 可以并行研究不同的相关知识主题",
        "证据集合稳定后",
    ):
        assert phrase in normalized
    assert "否则在主任务中直接调用同一个 specialist" in text


def test_goal_separates_task_disposition_from_migration_outcome() -> None:
    text = GOAL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for phrase in (
        "将任务处置和迁移结果作为两个独立的最终维度报告",
        "任务处置为 `COMPLETE`",
        "没有已知的安全可执行修复或验证",
        "仍有此类工作时为 `IN_PROGRESS`",
        "只有范围内每个 capability check 都通过、每个 representative group 都有代表项达到其 `verification_goal`，且没有请求中的 blocker 时，迁移结果才是 `SUPPORTED`",
        "部分功能有真实 NPU 证据、其他功能被阻塞或未验证时使用 `PARTIAL`",
        "基础要求阻止获得有意义的请求功能证据时使用 `BLOCKED`",
        "`COMPLETE + PARTIAL` 和 `COMPLETE + BLOCKED` 都是有效的任务结论，但绝不表示 NPU 完整支持",
        "未达到 verification goal 的代表项不能覆盖 deferred entry",
        "每个范围内 entry 都必须有真实结果、指向通过代表项的 deferred 链接，或明确的范围决策/ blocker",
        "静态结构不能证明另一个 backend 可以运行",
        "有用时将简洁的 goal 决定写入进度笔记",
        "修改明确的 goal 描述并说明差异，不要静默缩小范围",
    ):
        assert phrase in normalized

    report = REPORT.read_text(encoding="utf-8")
    assert "迁移进度、兼容性" in report
    assert "在可用时阅读 feature-model、本地资产、迁移进度、兼容性或既有报告文件" in report
    assert "不要修改目标源码或文档、重跑迁移命令、创建缺失证据、更新迁移工作笔记" in report
    assert "分别报告任务处置（`COMPLETE`、`IN_PROGRESS` 或 `STOPPED`）和迁移结果" in report
    assert "`COMPLETE` 可以与具体 blocker 共存，绝不能升级迁移结果" in report


def test_shared_agent_authority_keeps_delegation_bounded() -> None:
    authority = AUTHORITY.read_text(encoding="utf-8")
    normalized = " ".join(authority.split())
    for phrase in (
        "specialist skill 与 subagent 是正交的",
        "只负责明确委派给它的目标和文件",
        "共享 worktree 中最多允许一个活动 repair",
        "选定的绝对 Python、cwd、原始命令和有界日志",
        "不选择下一仓库级阶段、不作总体支持结论、不发布改动",
        "重新检查 live checkout、diff、来源和决定性命令",
        "subagent 不可用时，在主任务中直接调用同一个 specialist",
    ):
        assert phrase in normalized

    artifacts = ARTIFACTS.read_text(encoding="utf-8")
    assert "migration_progress.md" in artifacts
    assert "compatibility.json" in artifacts
    assert "不是崩溃恢复 checkpoint" in artifacts
    assert "不要创建 subagent handoff、result、ownership 或独立的完成状态产物" in artifacts
    for forbidden in (
        "subagent_handoff.json",
        "subagent_results.json",
        "agent_ownership.json",
        "completion_state.json",
    ):
        assert not list(ROOT.rglob(forbidden))
        assert forbidden not in GOAL.read_text(encoding="utf-8")
        assert forbidden not in authority
        assert forbidden not in artifacts


def test_goal_runs_initial_adaptation_then_tests_then_examples() -> None:
    text = GOAL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    stages = (
        "### 3. 盘点功能、测试和示例",
        "### 4. 盘点本地资产并确定代表项",
        "### 5. 执行范围内的初始 NPU 适配",
        "### 6. 运行并修复所有可行的 unit/component test",
        "### 7. 运行并修复代表性示例",
        "### 8. 协调覆盖范围并报告",
    )
    positions = [text.index(stage) for stage in stages]
    assert positions == sorted(positions)
    for phrase in (
        "根据 feature-model 的 task、runtime、topology、model、risk 维度、entry 命令、实际 API symbol 和调用路径提取知识搜索词",
        "阅读迁移知识索引（`<plugin-root>/references/migration_knowledge/index.md`）",
        "刷新当前官方文档",
        "实际 API symbol 和调用路径",
        "`PATCH`、`DEFER`、`OUT_OF_SCOPE` 或 `BLOCKED`",
        "跟踪每个范围内的静态命中",
        "只修复范围内 caller 实际触及且已证明的可移植性问题",
        "不要盲目进行仓库级 CUDA 字符串替换",
        "如果具体检查失败，继续前先使用同一命令和证据调用 `$ecosystem-code-repository-npu-migration:repair`",
        "同一命令和证据",
        "运行 `coverage_plan.json.capability_checks` 中的每个选定 entry",
        "不要仅为节省时间去重不同的可行测试",
        "unit/component 阶段结束后",
        "缺失或有歧义的资产只阻塞受影响的示例",
        "绝不静默下载",
    ):
        assert phrase in normalized
    for forbidden in (
        "asset_bindings.json",
        "entry_asset_bindings.json",
        "queue.json",
        "state.json",
        ".npu-agent/knowledge",
        "preflight_adaptation.json",
    ):
        assert forbidden not in text
