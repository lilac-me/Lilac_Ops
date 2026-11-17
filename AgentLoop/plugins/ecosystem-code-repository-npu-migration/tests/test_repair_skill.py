"""Prompt and deterministic-classifier contracts for NPU repair."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "skills" / "repair"
CLASSIFIER = REPAIR / "scripts" / "classify_failure.py"


def load_classifier():
    spec = importlib.util.spec_from_file_location("repair_classifier", CLASSIFIER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_requires_knowledge_lookup_before_hypothesis() -> None:
    text = (REPAIR / "SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    lookup = text.index("选择修复假设前")
    change = text.index("每次只改变一个相互独立的假设")
    assert lookup < change
    for phrase in (
        "最深层异常、失败 API、framework、task family、backend、topology",
        "搜索迁移知识索引（`<plugin-root>/references/migration_knowledge/index.md`）",
        "刷新与已安装版本对应的官方文档",
        "当前源码、调用方、版本保护、日志和导入路径",
        "不能替代复现或真实 NPU 证据",
        "使用相同解释器和输入重跑原始目标",
        "无效改动必须在尝试其他假设前还原",
    ):
        assert phrase in normalized


def test_repair_has_no_knowledge_or_control_artifact_contract() -> None:
    text = (REPAIR / "SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "向主 Agent 返回事实供其记录工作笔记" in normalized
    assert "不要直接创建或更新这些笔记" in normalized
    for forbidden in (
        ".npu-agent/knowledge",
        "knowledge_query.json",
        "repair.json",
        "command_proposal.json",
        "attempt_tree.json",
        "ascend_kg",
    ):
        assert forbidden not in text


def test_classifier_returns_topics_and_evidence_only() -> None:
    classifier = load_classifier()
    result = classifier.classify_text(
        "Ray actor ASCEND_RT_VISIBLE_DEVICES is empty; NPU resource unavailable\n"
        "vLLM update_weights failed with a timeout",
        ["failure.log"],
    )
    assert result["status"] == "CLASSIFIED"
    assert result["primary_category"] == "Ray NPU worker visibility or placement issue"
    assert result["knowledge_topics"] == ["distributed-serving", "backend-runtime"]
    assert set(result) == {
        "status", "primary_category", "knowledge_topics", "matches", "logs",
    }
    for match in result["matches"]:
        assert set(match) == {"category", "excerpt", "knowledge_topics"}
    serialized = json.dumps(result).lower()
    for forbidden in ("patch", "command_proposal", "selected_fix", "artifact_path"):
        assert forbidden not in serialized


def test_classifier_covers_common_knowledge_topics() -> None:
    classifier = load_classifier()
    cases = {
        "Hydra config name is missing required override": "checkpoint-launchers",
        "flash_attn failed: unsupported layout": "operator-semantics",
        "checkpoint load failed: missing shard": "checkpoint-launchers",
        "Process group HCCL collective timeout on rank 1": "distributed-serving",
        "torch.cuda.amp autocast failed for device_type npu": "backend-runtime",
    }
    for log, expected_topic in cases.items():
        result = classifier.classify_text(log, ["failure.log"])
        assert result["status"] == "CLASSIFIED", log
        assert expected_topic in result["knowledge_topics"], log


def test_classifier_unknown_log_is_nonzero_and_read_only(tmp_path: Path) -> None:
    log = tmp_path / "unknown.log"
    log.write_text("plain progress message without a failure signature\n", encoding="utf-8")
    before = log.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(CLASSIFIER), str(log)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "UNCLASSIFIED"
    assert log.read_bytes() == before
    assert {path.name for path in tmp_path.iterdir()} == {"unknown.log"}
