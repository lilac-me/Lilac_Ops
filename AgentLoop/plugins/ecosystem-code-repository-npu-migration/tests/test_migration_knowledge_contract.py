"""Contracts for the prompt-first shared migration knowledge library."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "references" / "migration_knowledge"


def test_knowledge_library_has_navigable_topics() -> None:
    expected = {
        "index.md",
        "backend_runtime.md",
        "distributed_serving.md",
        "operators_semantics.md",
        "checkpoint_launchers.md",
    }
    assert {path.name for path in KNOWLEDGE.iterdir()} == expected
    index = (KNOWLEDGE / "index.md").read_text(encoding="utf-8")
    for filename in sorted(expected - {"index.md"}):
        assert f"]({filename})" in index
    for phrase in (
        "最深层 exception",
        "rg -n -i",
        "当前官方来源",
        "TorchNPU 官方仓库",
        "昇腾 PyTorch 安装文档",
        "精确 URL、访问日期",
    ):
        assert phrase in index


def test_topics_explain_recognition_local_evidence_fix_and_validation() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(KNOWLEDGE.glob("*.md"))
        if path.name != "index.md"
    )
    for phrase in (
        "识别信号",
        "检查",
        "保留",
        "验证",
        "不能证明",
        "CPU",
        "CUDA",
        "NPU",
    ):
        assert phrase.lower() in combined.lower()


def test_knowledge_library_has_no_legacy_control_or_kg_client_language() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(KNOWLEDGE.glob("*.md"))
    ).lower()
    for forbidden in (
        "queue.json",
        "state.json",
        "cursor",
        "attempt tree",
        "controller handoff",
        "ascend_kg",
        "ascend.wiki",
        "snapshot_manifest",
    ):
        assert forbidden not in combined
