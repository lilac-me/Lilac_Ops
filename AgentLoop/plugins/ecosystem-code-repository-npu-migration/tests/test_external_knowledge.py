"""Contracts for task-scoped external Markdown knowledge discovery."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "discover_external_knowledge.py"
GOAL = ROOT / "skills" / "goal" / "SKILL.md"
REPAIR = ROOT / "skills" / "repair" / "SKILL.md"
POLICY = ROOT / "references" / "external_knowledge_policy.md"


def invoke(root: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(completed.stdout)


def test_discovery_finds_nested_markdown_in_stable_order_and_optional_index(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "nested").mkdir(parents=True)
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    (root / "nested" / "z.markdown").write_text("z\n", encoding="utf-8")
    (root / "nested" / "a.md").write_text("a\n", encoding="utf-8")
    (root / "notes.txt").write_text("ignore\n", encoding="utf-8")

    completed, result = invoke(root)

    assert completed.returncode == 0
    assert result["status"] == "PASS"
    assert result["reason_code"] == "EXTERNAL_KNOWLEDGE_DISCOVERED"
    assert result["index"] == "index.md"
    assert result["files"] == ["index.md", "nested/a.md", "nested/z.markdown"]
    assert result["markdown_count"] == 3
    assert result["errors"] == []


def test_discovery_accepts_markdown_without_index(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "topics").mkdir(parents=True)
    (root / "topics" / "runtime.md").write_text("runtime\n", encoding="utf-8")

    completed, result = invoke(root)

    assert completed.returncode == 0
    assert result["status"] == "PASS"
    assert result["index"] is None
    assert result["files"] == ["topics/runtime.md"]


def test_discovery_blocks_invalid_empty_and_non_markdown_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    regular_file = tmp_path / "knowledge.md"
    regular_file.write_text("not a root\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    no_markdown = tmp_path / "no-markdown"
    no_markdown.mkdir()
    (no_markdown / "notes.txt").write_text("not knowledge\n", encoding="utf-8")

    for target in (missing, regular_file, empty, no_markdown):
        completed, result = invoke(target)
        assert completed.returncode == 2
        assert result["status"] == "BLOCKED"
        assert result["reason_code"] == "BLOCKED_EXTERNAL_KNOWLEDGE"
        assert result["files"] == []
        assert result["errors"]


def test_discovery_requires_an_absolute_root(tmp_path: Path) -> None:
    del tmp_path
    completed, result = invoke(Path("relative-knowledge-root"))

    assert completed.returncode == 2
    assert result["status"] == "BLOCKED"
    assert "must be absolute" in result["errors"][0]


def test_discovery_does_not_follow_symlinks_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "inside.md").write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.md").write_text("outside\n", encoding="utf-8")
    try:
        (root / "escape.md").symlink_to(outside / "outside.md")
        (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")

    completed, result = invoke(root)

    assert completed.returncode == 0
    assert result["files"] == ["inside.md"]


def test_discovery_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "nested").mkdir(parents=True)
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    (root / "nested" / "topic.md").write_text("topic\n", encoding="utf-8")
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    completed, _result = invoke(root)

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert completed.returncode == 0
    assert after == before
    assert not (ROOT / ".npu-agent" / "knowledge").exists()


def test_entrypoints_document_external_root_and_shared_boundaries() -> None:
    for path in (GOAL, REPAIR):
        text = path.read_text(encoding="utf-8")
        assert "external_knowledge_root: /absolute/path/to/kb" in text
        assert "--knowledge-root /absolute/path/to/kb" in text
        assert "discover_external_knowledge.py" in text
        assert "任务范围内" in text
        assert "不可信" in text
        assert "真实 NPU" in text

    goal = GOAL.read_text(encoding="utf-8")
    assert "传递精确根目录和 helper JSON 结果" in goal

    policy = POLICY.read_text(encoding="utf-8")
    for phrase in (
        "任务级补充",
        "普通 `.md` 和 `.markdown` 文件",
        "BLOCKED_EXTERNAL_KNOWLEDGE",
        "不遍历目录 symlink",
        "不替换内置主题",
        "不可信参考数据",
        "不要创建 `.npu-agent/knowledge`",
    ):
        assert phrase in policy
