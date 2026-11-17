"""Skill routing and recursive resource checks for the ecosystem-code-repository-npu-migration plugin."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_skill_design.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("skill_design_validator", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positive_design() -> None:
    validator = load_validator()
    result = validator.validate(ROOT)
    assert result["status"] == "PASS", json.dumps(result, indent=2)
    assert result["skill_count"] == 8
    assert {item["name"] for item in result["skills"]} == {
        "docs-support", "env-setup", "feature-model", "goal", "local-assets",
        "repair", "report-pr", "vllm-ascend-env",
    }
    assert all(not item["alias"] for item in result["skills"])
    for item in result["skills"]:
        assert item["domain"] == "ecosystem-code-repository-npu-migration", item["name"]
        assert item["function"] == validator.SKILL_FUNCTION[item["name"]], item["name"]
        assert validator.SEMVER.fullmatch(item["version"]), item["name"]
    assert not (ROOT / "skills" / "custom-workflow").exists()
    assert not (ROOT / "skills" / "test-repair").exists()
    assert not (ROOT / "runtime").exists()


def test_missing_function_field_blocks(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    shutil.copytree(ROOT, root)
    skill = root / "skills" / "docs-support" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    assert "function: common" in text
    skill.write_text(text.replace("function: common\n", ""), encoding="utf-8")
    result = load_validator().validate(root)
    assert result["status"] == "BLOCKED"
    assert any("frontmatter missing required fields: function" in error for error in result["errors"])


def test_recursive_markdown_path_negative(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    shutil.copytree(ROOT, root)
    policy = root / "skills" / "env-setup" / "references" / "capability_install_policy.md"
    policy.write_text(policy.read_text(encoding="utf-8") + "\n[bad](../vllm-ascend-env/SKILL.md)\n", encoding="utf-8")
    result = load_validator().validate(root)
    assert result["status"] == "BLOCKED"
    assert any("capability_install_policy.md" in error for error in result["errors"])


def test_heading_prefixes_do_not_satisfy_required_semantics(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    shutil.copytree(ROOT, root)
    skill = root / "skills" / "docs-support" / "SKILL.md"
    skill.write_text(
        """---
name: docs-support
description: This skill should be used when the user asks to test heading validation.
---

# 标题校验

## 角色扮演技巧

占位内容。

## 工作流程图

占位内容。

## 边界值测试

占位内容。

## 完成条件格式

占位内容。

## 资源消耗

占位内容。
""",
        encoding="utf-8",
    )

    result = load_validator().validate(root)

    assert result["status"] == "BLOCKED"
    for semantic in ("role", "routing", "boundary", "completion", "resources"):
        assert any(f"missing {semantic} semantics" in error for error in result["errors"])


def test_no_public_skill_declares_legacy_alias_language() -> None:
    for skill in (ROOT / "skills").iterdir():
        if not skill.is_dir():
            continue
        text = (skill / "SKILL.md").read_text(encoding="utf-8").lower()
        assert "compatibility alias" not in text
        assert "deprecated alias" not in text
