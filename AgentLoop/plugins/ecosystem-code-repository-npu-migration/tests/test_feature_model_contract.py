"""Feature-model prompt and two-file read-only validation contracts."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "feature-model"
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_feature_model_artifacts.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("feature_validator", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_examples(root: Path) -> None:
    agent = root / ".npu-agent"
    agent.mkdir()
    for name in ("feature_model.json", "coverage_plan.json"):
        shutil.copyfile(SKILL_ROOT / "examples" / name, agent / name)


def read_json(root: Path, name: str) -> dict:
    return json.loads((root / ".npu-agent" / name).read_text(encoding="utf-8"))


def write_json(root: Path, name: str, value: dict) -> None:
    (root / ".npu-agent" / name).write_text(json.dumps(value), encoding="utf-8")


def validate(root: Path) -> dict:
    return load_validator().validate(root)


def test_skill_explains_repository_map_and_representative_deduplication() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for phrase in (
        "从整体上阅读仓库",
        "用户可见功能",
        "NPU 敏感维度",
        "equivalence group",
        "最小仓库原生目标",
        "feature_model.json",
        "coverage_plan.json",
        "sft_lora_tiny.py",
        "sft_lora_large.py",
        "full 与 LoRA",
        "dense 与 MoE",
        "单设备与分布式",
        "Hugging Face 与 Megatron",
        "[操作示例](references/worked_example.md)",
        "[字段 schema](schemas/feature_model.schema.json)",
        "[完整示例](examples/feature_model.json)",
        "[字段 schema](schemas/coverage_plan.schema.json)",
        "[完整示例](examples/coverage_plan.json)",
        "必须作为配对文件阅读",
        "每次仓库级迁移",
        "将发现的文件和命令与最终 entry 列表比对",
        "所有可行的迁移相关 unit/component test",
        "只放示例、脚本或文档启动命令",
        "仓库原生 `command`",
    ):
        assert phrase in text
    assert "discover_entrypoints.py" not in text
    assert "entrypoint_inventory.json" not in text


def test_examples_validate_and_validator_is_read_only(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    before = {path.name: path.read_bytes() for path in (tmp_path / ".npu-agent").iterdir()}
    result = validate(tmp_path)
    after = {path.name: path.read_bytes() for path in (tmp_path / ".npu-agent").iterdir()}
    assert result["status"] == "PASS", result
    assert result["entry_count"] == 5
    assert result["coverage"]["representative_group_count"] == 3
    assert before == after


def test_validator_rejects_duplicate_and_unstable_entry_ids(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    model = read_json(tmp_path, "feature_model.json")
    duplicate = dict(model["entries"][0])
    model["entries"].append(duplicate)
    model["entries"][1]["entry_id"] = "example-0001"
    write_json(tmp_path, "feature_model.json", model)
    errors = validate(tmp_path)["errors"]
    assert any("duplicate entry_id" in error for error in errors)
    assert any("must equal path or use <path>#<variant>" in error for error in errors)


def test_validator_rejects_unknown_feature_capability_and_group(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    model = read_json(tmp_path, "feature_model.json")
    model["features"][0]["requires_capabilities"] = ["unknown-capability"]
    model["entries"][1]["feature_ids"] = ["unknown-feature"]
    model["entries"][1]["equivalence_group"] = "unknown-group"
    write_json(tmp_path, "feature_model.json", model)
    errors = validate(tmp_path)["errors"]
    assert any("unknown capability" in error for error in errors)
    assert any("unknown feature" in error for error in errors)
    assert any("unknown equivalence group" in error for error in errors)


def test_validator_rejects_representative_outside_group(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["representative_groups"][0]["representative_entry_ids"] = ["examples/sft_full.py"]
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("representatives must belong" in error for error in errors)


def test_validator_rejects_selected_entry_without_command(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    model = read_json(tmp_path, "feature_model.json")
    model["entries"][0].pop("command")
    write_json(tmp_path, "feature_model.json", model)
    errors = validate(tmp_path)["errors"]
    assert any("entry tests/test_npu_device.py is missing non-empty command" in error for error in errors)


def test_validator_rejects_unit_test_as_example_representative(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    model = read_json(tmp_path, "feature_model.json")
    model["entries"][1]["entry_type"] = "unit_test"
    write_json(tmp_path, "feature_model.json", model)
    errors = validate(tmp_path)["errors"]
    assert any("members must be example, script or documented command entries" in error for error in errors)


def test_validator_rejects_example_in_capability_checks(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["capability_checks"][0]["selected_entry_ids"] = ["examples/sft_lora_tiny.py"]
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("must select focused unit_test entries" in error for error in errors)


def test_validator_rejects_deferred_entry_from_another_group(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["representative_groups"][0]["deferred_entry_ids"] = ["examples/sft_full.py"]
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("deferred entries must belong" in error for error in errors)


def test_validator_rejects_overlapping_and_missing_coverage_decisions(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    first = coverage["representative_groups"][0]
    first["deferred_entry_ids"] = ["examples/sft_lora_tiny.py"]
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("cannot select and defer" in error for error in errors)
    assert any("entries missing coverage decisions" in error for error in errors)


def test_validator_rejects_group_without_plan_or_scope_decisions(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["representative_groups"] = coverage["representative_groups"][1:]
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("neither a representative plan nor complete scope decisions" in error for error in errors)


def test_validator_accepts_explicit_scope_decisions_for_an_unplanned_group(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["representative_groups"] = coverage["representative_groups"][1:]
    coverage["scope_decisions"] = [
        {"entry_id": entry_id, "disposition": "BLOCKED", "reason": "Required assets are unavailable."}
        for entry_id in ("examples/sft_lora_tiny.py", "examples/sft_lora_large.py")
    ]
    write_json(tmp_path, "coverage_plan.json", coverage)
    assert validate(tmp_path)["status"] == "PASS"


def test_validator_rejects_workflow_or_result_fields(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    coverage = read_json(tmp_path, "coverage_plan.json")
    coverage["status"] = "PASS"
    write_json(tmp_path, "coverage_plan.json", coverage)
    errors = validate(tmp_path)["errors"]
    assert any("must not contain workflow or result field" in error for error in errors)


def test_legacy_artifacts_and_discovery_generator_are_removed() -> None:
    legacy_names = (
        "entry_feature" + "_map.json",
        "verification_dependency" + "_graph.json",
        "discover_" + "entrypoints.py",
    )
    assert not (SKILL_ROOT / "scripts" / legacy_names[2]).exists()
    for directory in (SKILL_ROOT / "examples", SKILL_ROOT / "schemas"):
        assert not (directory / legacy_names[0]).exists()
        assert not (directory / legacy_names[1]).exists()
    active_files = [SKILL_ROOT / "SKILL.md", *SKILL_ROOT.glob("references/*.md")]
    for path in active_files:
        content = path.read_text(encoding="utf-8")
        assert all(name not in content for name in legacy_names)
