#!/usr/bin/env python3
"""Validate VeOmni benchmark scenarios without launching a training job.

This runner checks that each declarative scenario is internally consistent,
references an existing fixture, and describes only supported workflow stages.
It intentionally does not download models, install dependencies, or launch
VeOmni/NPU training. Those operations must be explicitly authorized.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by command-line environments
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
WORKFLOW_STAGES = (
    "environment",
    "model-analysis",
    "adaptation",
    "functional-validation",
    "parallel-recipe",
    "precision-alignment",
    "mfu-calculation",
    "performance-tuning",
    "delivery",
)
MODEL_TYPES = {"dense", "moe", "vlm", "omni", "dit", "rl"}


def load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError(
            "缺少 PyYAML；请先执行 `python -m pip install -r "
            "benchmarks/veomni/requirements.txt`。"
        )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stage_names(stages: Any) -> tuple[list[str], list[str]]:
    """Return stage names and validation errors from a scenario stages list."""
    errors: list[str] = []
    names: list[str] = []
    if not isinstance(stages, list) or not stages:
        return names, ["stages 必须是非空列表"]
    for index, item in enumerate(stages):
        if not isinstance(item, dict) or len(item) != 1:
            errors.append(f"stages[{index}] 必须是仅含一个阶段名的对象")
            continue
        name = next(iter(item))
        if name not in WORKFLOW_STAGES:
            errors.append(f"stages[{index}] 包含未知阶段: {name}")
        elif name in names:
            errors.append(f"stages[{index}] 重复阶段: {name}")
        else:
            names.append(name)
    return names, errors


def validate_normal_scenario(path: Path, payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["场景根节点必须是对象"]
    for field in ("scenario", "model", "model_type", "priority", "trainer", "parallel", "stages", "fixtures"):
        if field not in payload:
            errors.append(f"缺少必填字段: {field}")
    if payload.get("model_type") not in MODEL_TYPES:
        errors.append(f"model_type 不受支持: {payload.get('model_type')!r}")
    if not isinstance(payload.get("parallel"), dict):
        errors.append("parallel 必须是对象")
    names, stage_errors = stage_names(payload.get("stages"))
    errors.extend(stage_errors)
    if names and names[0] != "environment":
        errors.append("第一个阶段必须为 environment")
    fixture = payload.get("fixtures", {}).get("config") if isinstance(payload.get("fixtures"), dict) else None
    if not isinstance(fixture, str) or not fixture:
        errors.append("fixtures.config 必须指定 fixture 路径")
    else:
        fixture_path = REPO_ROOT / fixture
        if not fixture_path.is_file():
            errors.append(f"fixture 不存在: {fixture}")
        else:
            try:
                fixture_payload = load_yaml(fixture_path)
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - exact parser messages vary
                errors.append(f"fixture 无法解析: {fixture} ({exc})")
            else:
                if not isinstance(fixture_payload, dict):
                    errors.append(f"fixture 根节点必须是对象: {fixture}")
    return errors


def validate_exception_scenarios(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        return ["异常场景必须包含 scenarios 列表"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["scenarios"]):
        prefix = f"scenarios[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} 必须是对象")
            continue
        for field in ("id", "name", "trigger", "expect"):
            if field not in item:
                errors.append(f"{prefix} 缺少字段: {field}")
        identifier = item.get("id")
        if isinstance(identifier, str):
            if identifier in seen:
                errors.append(f"{prefix} 重复 id: {identifier}")
            seen.add(identifier)
        expect = item.get("expect")
        if not isinstance(expect, dict):
            errors.append(f"{prefix}.expect 必须是对象")
            continue
        if expect.get("stage") not in WORKFLOW_STAGES:
            errors.append(f"{prefix}.expect.stage 不受支持: {expect.get('stage')!r}")
        if expect.get("status") not in {"FAILED", "BLOCKED", "NEEDS_HUMAN"}:
            errors.append(f"{prefix}.expect.status 不受支持: {expect.get('status')!r}")
        if not isinstance(expect.get("output"), str) or not expect["output"].strip():
            errors.append(f"{prefix}.expect.output 必须是非空字符串")
    return errors


def validate_scenario(path: Path) -> list[str]:
    try:
        payload = load_yaml(path)
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - exact parser messages vary
        return [f"YAML 无法解析: {exc}"]
    if path.name == "exception-cases.yaml":
        return validate_exception_scenarios(payload)
    return validate_normal_scenario(path, payload)


def resolve_scenarios(selected: list[str]) -> list[Path]:
    if not selected:
        return sorted(SCENARIO_DIR.glob("*.yaml"))
    paths: list[Path] = []
    for value in selected:
        path = Path(value)
        paths.append(path if path.is_absolute() else SCENARIO_DIR / path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="VeOmni 场景与 fixture 静态验收")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="场景文件名或绝对路径；可重复。默认校验全部场景。",
    )
    args = parser.parse_args()
    try:
        paths = resolve_scenarios(args.scenario)
        failures = 0
        for path in paths:
            if not path.is_file():
                print(f"[FAIL] {path}: 场景文件不存在")
                failures += 1
                continue
            errors = validate_scenario(path)
            if errors:
                print(f"[FAIL] {path}")
                for error in errors:
                    print(f"       - {error}")
                failures += 1
            else:
                print(f"[PASS] {path}")
    except RuntimeError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(f"\n场景静态验收: {len(paths) - failures}/{len(paths)} 通过")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
