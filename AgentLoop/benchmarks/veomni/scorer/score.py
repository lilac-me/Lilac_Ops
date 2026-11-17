#!/usr/bin/env python3
"""VeOmni 阶段门禁评分器。

按 plugins/veomni-agent/veomni.yaml 流水线检查 outputs/<model>/ 产物：
产物存在、非空（min_bytes）、通过 Schema 校验，输出各阶段状态与总分。

用法:
    python benchmarks/veomni/scorer/score.py outputs/<model>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))

from evaluate import YamlUnavailable, find_artifact, load_artifact, load_json, validate  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

# 阶段 → (产物, min_bytes, schema)；与 veomni.yaml pipeline.gate 对齐
# performance_baseline 仅做存在性 + 大小门禁，数据完整性由
# performance_report.json schema 的必填 baseline 字段保证
STAGES = [
    ("environment",          [("environment_report.json", 200, True)]),
    ("model-analysis",       [("model_inventory.json", 200, True)]),
    ("adaptation",           [("patch_manifest.json", 200, True)]),
    ("functional-validation", [("functional_report.json", 200, True)]),
    ("parallel-recipe",      [("parallel_recipe.yaml", 200, True)]),
    ("precision-alignment",  [("golden_manifest.json", 200, True),
                              ("accuracy_report.json", 200, True)]),
    ("mfu-calculation",      [("mfu_report.json", 200, True)]),
    ("performance-tuning",   [("performance_baseline.json", 200, False),
                              ("performance_report.json", 200, True)]),
    ("delivery",             [("acceptance_report.json", 300, True),
                              ("delivery_coverage_report.md", 800, False),
                              ("migration_process_record.md", 600, False)]),
]
STAGE_INDEX = {name: artifacts for name, artifacts in STAGES}


def load_scenario_stages(path: Path) -> list[str]:
    """Return the explicitly declared stages for a normal scenario YAML."""
    try:
        payload: Any = load_artifact(path)
    except YamlUnavailable as exc:
        raise RuntimeError("需要 PyYAML 才能按场景评分") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        raise ValueError("场景必须包含 stages 列表；异常场景不能用于产物评分")
    names: list[str] = []
    for index, item in enumerate(payload["stages"]):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"stages[{index}] 必须是仅含一个阶段名的对象")
        name = next(iter(item))
        if name not in STAGE_INDEX:
            raise ValueError(f"stages[{index}] 是未知阶段: {name}")
        if name in names:
            raise ValueError(f"stages[{index}] 重复阶段: {name}")
        names.append(name)
    return names


def check_stage(base: Path, artifacts: list) -> dict:
    result = {"artifacts": [], "status": "PASSED"}
    for name, min_bytes, need_schema in artifacts:
        entry = {"artifact": name}
        path = base / name
        if not path.exists():
            path = find_artifact(base, name) or path
        if not path.exists():
            entry["status"] = "MISSING"
            result["status"] = "PENDING"
        elif path.stat().st_size < min_bytes:
            entry["status"] = "TOO_SMALL"
            result["status"] = "FAILED"
        else:
            if need_schema:
                schema = load_json(SCHEMA_DIR / f"{Path(name).stem}.json")
                try:
                    data = load_artifact(path)
                except YamlUnavailable:
                    entry["status"] = "SCHEMA_UNAVAILABLE"
                    result["status"] = "BLOCKED"
                else:
                    errs = validate(data, schema)
                    if errs:
                        entry["status"] = "SCHEMA_INVALID"
                        entry["errors"] = errs[:5]
                        result["status"] = "FAILED"
                    else:
                        entry["status"] = "OK"
            else:
                entry["status"] = "OK"
        result["artifacts"].append(entry)
        if entry["status"] in ("MISSING",) and result["status"] == "PENDING":
            continue
    # 所有产物缺失 → PENDING；任一失败 → FAILED
    if all(a["status"] == "MISSING" for a in result["artifacts"]):
        result["status"] = "PENDING"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="VeOmni 阶段门禁评分")
    parser.add_argument("output_dir", type=Path, help="outputs/<model>/ 产物目录")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="只评分该 YAML 场景声明的阶段；省略时评分完整 9 阶段流水线。",
    )
    parser.add_argument("--json-out", type=Path, default=None,
                        help="将评分结果写入 JSON（如最终交付前的门禁预检）")
    args = parser.parse_args()

    base = args.output_dir
    if not base.is_dir():
        print(f"[FAIL] 产物目录不存在: {base}")
        return 2

    try:
        selected_stages = load_scenario_stages(args.scenario) if args.scenario else [name for name, _ in STAGES]
    except (RuntimeError, ValueError) as exc:
        print(f"[FAIL] 无法加载场景: {exc}")
        return 2

    passed_count = 0
    report = {"output_dir": str(base), "stages": []}
    for stage in selected_stages:
        artifacts = STAGE_INDEX[stage]
        result = check_stage(base, artifacts)
        result["stage"] = stage
        report["stages"].append(result)
        mark = {"PASSED": "[PASS]", "FAILED": "[FAIL]", "BLOCKED": "[BLOCK]",
                "PENDING": "[PEND]"}[result["status"]]
        if result["status"] == "PASSED":
            passed_count += 1
        print(f"{mark} {stage:<24} {result['status']}")
        for a in result["artifacts"]:
            detail = ""
            if a["status"] == "SCHEMA_INVALID":
                detail = " :: " + "; ".join(a.get("errors", []))
            elif a["status"] == "SCHEMA_UNAVAILABLE":
                detail = " :: 需 PyYAML（pip install pyyaml）"
            print(f"         - {a['artifact']:<28} {a['status']}{detail}")

    total = len(selected_stages)
    print(f"\n评分: {passed_count}/{total} 阶段通过")
    report["passed"] = passed_count
    report["total"] = total
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"评分结果已写入: {args.json_out}")
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
