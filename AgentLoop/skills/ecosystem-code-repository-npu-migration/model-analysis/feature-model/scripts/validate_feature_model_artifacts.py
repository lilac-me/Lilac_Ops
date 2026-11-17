#!/usr/bin/env python3
"""Validate the two model-authored feature-model files without rewriting them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any


FEATURE_MODEL_SCHEMA_VERSION = "npu-feature-model-v2"
COVERAGE_PLAN_SCHEMA_VERSION = "npu-coverage-plan-v2"
VALID_ENTRY_TYPES = {"unit_test", "e2e_example", "script", "doc_command", "other"}
EXECUTABLE_ENTRY_TYPES = {"unit_test", "e2e_example", "script", "doc_command"}
REPRESENTATIVE_ENTRY_TYPES = {"e2e_example", "script", "doc_command"}
VALID_COVERAGE_MODES = {"representative", "adaptive", "exhaustive"}
VALID_SCOPE_DISPOSITIONS = {"OUT_OF_SCOPE", "MANUAL", "BLOCKED"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
FORBIDDEN_CONTROL_KEYS = {
    "queue", "cursor", "attempt", "attempts", "state", "execution_status",
    "validation_status", "validation_disposition", "support_status", "final_status", "status",
}


def read_object(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing semantic output: {path}")
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"semantic output must be an object: {path}")
        return {}
    return value


def objects(doc: dict[str, Any], key: str, errors: list[str]) -> list[dict[str, Any]]:
    value = doc.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be an array")
        return []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{key}[{index}] must be an object")
        else:
            result.append(item)
    return result


def text(row: dict[str, Any], key: str, context: str, errors: list[str]) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context} is missing non-empty {key}")
        return ""
    return value.strip()


def strings(row: dict[str, Any], key: str, context: str, errors: list[str], *, nonempty: bool = False) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        errors.append(f"{context}.{key} must be an array")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{key}[{index}] must be a non-empty string")
        else:
            result.append(item.strip())
    if nonempty and not result:
        errors.append(f"{context}.{key} must not be empty")
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        errors.append(f"{context}.{key} contains duplicates: {', '.join(duplicates)}")
    return result


def index_by(rows: list[dict[str, Any]], key: str, label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        ident = text(row, key, f"{label}[{index}]", errors)
        if not ident:
            continue
        if ident in result:
            errors.append(f"duplicate {key}: {ident}")
        result[ident] = row
    return result


def find_forbidden_keys(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}" if path else key
            if key.lower() in FORBIDDEN_CONTROL_KEYS:
                errors.append(f"semantic outputs must not contain workflow or result field: {current}")
            find_forbidden_keys(item, current, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            find_forbidden_keys(item, f"{path}[{index}]", errors)


def valid_relative_path(value: str) -> bool:
    if not value or "\\" in value or value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return value not in {".", ".."} and ".." not in path.parts


def validate_summary(doc: dict[str, Any], errors: list[str]) -> None:
    summary = doc.get("repository_summary")
    if not isinstance(summary, dict):
        errors.append("repository_summary must be an object")
        return
    text(summary, "primary_purpose", "repository_summary", errors)
    strings(summary, "in_scope_roots", "repository_summary", errors)
    strings(summary, "inspected_sources", "repository_summary", errors, nonempty=True)
    text(summary, "entry_organization", "repository_summary", errors)


def validate_feature_model(doc: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if doc.get("schema_version") != FEATURE_MODEL_SCHEMA_VERSION:
        errors.append(f"feature_model schema_version must be {FEATURE_MODEL_SCHEMA_VERSION}")
    find_forbidden_keys(doc, "feature_model", errors)
    validate_summary(doc, errors)

    capability_rows = objects(doc, "capabilities", errors)
    feature_rows = objects(doc, "features", errors)
    entry_rows = objects(doc, "entries", errors)
    group_rows = objects(doc, "equivalence_groups", errors)
    capabilities = index_by(capability_rows, "capability_id", "capabilities", errors)
    features = index_by(feature_rows, "feature_id", "features", errors)
    entries = index_by(entry_rows, "entry_id", "entries", errors)
    groups = index_by(group_rows, "group_id", "equivalence_groups", errors)

    for ident, row in capabilities.items():
        context = f"capability {ident}"
        text(row, "name", context, errors)
        text(row, "description", context, errors)
        strings(row, "source_refs", context, errors, nonempty=True)

    for ident, row in features.items():
        context = f"feature {ident}"
        text(row, "name", context, errors)
        text(row, "category", context, errors)
        text(row, "description", context, errors)
        dimensions = row.get("dimensions")
        if not isinstance(dimensions, dict) or not dimensions:
            errors.append(f"{context}.dimensions must be a non-empty object")
        required = strings(row, "requires_capabilities", context, errors)
        for capability in required:
            if capability not in capabilities:
                errors.append(f"{context} references unknown capability: {capability}")
        strings(row, "source_refs", context, errors, nonempty=True)

    if not entries:
        errors.append("feature_model.entries must contain every in-scope entry")
    declared_group_entries: dict[str, set[str]] = {group_id: set() for group_id in groups}
    for ident, row in entries.items():
        context = f"entry {ident}"
        path = text(row, "path", context, errors)
        if path and not valid_relative_path(path):
            errors.append(f"{context}.path must be repository-relative: {path}")
        if path and ident != path and not ident.startswith(path + "#"):
            errors.append(f"{context}.entry_id must equal path or use <path>#<variant>")
        entry_type = text(row, "entry_type", context, errors)
        if entry_type and entry_type not in VALID_ENTRY_TYPES:
            errors.append(f"{context} has invalid entry_type: {entry_type}")
        if entry_type in EXECUTABLE_ENTRY_TYPES:
            text(row, "command", context, errors)
        feature_refs = strings(row, "feature_ids", context, errors)
        capability_refs = strings(row, "capability_ids", context, errors)
        if not feature_refs and not capability_refs:
            errors.append(f"{context} must reference at least one feature or capability")
        for feature in feature_refs:
            if feature not in features:
                errors.append(f"{context} references unknown feature: {feature}")
        for capability in capability_refs:
            if capability not in capabilities:
                errors.append(f"{context} references unknown capability: {capability}")
        text(row, "classification_reason", context, errors)
        group_id = row.get("equivalence_group")
        if group_id is not None and (not isinstance(group_id, str) or not group_id.strip()):
            errors.append(f"{context}.equivalence_group must be a non-empty string")
            group_id = ""
        group_id = str(group_id or "").strip()
        if feature_refs and entry_type != "unit_test" and not group_id:
            errors.append(f"{context} has features but no equivalence_group")
        if group_id:
            if group_id not in groups:
                errors.append(f"{context} references unknown equivalence group: {group_id}")
            else:
                declared_group_entries[group_id].add(ident)

    grouped_entries: set[str] = set()
    group_members: dict[str, set[str]] = {}
    for ident, row in groups.items():
        context = f"equivalence group {ident}"
        text(row, "name", context, errors)
        members = set(strings(row, "entry_ids", context, errors, nonempty=True))
        group_members[ident] = members
        for entry_id in members:
            if entry_id not in entries:
                errors.append(f"{context} references unknown entry: {entry_id}")
            if entry_id in grouped_entries:
                errors.append(f"entry belongs to multiple equivalence groups: {entry_id}")
            grouped_entries.add(entry_id)
        if members != declared_group_entries.get(ident, set()):
            errors.append(f"{context}.entry_ids do not match entries declaring this group")
        text(row, "equivalence_basis", context, errors)
        strings(row, "differentiators_checked", context, errors, nonempty=True)

    return {
        "capabilities": capabilities,
        "features": features,
        "entries": entries,
        "groups": groups,
        "group_members": group_members,
    }


def validate_coverage_plan(doc: dict[str, Any], model: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if doc.get("schema_version") != COVERAGE_PLAN_SCHEMA_VERSION:
        errors.append(f"coverage_plan schema_version must be {COVERAGE_PLAN_SCHEMA_VERSION}")
    find_forbidden_keys(doc, "coverage_plan", errors)
    mode = str(doc.get("coverage_mode") or "").strip().lower()
    if mode not in VALID_COVERAGE_MODES:
        errors.append(f"coverage_mode must be one of: {', '.join(sorted(VALID_COVERAGE_MODES))}")
    text(doc, "rationale", "coverage_plan", errors)

    entries = model["entries"]
    features = model["features"]
    capabilities = model["capabilities"]
    groups = model["groups"]
    group_members = model["group_members"]
    selected: set[str] = set()
    deferred: set[str] = set()
    scoped: set[str] = set()

    capability_rows = objects(doc, "capability_checks", errors)
    seen_capabilities: set[str] = set()
    for index, row in enumerate(capability_rows):
        context = f"capability_checks[{index}]"
        capability = text(row, "capability_id", context, errors)
        if capability in seen_capabilities:
            errors.append(f"duplicate capability check: {capability}")
        seen_capabilities.add(capability)
        if capability and capability not in capabilities:
            errors.append(f"{context} references unknown capability: {capability}")
        entry_ids = strings(row, "selected_entry_ids", context, errors, nonempty=True)
        for entry_id in entry_ids:
            entry = entries.get(entry_id)
            if entry is None:
                errors.append(f"{context} references unknown entry: {entry_id}")
                continue
            if entry.get("entry_type") != "unit_test":
                errors.append(f"{context} must select focused unit_test entries: {entry_id}")
            if capability and capability not in entry.get("capability_ids", []):
                errors.append(f"{entry_id} does not map to capability {capability}")
            selected.add(entry_id)
        text(row, "rationale", context, errors)

    representative_rows = objects(doc, "representative_groups", errors)
    planned_groups: set[str] = set()
    for index, row in enumerate(representative_rows):
        context = f"representative_groups[{index}]"
        group_id = text(row, "group_id", context, errors)
        if group_id in planned_groups:
            errors.append(f"duplicate representative group: {group_id}")
        planned_groups.add(group_id)
        if group_id and group_id not in groups:
            errors.append(f"{context} references unknown equivalence group: {group_id}")
        feature_ids = set(strings(row, "feature_ids", context, errors, nonempty=True))
        for feature_id in feature_ids:
            if feature_id not in features:
                errors.append(f"{context} references unknown feature: {feature_id}")
        members = set(strings(row, "member_entry_ids", context, errors, nonempty=True))
        representatives = set(strings(row, "representative_entry_ids", context, errors, nonempty=True))
        deferred_ids = set(strings(row, "deferred_entry_ids", context, errors))
        for entry_id in members:
            entry = entries.get(entry_id)
            if entry is not None and entry.get("entry_type") not in REPRESENTATIVE_ENTRY_TYPES:
                errors.append(
                    f"{context} members must be example, script or documented command entries: {entry_id}"
                )
        if group_id in group_members and members != group_members[group_id]:
            errors.append(f"{context}.member_entry_ids must match feature_model equivalence group members")
        if not representatives <= members:
            errors.append(f"{context} representatives must belong to the equivalence group")
        if not deferred_ids <= members:
            errors.append(f"{context} deferred entries must belong to the equivalence group")
        if representatives & deferred_ids:
            errors.append(f"{context} cannot select and defer the same entry")
        if representatives | deferred_ids != members:
            errors.append(f"{context} selected and deferred entries must partition all group members")
        if mode == "exhaustive" and deferred_ids:
            errors.append(f"{context} cannot defer entries in exhaustive mode")
        member_features = {
            feature_id
            for entry_id in members
            for feature_id in entries.get(entry_id, {}).get("feature_ids", [])
        }
        if feature_ids != member_features:
            errors.append(f"{context}.feature_ids must match features mapped by group members")
        text(row, "selection_reason", context, errors)
        text(row, "verification_goal", context, errors)
        risk = text(row, "risk_level", context, errors).lower()
        if risk and risk not in VALID_RISK_LEVELS:
            errors.append(f"{context} has invalid risk_level: {risk}")
        selected.update(representatives)
        deferred.update(deferred_ids)

    scope_rows = objects(doc, "scope_decisions", errors)
    for index, row in enumerate(scope_rows):
        context = f"scope_decisions[{index}]"
        entry_id = text(row, "entry_id", context, errors)
        if entry_id in scoped:
            errors.append(f"duplicate scope decision: {entry_id}")
        scoped.add(entry_id)
        if entry_id and entry_id not in entries:
            errors.append(f"{context} references unknown entry: {entry_id}")
        disposition = text(row, "disposition", context, errors).upper()
        if disposition and disposition not in VALID_SCOPE_DISPOSITIONS:
            errors.append(f"{context} has invalid disposition: {disposition}")
        text(row, "reason", context, errors)

    for group_id, members in group_members.items():
        if group_id not in planned_groups and not members <= scoped:
            errors.append(f"equivalence group has neither a representative plan nor complete scope decisions: {group_id}")

    overlap = (selected & deferred) | (selected & scoped) | (deferred & scoped)
    if overlap:
        errors.append("entries have overlapping coverage decisions: " + ", ".join(sorted(overlap)))
    accounted = selected | deferred | scoped
    missing = set(entries) - accounted
    extra = accounted - set(entries)
    if missing:
        errors.append("entries missing coverage decisions: " + ", ".join(sorted(missing)))
    if extra:
        errors.append("coverage decisions reference unknown entries: " + ", ".join(sorted(extra)))
    return {
        "coverage_mode": mode,
        "selected_entry_ids": sorted(selected),
        "deferred_entry_ids": sorted(deferred),
        "scoped_entry_ids": sorted(scoped),
        "representative_group_count": len(planned_groups),
    }


def validate(repo: Path) -> dict[str, Any]:
    errors: list[str] = []
    agent = repo / ".npu-agent"
    feature_doc = read_object(agent / "feature_model.json", errors)
    coverage_doc = read_object(agent / "coverage_plan.json", errors)
    model = validate_feature_model(feature_doc, errors)
    coverage = validate_coverage_plan(coverage_doc, model, errors)
    return {
        "schema_version": "npu-feature-model-validation-v2",
        "status": "PASS" if not errors else "BLOCKED",
        "reason_code": "FEATURE_MODEL_VALID" if not errors else "FEATURE_MODEL_INVALID",
        "capability_count": len(model["capabilities"]),
        "feature_count": len(model["features"]),
        "entry_count": len(model["entries"]),
        "equivalence_group_count": len(model["groups"]),
        "coverage": coverage,
        "errors": errors,
    }


def self_test() -> int:
    import tempfile

    skill_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="npu-feature-model-") as tmp:
        repo = Path(tmp)
        agent = repo / ".npu-agent"
        agent.mkdir()
        for name in ("feature_model.json", "coverage_plan.json"):
            (agent / name).write_bytes((skill_root / "examples" / name).read_bytes())
        before = {path.name: path.read_bytes() for path in agent.iterdir()}
        result = validate(repo)
        after = {path.name: path.read_bytes() for path in agent.iterdir()}
        if result["status"] != "PASS" or before != after:
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
        coverage = json.loads((agent / "coverage_plan.json").read_text(encoding="utf-8"))
        coverage["representative_groups"][0]["representative_entry_ids"] = ["examples/sft_full.py"]
        (agent / "coverage_plan.json").write_text(json.dumps(coverage), encoding="utf-8")
        if validate(repo)["status"] != "BLOCKED":
            return 1
    print("SELF_TEST_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    result = validate(Path(args.repo).resolve())
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
