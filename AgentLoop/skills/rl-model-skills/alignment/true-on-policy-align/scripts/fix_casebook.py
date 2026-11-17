#!/usr/bin/env python3
"""Validate, explicitly promote, and search compact verified alignment fix cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 2
MAX_CASE_BYTES = 256 * 1024
MAX_STRING_LENGTH = 4096
ROOT_CAUSES = {"Op path", "Input precision", "Config", "Missing step", "Framework bug"}
METHODS = {"config", "runtime_monkey_patch", "source_patch", "weight_or_reshard_patch"}
FORBIDDEN_KEYS = {
    "raw_tensor",
    "tensor_data",
    "raw_dump",
    "dump_contents",
    "prompt_text",
    "token_text",
    "source_code",
    "patch_body",
    "credentials",
}
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
SHA256_VALUE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class CaseError(ValueError):
    pass


def default_casebook_root() -> Path:
    return Path(__file__).resolve().parent.parent / "references" / "fix-cases"


def load_json(path: Path, *, size_limit: int | None = None) -> Any:
    try:
        if size_limit is not None and path.stat().st_size > size_limit:
            raise CaseError(f"JSON exceeds {size_limit} bytes: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except CaseError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseError(f"Cannot read JSON {path}: {exc}") from exc


def req_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise CaseError(f"{key} must be an object")
    return value


def req_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CaseError(f"{key} must be a non-empty string")
    return value.strip()


def req_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise CaseError(f"{key} must be a boolean")
    return value


def req_strings(parent: dict[str, Any], key: str, *, empty: bool = False) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CaseError(f"{key} must be a list of non-empty strings")
    if not empty and not value:
        raise CaseError(f"{key} may not be empty")
    return [item.strip() for item in value]


def walk_payload(value: Any, path: str = "$", key_name: str | None = None) -> None:
    if key_name and key_name.lower() in FORBIDDEN_KEYS:
        raise CaseError(f"forbidden raw/sensitive field at {path}: {key_name}")
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise CaseError(f"string exceeds {MAX_STRING_LENGTH} characters at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            walk_payload(child, f"{path}.{key}", str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_payload(child, f"{path}[{index}]")


def validate_relative_artifact(raw: str) -> None:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", normalized):
        raise CaseError("provenance.patch_artifact must be repository/package-relative")
    if ".." in path.parts or any(token in normalized for token in ("*", "?", "[", "]")):
        raise CaseError("provenance.patch_artifact must be a fixed relative path")


def validate_model(case: dict[str, Any]) -> None:
    model = req_dict(case, "model")
    for key in ("family", "architecture", "revision", "attention_type", "structure", "quantization"):
        req_string(model, key)


def validate_topology(case: dict[str, Any]) -> None:
    topology = req_dict(case, "topology")
    for key in ("tp", "pp", "ep"):
        value = topology.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CaseError(f"topology.{key} must be a positive integer")
    req_string(topology, "notes")


def validate_diagnosis(case: dict[str, Any]) -> None:
    diagnosis = req_dict(case, "diagnosis")
    req_string(diagnosis, "first_divergence_proof")
    hashes = req_strings(diagnosis, "evidence_hashes")
    for value in hashes:
        if not SHA256_VALUE.match(value):
            raise CaseError(f"diagnosis.evidence_hashes must contain full sha256 values: {value}")


def validate_repair(case: dict[str, Any]) -> None:
    repair = req_dict(case, "repair")
    if req_string(repair, "method") not in METHODS:
        raise CaseError("repair.method is unsupported")
    req_string(repair, "mathematical_delta")
    req_strings(repair, "interface_invariants")
    req_strings(repair, "applicability_guards")
    req_string(repair, "implementation_summary")
    if not req_bool(repair, "rederived_for_current_model"):
        raise CaseError("case repair must be re-derived for the current model")
    if req_bool(repair, "copied_historical_patch"):
        raise CaseError("copied historical/model-specific patches cannot enter the casebook")


def validate_verification(case: dict[str, Any], lifecycle: str) -> None:
    verification = req_dict(case, "verification")
    req_strings(verification, "regressions", empty=True)
    if lifecycle == "candidate":
        return
    exact_bools = (
        "target_native_exact",
        "same_shape",
        "same_dtype",
        "no_nonfinite",
        "torch_equal",
        "downstream_checked",
    )
    for key in exact_bools:
        if verification.get(key) is not True:
            raise CaseError(f"verified case requires verification.{key}=true")
    if verification.get("max_abs_diff") != 0:
        raise CaseError("verified case requires verification.max_abs_diff=0")
    upstream_replays = req_strings(verification, "upstream_replay_ids", empty=True)
    _ = upstream_replays
    prefill_statuses = {"exact", "nonzero", "not_run"}
    if req_string(verification, "final_native_prefill_status") not in prefill_statuses:
        raise CaseError("verification.final_native_prefill_status is invalid")
    if req_string(verification, "final_conditional_prefill_status") not in prefill_statuses:
        raise CaseError("verification.final_conditional_prefill_status is invalid")
    e2e_status = req_string(verification, "e2e_metric_status")
    if e2e_status not in {"zero", "nonzero", "not_run"}:
        raise CaseError("verification.e2e_metric_status is invalid")
    if e2e_status == "not_run":
        if verification.get("e2e_metric_value") is not None:
            raise CaseError("not_run E2E status requires e2e_metric_value=null")
    else:
        req_string(verification, "e2e_metric_name")
        value = verification.get("e2e_metric_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CaseError("verification.e2e_metric_value must be numeric")
        if e2e_status == "zero" and value != 0:
            raise CaseError("zero E2E status requires e2e_metric_value=0")
        if e2e_status == "nonzero" and value == 0:
            raise CaseError("nonzero E2E status requires a nonzero metric value")
        if not SHA256_VALUE.match(req_string(verification, "e2e_evidence")):
            raise CaseError("verification.e2e_evidence must be a full sha256 value")
    if verification["regressions"]:
        raise CaseError("a verified case may not contain known regressions")


def validate_provenance(case: dict[str, Any], lifecycle: str) -> None:
    provenance = req_dict(case, "provenance")
    repositories = provenance.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise CaseError("provenance.repositories must be a non-empty list")
    for index, repository in enumerate(repositories):
        if not isinstance(repository, dict):
            raise CaseError(f"provenance.repositories[{index}] must be an object")
        for key in ("name", "baseline", "tested"):
            req_string(repository, key)
    recorded = req_string(provenance, "recorded_at_utc")
    if not ISO_UTC.match(recorded):
        raise CaseError("provenance.recorded_at_utc must be ISO-8601 UTC ending in Z")
    if lifecycle == "verified":
        verified = req_string(provenance, "verified_at_utc")
        if not ISO_UTC.match(verified):
            raise CaseError("provenance.verified_at_utc must be ISO-8601 UTC ending in Z")
    req_string(provenance, "session_id")
    artifact = req_string(provenance, "patch_artifact")
    validate_relative_artifact(artifact)


def validate_storage(case: dict[str, Any]) -> None:
    storage = req_dict(case, "storage")
    for key in (
        "raw_tensor_included",
        "raw_dump_included",
        "prompt_included",
        "full_source_or_patch_body_included",
    ):
        if req_bool(storage, key):
            raise CaseError(f"storage.{key} must be false")


def validate_generalization(case: dict[str, Any], lifecycle: str) -> None:
    generalization = req_dict(case, "generalization")
    level = req_string(generalization, "level")
    expected = "candidate" if lifecycle == "candidate" else "single_model_verified"
    if level != expected:
        raise CaseError(f"{lifecycle} case requires generalization.level={expected}")
    req_strings(generalization, "limitations")
    req_strings(generalization, "independent_case_ids", empty=True)


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def fingerprint(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_ids(case: dict[str, Any]) -> tuple[str, str]:
    repair = case["repair"]
    model = case["model"]
    pattern_id = fingerprint(
        {
            "root_cause_class": case["root_cause_class"],
            "pattern": normalize_text(case["pattern"]),
            "semantic_boundary": normalize_text(case["semantic_boundary"]),
            "mathematical_delta": normalize_text(repair["mathematical_delta"]),
        }
    )
    case_id = fingerprint(
        {
            "pattern_fingerprint": pattern_id,
            "model": model,
            "topology": case["topology"],
            "repositories": case["provenance"]["repositories"],
            "patch_artifact": case["provenance"]["patch_artifact"],
        }
    )
    return case_id, pattern_id


def validate_case(case: Any) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise CaseError("case root must be an object")
    walk_payload(case)
    if case.get("schema_version") != SCHEMA_VERSION:
        raise CaseError(f"schema_version must be {SCHEMA_VERSION}")
    lifecycle = req_string(case, "lifecycle")
    if lifecycle not in {"candidate", "verified"}:
        raise CaseError("lifecycle must be candidate or verified")
    req_string(case, "title")
    if req_string(case, "root_cause_class") not in ROOT_CAUSES:
        raise CaseError("root_cause_class is unsupported")
    req_string(case, "pattern")
    req_string(case, "semantic_boundary")
    validate_model(case)
    validate_topology(case)
    validate_diagnosis(case)
    validate_repair(case)
    validate_verification(case, lifecycle)
    rejected = req_strings(case, "rejected_alternatives", empty=True)
    _ = rejected
    validate_provenance(case, lifecycle)
    validate_storage(case)
    validate_generalization(case, lifecycle)
    case_id, pattern_id = compute_ids(case)
    return {
        "valid": True,
        "lifecycle": lifecycle,
        "case_id": case_id,
        "pattern_fingerprint": pattern_id,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def load_index(root: Path) -> dict[str, Any]:
    path = root / "index.json"
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "cases": []}
    index = load_json(path)
    if not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION:
        raise CaseError(f"invalid casebook index: {path}")
    if not isinstance(index.get("cases"), list):
        raise CaseError(f"casebook index cases must be a list: {path}")
    return index


def command_validate(args: argparse.Namespace) -> int:
    result = validate_case(load_json(args.case.resolve(), size_limit=MAX_CASE_BYTES))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_add(args: argparse.Namespace) -> int:
    if not args.confirm_maintainer_update:
        raise CaseError(
            "add requires --confirm-maintainer-update and an explicit --casebook-root; "
            "alignment tasks must not modify an installed Skill by default"
        )
    source = args.case.resolve()
    case = load_json(source, size_limit=MAX_CASE_BYTES)
    result = validate_case(case)
    if result["lifecycle"] != "verified":
        raise CaseError("only verified cases may be added to the global casebook")
    root = args.casebook_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stored = copy.deepcopy(case)
    stored["case_id"] = result["case_id"]
    stored["pattern_fingerprint"] = result["pattern_fingerprint"]
    case_path = root / f"{result['case_id']}.json"
    if case_path.exists():
        existing = load_json(case_path, size_limit=MAX_CASE_BYTES)
        if existing != stored:
            raise CaseError(f"case ID collision with different content: {case_path}")
        result["added"] = False
        result["case_path"] = str(case_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    atomic_write_json(case_path, stored)
    index = load_index(root)
    index["cases"].append(
        {
            "case_id": result["case_id"],
            "pattern_fingerprint": result["pattern_fingerprint"],
            "title": case["title"],
            "root_cause_class": case["root_cause_class"],
            "pattern": case["pattern"],
            "semantic_boundary": case["semantic_boundary"],
            "model_family": case["model"]["family"],
            "architecture": case["model"]["architecture"],
            "model_revision": case["model"]["revision"],
            "verified_at_utc": case["provenance"]["verified_at_utc"],
            "case_file": case_path.name,
        }
    )
    index["cases"] = sorted(index["cases"], key=lambda item: item["case_id"])
    atomic_write_json(root / "index.json", index)
    result["added"] = True
    result["case_path"] = str(case_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def contains(value: str, query: str | None) -> bool:
    return query is None or query.lower() in value.lower()


def command_search(args: argparse.Namespace) -> int:
    root = args.casebook_root.resolve()
    index = load_index(root)
    matches = []
    for entry in index["cases"]:
        if not contains(str(entry.get("root_cause_class", "")), args.root_cause):
            continue
        if not contains(str(entry.get("pattern", "")), args.pattern):
            continue
        if not contains(str(entry.get("architecture", "")), args.architecture):
            continue
        if not contains(str(entry.get("model_family", "")), args.model_family):
            continue
        matches.append(entry)
        if len(matches) >= args.limit:
            break
    print(
        json.dumps(
            {"casebook_root": str(root), "match_count": len(matches), "cases": matches},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage verified alignment fix cases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--case", type=Path, required=True)
    validate_parser.set_defaults(handler=command_validate)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--case", type=Path, required=True)
    add_parser.add_argument("--casebook-root", type=Path, required=True)
    add_parser.add_argument("--confirm-maintainer-update", action="store_true", required=True)
    add_parser.set_defaults(handler=command_add)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--casebook-root", type=Path, default=default_casebook_root())
    search_parser.add_argument("--root-cause")
    search_parser.add_argument("--pattern")
    search_parser.add_argument("--architecture")
    search_parser.add_argument("--model-family")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.set_defaults(handler=command_search)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "limit", 1) < 1:
        print("ERROR: --limit must be positive", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args))
    except CaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
