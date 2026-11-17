#!/usr/bin/env python3
"""Build, query, and validate local model and dataset inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "local-asset-inventory-v1"
MODEL_WEIGHT_FILENAMES = {
    "model.safetensors",
    "pytorch_model.bin",
}
MODEL_WEIGHT_INDEX_FILENAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
DATASET_FILE_SUFFIXES = {".arrow", ".csv", ".json", ".jsonl", ".parquet"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()).expanduser().resolve() for item in value.split(",") if item.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_path(repo: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo / path).resolve()
    agent = (repo / ".npu-agent").resolve()
    if not resolved.is_relative_to(agent):
        raise ValueError(f"output must be under {agent}: {resolved}")
    return resolved


def _contained_file(root: Path, value: str) -> Path | None:
    if not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def model_evidence(path: Path) -> list[str]:
    if not path.is_dir() or not (path / "config.json").is_file():
        return []
    if _json_object(path / "config.json") is None:
        return []
    evidence = ["config.json"]
    try:
        names = {child.name for child in path.iterdir() if child.is_file()}
    except OSError:
        return []

    for filename in sorted(names.intersection(MODEL_WEIGHT_INDEX_FILENAMES)):
        index = _json_object(path / filename)
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards = sorted({value for value in weight_map.values() if isinstance(value, str) and value})
        if not shards or any(_contained_file(path, shard) is None for shard in shards):
            continue
        return [*evidence, filename, *shards]

    for filename in sorted(names.intersection(MODEL_WEIGHT_FILENAMES)):
        if _contained_file(path, filename) is not None:
            return [*evidence, filename]
    return []


def _state_data_files(path: Path) -> list[str] | None:
    state = _json_object(path / "state.json")
    if state is None:
        return None
    values = state.get("_data_files")
    if not isinstance(values, list) or not values:
        return None
    files: list[str] = []
    for item in values:
        filename = item.get("filename") if isinstance(item, dict) else None
        if not isinstance(filename, str) or _contained_file(path, filename) is None:
            return None
        files.append(filename)
    return files


def _dataset_dict_evidence(path: Path) -> list[str] | None:
    dataset_dict = _json_object(path / "dataset_dict.json")
    if dataset_dict is None:
        return None
    split_values = dataset_dict.get("splits")
    if not isinstance(split_values, list) or not split_values:
        return None
    evidence = ["dataset_dict.json"]
    for split in split_values:
        if not isinstance(split, str) or not split:
            return None
        split_path = (path / split).resolve()
        try:
            split_path.relative_to(path.resolve())
        except ValueError:
            return None
        data_files = _state_data_files(split_path)
        if not data_files:
            return None
        evidence.extend(f"{split}/{filename}" for filename in data_files)
    return evidence


def dataset_evidence(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    state_files = _state_data_files(path)
    if state_files:
        return ["state.json", *state_files]
    dataset_dict = _dataset_dict_evidence(path)
    if dataset_dict:
        return dataset_dict
    evidence = ["dataset_infos.json"] if _json_object(path / "dataset_infos.json") is not None else []
    for directory in (path, path / "data"):
        if not directory.is_dir():
            continue
        try:
            data_files = sorted(
                child.name
                for child in directory.iterdir()
                if child.is_file()
                and child.stat().st_size > 0
                and child.suffix.lower() in DATASET_FILE_SUFFIXES
                and child.name not in {"dataset_infos.json", "dataset_dict.json", "state.json"}
            )
        except OSError:
            continue
        if data_files:
            prefix = "" if directory == path else "data/"
            marker = prefix + data_files[0]
            if marker not in evidence:
                evidence.append(marker)
            break
    return evidence


def dataset_looks_like_collection(path: Path, evidence: list[str]) -> bool:
    if any(marker in evidence for marker in ("state.json", "dataset_dict.json", "dataset_infos.json")):
        return False
    try:
        child_directories = [child.name for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")]
    except OSError:
        return False
    return any(name != "data" for name in child_directories)


def aliases_for(root: Path, leaf: Path) -> list[str]:
    relative = leaf.relative_to(root).as_posix()
    aliases = [relative, leaf.name]
    if len(Path(relative).parts) >= 2:
        aliases.append("/".join(Path(relative).parts[-2:]))
    return list(dict.fromkeys(alias for alias in aliases if alias and alias != "."))


def asset_record(kind: str, root: Path, leaf: Path, evidence: list[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(leaf),
        "root": str(root),
        "relative_path": leaf.relative_to(root).as_posix(),
        "name": leaf.name,
        "aliases": aliases_for(root, leaf),
        "evidence": evidence,
    }


def discover_assets(
    kind: str,
    roots: list[Path],
    *,
    max_depth: int,
    max_directories: int,
    max_assets: int,
) -> tuple[list[dict[str, Any]], list[str], list[str], bool]:
    evidence_fn = model_evidence if kind == "model" else dataset_evidence
    assets: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    truncated = False
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            errors.append(f"{kind} root does not exist or is not a directory: {root}")
            continue
        root_evidence = evidence_fn(root)
        if kind == "dataset" and root_evidence and dataset_looks_like_collection(root, root_evidence):
            root_evidence = []
        if root_evidence:
            if len(assets) >= max_assets:
                truncated = True
                errors.append(f"{kind} scan exceeded max_assets={max_assets}")
                return sorted(assets, key=lambda item: item["path"].lower()), errors, warnings, truncated
            assets.append(asset_record(kind, root, root, root_evidence))
            seen.add(root)
            continue
        visited = 0
        directory_limit_reported = False
        depth_limit_reported = False
        for current, directories, _files in os.walk(root, followlinks=False):
            visited += 1
            current_path = Path(current).resolve()
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                directories[:] = []
                continue
            directories[:] = sorted(name for name in directories if not name.startswith("."))
            if depth >= max_depth and directories:
                truncated = True
                if not depth_limit_reported:
                    errors.append(f"{kind} scan exceeded max_depth={max_depth} under root: {root}")
                    depth_limit_reported = True
                directories[:] = []
            if visited >= max_directories and directories:
                truncated = True
                if not directory_limit_reported:
                    errors.append(f"{kind} scan exceeded max_directories={max_directories} under root: {root}")
                    directory_limit_reported = True
                directories[:] = []
            evidence = evidence_fn(current_path)
            if kind == "dataset" and evidence and dataset_looks_like_collection(current_path, evidence):
                evidence = []
            if not evidence or current_path in seen:
                continue
            if len(assets) >= max_assets:
                truncated = True
                errors.append(f"{kind} scan exceeded max_assets={max_assets}")
                return sorted(assets, key=lambda item: item["path"].lower()), errors, warnings, truncated
            assets.append(asset_record(kind, root, current_path, evidence))
            seen.add(current_path)
            directories[:] = []
    return sorted(assets, key=lambda item: item["path"].lower()), errors, warnings, truncated


def inventory_digest(value: dict[str, Any]) -> str:
    payload = {
        "schema_version": value.get("schema_version"),
        "model_roots": value.get("model_roots"),
        "dataset_roots": value.get("dataset_roots"),
        "models": value.get("models"),
        "datasets": value.get("datasets"),
        "truncated": value.get("truncated"),
        "scan_limits": value.get("scan_limits"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_inventory(model_roots: list[Path], dataset_roots: list[Path]) -> dict[str, Any]:
    models, model_errors, model_warnings, model_truncated = discover_assets(
        "model",
        model_roots,
        max_depth=4,
        max_directories=5000,
        max_assets=512,
    )
    datasets, dataset_errors, dataset_warnings, dataset_truncated = discover_assets(
        "dataset",
        dataset_roots,
        max_depth=3,
        max_directories=8000,
        max_assets=1024,
    )
    errors = [*model_errors, *dataset_errors]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "BLOCKED",
        "created_at": utc_now(),
        "model_roots": [str(path) for path in model_roots],
        "dataset_roots": [str(path) for path in dataset_roots],
        "models": models,
        "datasets": datasets,
        "truncated": model_truncated or dataset_truncated,
        "scan_limits": {
            "models": {"max_depth": 4, "max_directories": 5000, "max_assets": 512},
            "datasets": {"max_depth": 3, "max_directories": 8000, "max_assets": 1024},
        },
        "errors": errors,
        "warnings": [*model_warnings, *dataset_warnings],
    }
    result["inventory_digest"] = inventory_digest(result)
    return result


def normalized_alias(value: str) -> str:
    return value.strip().strip("'\"").rstrip("/").lower().replace("_", "-")


def query_inventory(inventory: dict[str, Any], kind: str, reference: str) -> dict[str, Any]:
    collection = inventory.get("models" if kind == "model" else "datasets")
    collection = collection if isinstance(collection, list) else []
    wanted = normalized_alias(reference)
    exact = [
        item
        for item in collection
        if normalized_alias(item.get("relative_path", "")) == wanted
        or normalized_alias(item.get("path", "")) == wanted
    ]
    if len(exact) == 1:
        matches = exact
    else:
        matches = [
            item
            for item in collection
            if wanted in {normalized_alias(alias) for alias in item.get("aliases", [])}
        ]
    status = "FOUND" if len(matches) == 1 else ("MISSING" if not matches else "AMBIGUOUS")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "kind": kind,
        "reference": reference,
        "selected": matches[0] if len(matches) == 1 else None,
        "matches": matches,
        "inventory_digest": inventory.get("inventory_digest", ""),
    }


def validate_inventory(
    inventory: dict[str, Any],
    model_roots: list[Path] | None = None,
    dataset_roots: list[Path] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if inventory.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if inventory.get("status") not in {"PASS", "BLOCKED"}:
        errors.append("status must be PASS or BLOCKED")
    if inventory.get("truncated") is True:
        errors.append("inventory scan is truncated")
    expected_models = [str(path) for path in model_roots] if model_roots is not None else None
    expected_datasets = [str(path) for path in dataset_roots] if dataset_roots is not None else None
    if expected_models is not None and inventory.get("model_roots") != expected_models:
        errors.append("model_roots do not match current config")
    if expected_datasets is not None and inventory.get("dataset_roots") != expected_datasets:
        errors.append("dataset_roots do not match current config")
    for kind, key, evidence_fn in (
        ("model", "models", model_evidence),
        ("dataset", "datasets", dataset_evidence),
    ):
        root_values = inventory.get(f"{kind}_roots")
        if not isinstance(root_values, list) or not all(isinstance(value, str) for value in root_values):
            errors.append(f"{kind}_roots must be a list of paths")
            root_values = []
        roots = [Path(value).resolve() for value in root_values]
        collection = inventory.get(key)
        if not isinstance(collection, list):
            errors.append(f"{key} must be a list")
            collection = []
        for item in collection:
            if not isinstance(item, dict):
                errors.append(f"{kind} asset record must be an object")
                continue
            path = Path(str(item.get("path") or "")).resolve()
            if not any(path == root or path.is_relative_to(root) for root in roots):
                errors.append(f"{kind} asset outside configured roots: {path}")
            current_evidence = evidence_fn(path)
            if not current_evidence:
                errors.append(f"incomplete {kind} asset: {path}")
            elif item.get("evidence") != current_evidence:
                errors.append(f"{kind} asset evidence is stale: {path}")
    try:
        expected_digest = inventory_digest(inventory)
        if inventory.get("inventory_digest") != expected_digest:
            errors.append("inventory_digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"inventory digest cannot be computed: {exc}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "BLOCKED",
        "errors": errors,
        "inventory_digest": inventory.get("inventory_digest", ""),
        "model_count": len(inventory.get("models", [])) if isinstance(inventory.get("models"), list) else 0,
        "dataset_count": len(inventory.get("datasets", [])) if isinstance(inventory.get("datasets"), list) else 0,
    }


def write_report(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Local Asset Inventory",
        "",
        f"- status: `{inventory['status']}`",
        f"- inventory_digest: `{inventory['inventory_digest']}`",
        f"- model_roots: `{len(inventory['model_roots'])}`",
        f"- dataset_roots: `{len(inventory['dataset_roots'])}`",
        f"- models: `{len(inventory['models'])}`",
        f"- datasets: `{len(inventory['datasets'])}`",
        "",
        "## Models",
        "",
    ]
    lines.extend(
        f"- `{item['relative_path']}` -> `{item['path']}`"
        for item in inventory["models"]
    )
    lines.extend(["", "## Datasets", ""])
    lines.extend(
        f"- `{item['relative_path']}` -> `{item['path']}`"
        for item in inventory["datasets"]
    )
    if inventory["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in inventory["errors"])
    atomic_write_text(path, "\n".join(lines) + "\n")


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="npu-local-assets-") as tmp:
        repo = Path(tmp)
        models = repo / "models"
        model = models / "Qwen" / "Qwen3-0.6B"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"model")
        datasets = repo / "datasets"
        dataset = datasets / "allenai" / "tulu-3-sft-personas-algebra"
        (dataset / "data").mkdir(parents=True)
        (dataset / "data" / "train.parquet").write_bytes(b"parquet")
        inventory = build_inventory([models], [datasets])
        validation = validate_inventory(inventory, [models.resolve()], [datasets.resolve()])
        model_query = query_inventory(inventory, "model", "Qwen/Qwen3-0.6B")
        dataset_query = query_inventory(inventory, "dataset", "allenai/tulu-3-sft-personas-algebra")
        missing = query_inventory(inventory, "dataset", "allenai/missing")
        malformed = validate_inventory({"schema_version": SCHEMA_VERSION, "models": "bad", "datasets": None})
        broken_model = models / "broken" / "missing-shard"
        broken_model.mkdir(parents=True)
        (broken_model / "config.json").write_text("{}\n", encoding="utf-8")
        (broken_model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
        second_model = models / "Qwen" / "Qwen2.5-0.5B"
        second_model.mkdir(parents=True)
        (second_model / "config.json").write_text("{}\n", encoding="utf-8")
        (second_model / "model.safetensors").write_bytes(b"model")
        broken_dataset = datasets / "broken" / "missing-arrow"
        broken_dataset.mkdir(parents=True)
        (broken_dataset / "state.json").write_text(
            json.dumps({"_data_files": [{"filename": "data-00000-of-00001.arrow"}]}),
            encoding="utf-8",
        )
        broken_inventory = build_inventory([models], [datasets])
        limited_assets, limited_errors, _limited_warnings, limited_truncated = discover_assets(
            "model",
            [models],
            max_depth=4,
            max_directories=5000,
            max_assets=1,
        )
        if (
            inventory["status"] != "PASS"
            or validation["status"] != "PASS"
            or model_query["status"] != "FOUND"
            or dataset_query["status"] != "FOUND"
            or missing["status"] != "MISSING"
            or malformed["status"] != "BLOCKED"
            or model_query["selected"]["path"] != str(model.resolve())
            or dataset_query["selected"]["path"] != str(dataset.resolve())
            or any(item["path"] == str(broken_model.resolve()) for item in broken_inventory["models"])
            or any(item["path"] == str(broken_dataset.resolve()) for item in broken_inventory["datasets"])
            or not limited_truncated
            or not limited_errors
            or len(limited_assets) != 1
        ):
            print(
                json.dumps(
                    {
                        "inventory": inventory,
                        "validation": validation,
                        "model_query": model_query,
                        "dataset_query": dataset_query,
                        "missing": missing,
                    },
                    indent=2,
                )
            )
            return 1
    print("SELF_TEST_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--repo", required=True)
    build_parser.add_argument("--model-roots", default="")
    build_parser.add_argument("--dataset-roots", default="")
    build_parser.add_argument("--output", default=".npu-agent/local_asset_inventory.json")
    build_parser.add_argument("--report", default=".npu-agent/reports/local_asset_inventory.md")
    build_parser.add_argument("--json", action="store_true")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--inventory", required=True)
    query_parser.add_argument("--kind", choices=("model", "dataset"), required=True)
    query_parser.add_argument("--reference", required=True)
    query_parser.add_argument("--required", action="store_true")
    query_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--repo", required=True)
    validate_parser.add_argument("--inventory", default=".npu-agent/local_asset_inventory.json")
    validate_parser.add_argument("--model-roots")
    validate_parser.add_argument("--dataset-roots")
    validate_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.command == "build":
        repo = Path(args.repo).resolve()
        inventory = build_inventory(split_paths(args.model_roots), split_paths(args.dataset_roots))
        destination = output_path(repo, args.output)
        report = output_path(repo, args.report)
        write_json(destination, inventory)
        write_report(report, inventory)
        summary = {
            "status": inventory["status"],
            "output": str(destination),
            "report": str(report),
            "inventory_digest": inventory["inventory_digest"],
            "model_count": len(inventory["models"]),
            "dataset_count": len(inventory["datasets"]),
            "errors": inventory["errors"],
        }
        print(json.dumps(inventory if args.json else summary, indent=2))
        return 0 if inventory["status"] == "PASS" else 2
    if args.command == "query":
        result = query_inventory(load_json(Path(args.inventory).resolve()), args.kind, args.reference)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "FOUND" or not args.required else 3
    if args.command == "validate":
        repo = Path(args.repo).resolve()
        inventory_path = output_path(repo, args.inventory)
        result = validate_inventory(
            load_json(inventory_path),
            split_paths(args.model_roots) if args.model_roots is not None else None,
            split_paths(args.dataset_roots) if args.dataset_roots is not None else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    parser.error("command is required unless --self-test is used")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
