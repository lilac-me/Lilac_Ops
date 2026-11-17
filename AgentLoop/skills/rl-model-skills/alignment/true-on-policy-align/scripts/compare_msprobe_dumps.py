#!/usr/bin/env python3
"""Inspect and exactly compare explicitly paired msprobe dump tensors."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


STAT_KEYS = {
    "max": "Max",
    "min": "Min",
    "mean": "Mean",
    "norm": "Norm",
}
RECORD_MARKERS = {"data_name", "dtype", "shape", *STAT_KEYS}


class CompareError(RuntimeError):
    def __init__(self, message: str, status: str = "INVALID_INPUT") -> None:
        super().__init__(message)
        self.status = status


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"Cannot read JSON {path}: {exc}", "INVALID_JSON") from exc


def write_json(data: Any, output: Path | None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def is_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    lowered = {str(key).lower() for key in value}
    return bool(lowered & RECORD_MARKERS)


def contains_record(value: Any, depth: int = 0) -> bool:
    if depth > 3:
        return False
    if is_record(value):
        return True
    if isinstance(value, dict):
        return any(contains_record(child, depth + 1) for child in value.values())
    if isinstance(value, list):
        return any(contains_record(child, depth + 1) for child in value)
    return False


def extract_op_map(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visit(node: Any, depth: int) -> None:
        if depth > 4 or not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(key, str) and isinstance(value, dict) and contains_record(value):
                if ".forward" in key or ".backward" in key or key.startswith(
                    ("Module.", "Functional.", "Tensor.", "Torch.")
                ):
                    result.setdefault(key, value)
                    continue
            if key in {"data", "dump_data", "api_data", "module_data"}:
                visit(value, depth + 1)

    visit(data, 0)
    if not result and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, dict) and contains_record(value):
                result[key] = value
    return result


def iter_records(node: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if is_record(node):
        yield prefix or "$", node
        return
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_records(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_records(value, child)


def select(node: Any, selector: str) -> Any:
    if selector in {"", "$"}:
        return node
    current = node
    for token in selector.split("."):
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise CompareError(f"Invalid list selector component {token!r}") from exc
        elif isinstance(current, dict):
            if token not in current:
                raise CompareError(f"Selector component {token!r} not found")
            current = current[token]
        else:
            raise CompareError(f"Cannot descend through selector component {token!r}")
    return current


def scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def extract_stats(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    stats: dict[str, Any] = {}
    for key, value in record.items():
        canonical = STAT_KEYS.get(str(key).lower())
        if canonical and scalar(value):
            stats[canonical] = value
        elif str(key).lower() in {"statistics", "summary"} and isinstance(value, dict):
            stats.update(extract_stats(value))
    return stats


def first_value(record: Any, wanted_key: str) -> Any:
    if isinstance(record, dict):
        for key, value in record.items():
            if str(key).lower() == wanted_key.lower():
                return value
        for value in record.values():
            found = first_value(value, wanted_key)
            if found is not None:
                return found
    elif isinstance(record, list):
        for value in record:
            found = first_value(value, wanted_key)
            if found is not None:
                return found
    return None


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": first_value(record, "dtype"),
        "shape": first_value(record, "shape"),
        "data_name": first_value(record, "data_name"),
        "stats": extract_stats(record),
    }


def resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def find_record(
    dump_data: Any, op_key: str, selector: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    op_map = extract_op_map(dump_data)
    if op_key not in op_map:
        matches = [key for key in op_map if key.endswith(op_key) or op_key in key]
        hint = f"; possible matches: {matches[:5]}" if matches else ""
        raise CompareError(
            f"Dump key {op_key!r} not found{hint}", "MISSING_RECORD"
        )
    record = select(op_map[op_key], selector)
    if not isinstance(record, dict):
        raise CompareError(
            f"{op_key}:{selector} does not select a tensor record",
            "INVALID_SELECTOR",
        )
    return record, op_map[op_key]


def tensor_path(
    side: dict[str, Any], dump_json: Path, record: dict[str, Any]
) -> Path:
    explicit = side.get("tensor_path")
    if explicit:
        return resolve_path(dump_json.parent, str(explicit))
    data_name = first_value(record, "data_name")
    if not isinstance(data_name, str) or not data_name:
        raise CompareError(
            f"No data_name for {side.get('key')}:{side.get('selector', '$')}",
            "MISSING_TENSOR_REFERENCE",
        )
    path = Path(data_name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "dump_tensor_data":
        return (dump_json.parent / path).resolve()
    return (dump_json.parent / "dump_tensor_data" / path).resolve()


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise CompareError(
            "PyTorch is required for tensor comparison; run in the dump environment",
            "ENVIRONMENT_ERROR",
        ) from exc
    return torch


def load_pt(path: Path, loaded_selector: str | None) -> Any:
    if not path.is_file():
        raise CompareError(f"Tensor file not found: {path}", "MISSING_TENSOR")
    torch = import_torch()
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if loaded_selector:
        value = select(value, loaded_selector)
    if not torch.is_tensor(value):
        raise CompareError(f"Loaded object is not a Tensor: {path}", "INVALID_TENSOR")
    return value


def get_transforms(spec: dict[str, Any], label: str) -> list[str]:
    transforms = spec.get("transforms", [])
    if not isinstance(transforms, list) or not all(
        isinstance(item, str) and item for item in transforms
    ):
        raise CompareError(
            f"{label}: transforms must be a list of non-empty strings",
            "INVALID_MANIFEST",
        )
    return transforms


def require_semantic_basis(
    spec: dict[str, Any], label: str, transforms: list[str]
) -> str | None:
    has_parts = bool(spec.get("parts"))
    has_combine = bool(spec.get("combine"))
    if not transforms and not has_parts and not has_combine:
        return None
    basis = spec.get("semantic_basis")
    if not isinstance(basis, str) or not basis.strip():
        raise CompareError(
            f"{label}: transforms/TP reconstruction require a non-empty "
            "semantic_basis derived from source, stack, topology, or model semantics",
            "MISSING_SEMANTIC_BASIS",
        )
    return basis.strip()


def parse_ints(text: str) -> list[int]:
    if not text:
        return []
    try:
        return [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise CompareError(f"Expected comma-separated integers, got {text!r}") from exc


def apply_transforms(tensor: Any, transforms: list[str], label: str) -> Any:
    result = tensor
    for raw in transforms:
        name, separator, argument = raw.partition(":")
        args = parse_ints(argument) if separator else []
        if name == "squeeze":
            result = result.squeeze() if not args else result.squeeze(args[0])
        elif name == "reshape":
            if not args:
                raise CompareError(f"{label}: reshape requires dimensions")
            result = result.reshape(*args)
        elif name == "permute":
            if not args:
                raise CompareError(f"{label}: permute requires dimensions")
            result = result.permute(*args)
        elif name == "transpose":
            if len(args) != 2:
                raise CompareError(f"{label}: transpose requires two dimensions")
            result = result.transpose(args[0], args[1])
        elif name == "flatten":
            if len(args) > 2:
                raise CompareError(f"{label}: flatten accepts zero to two dimensions")
            result = result.flatten(*args)
        elif name == "slice":
            if len(args) not in {3, 4}:
                raise CompareError(
                    f"{label}: slice requires dim,start,stop[,step]"
                )
            dim, start, stop = args[:3]
            step = args[3] if len(args) == 4 else 1
            indices = [slice(None)] * result.ndim
            indices[dim] = slice(start, stop, step)
            result = result[tuple(indices)]
        elif name == "select":
            if len(args) != 2:
                raise CompareError(f"{label}: select requires dim,index")
            result = result.select(args[0], args[1])
        elif name == "contiguous":
            if args:
                raise CompareError(f"{label}: contiguous takes no arguments")
            result = result.contiguous()
        else:
            raise CompareError(f"{label}: unsupported transform {name!r}")
    return result


def load_single_side(
    spec: dict[str, Any], dump_json: Path, dump_data: Any, label: str
) -> tuple[Any, dict[str, Any]]:
    transforms = get_transforms(spec, label)
    semantic_basis = require_semantic_basis(spec, label, transforms)
    key = spec.get("key")
    if not isinstance(key, str) or not key:
        raise CompareError(f"{label}: missing dump key")
    selector = str(spec.get("selector", "$"))
    record, _ = find_record(dump_data, key, selector)
    path = tensor_path(spec, dump_json, record)
    tensor = load_pt(path, spec.get("loaded_selector"))
    tensor = apply_transforms(tensor, transforms, label)
    return tensor, {
        "key": key,
        "selector": selector,
        "tensor_path": str(path),
        "record": record_summary(record),
        "transforms": transforms,
        "semantic_basis": semantic_basis,
    }


def load_side(
    spec: dict[str, Any], dump_json: Path, dump_data: Any, label: str
) -> tuple[Any, dict[str, Any]]:
    parts = spec.get("parts")
    if not parts:
        return load_single_side(spec, dump_json, dump_data, label)
    transforms = get_transforms(spec, label)
    semantic_basis = require_semantic_basis(spec, label, transforms)
    if not isinstance(parts, list) or not parts:
        raise CompareError(f"{label}: parts must be a non-empty list")
    tensors = []
    sources = []
    for index, part in enumerate(parts):
        tensor, source = load_single_side(
            part, dump_json, dump_data, f"{label}.parts[{index}]"
        )
        tensors.append(tensor)
        sources.append(source)
    combine = spec.get("combine", {})
    if not isinstance(combine, dict):
        raise CompareError(f"{label}: combine must be an object", "INVALID_MANIFEST")
    op = combine.get("op")
    dim = int(combine.get("dim", 0))
    torch = import_torch()
    if op == "cat":
        tensor = torch.cat(tensors, dim=dim)
    elif op == "stack":
        tensor = torch.stack(tensors, dim=dim)
    else:
        raise CompareError(f"{label}: combine.op must be 'cat' or 'stack'")
    tensor = apply_transforms(tensor, transforms, label)
    return tensor, {
        "parts": sources,
        "combine": {"op": op, "dim": dim},
        "transforms": transforms,
        "semantic_basis": semantic_basis,
    }


def json_number(value: float | int) -> float | int | str:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def compare_stats(infer_meta: dict[str, Any], train_meta: dict[str, Any]) -> dict[str, Any]:
    infer_record = infer_meta.get("record")
    train_record = train_meta.get("record")
    if not infer_record or not train_record:
        return {
            "comparable": False,
            "reason": "multi-part tensor or missing single-record metadata",
        }
    infer_shape = infer_record.get("shape")
    train_shape = train_record.get("shape")
    if infer_shape != train_shape:
        return {
            "comparable": False,
            "reason": "raw shapes differ; load tensors and apply explicit semantic transforms",
            "infer_shape": infer_shape,
            "train_shape": train_shape,
        }
    infer_stats = infer_record.get("stats", {})
    train_stats = train_record.get("stats", {})
    common = sorted(set(infer_stats) & set(train_stats))
    if not common:
        return {"comparable": False, "reason": "no common statistics"}
    differences = {
        key: {"infer": infer_stats[key], "train": train_stats[key]}
        for key in common
        if infer_stats[key] != train_stats[key]
    }
    return {
        "comparable": True,
        "common_fields": common,
        "exact": not differences,
        "differences": differences,
    }


def compare_tensors(infer: Any, train: Any, rtol: float, atol: float) -> dict[str, Any]:
    torch = import_torch()
    same_shape = tuple(infer.shape) == tuple(train.shape)
    same_dtype = infer.dtype == train.dtype
    result: dict[str, Any] = {
        "infer_shape": list(infer.shape),
        "train_shape": list(train.shape),
        "infer_dtype": str(infer.dtype),
        "train_dtype": str(train.dtype),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "torch_equal": False,
        "allclose": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "nonzero_count": None,
        "infer_nan_count": None,
        "train_nan_count": None,
        "infer_inf_count": None,
        "train_inf_count": None,
        "nonfinite_present": False,
        "blocker": None,
        "aligned_exact": False,
    }
    if not same_shape:
        result["status"] = "SHAPE_MISMATCH"
        result["blocker"] = (
            "normalized shapes still differ; correct the explicit semantic mapping"
        )
        return result
    equal = bool(torch.equal(infer, train))
    result["torch_equal"] = equal
    infer_supports_nonfinite = infer.is_floating_point() or infer.is_complex()
    train_supports_nonfinite = train.is_floating_point() or train.is_complex()
    infer_nan = (
        int(torch.count_nonzero(torch.isnan(infer)).item())
        if infer_supports_nonfinite
        else 0
    )
    train_nan = (
        int(torch.count_nonzero(torch.isnan(train)).item())
        if train_supports_nonfinite
        else 0
    )
    infer_inf = (
        int(torch.count_nonzero(torch.isinf(infer)).item())
        if infer_supports_nonfinite
        else 0
    )
    train_inf = (
        int(torch.count_nonzero(torch.isinf(train)).item())
        if train_supports_nonfinite
        else 0
    )
    nonfinite_present = any((infer_nan, train_nan, infer_inf, train_inf))
    result.update(
        {
            "infer_nan_count": infer_nan,
            "train_nan_count": train_nan,
            "infer_inf_count": infer_inf,
            "train_inf_count": train_inf,
            "nonfinite_present": nonfinite_present,
        }
    )
    if nonfinite_present:
        result.update(
            {
                "status": "NONFINITE_TENSOR",
                "nonzero_count": int(torch.count_nonzero(infer != train).item()),
                "blocker": (
                    "NaN/Inf is invalid alignment evidence; localize its first origin "
                    "before ordinary difference analysis"
                ),
            }
        )
        return result
    if infer.numel() == 0:
        max_abs = mean_abs = 0.0
        nonzero = 0
        allclose = True
    else:
        if infer.is_complex() or train.is_complex():
            left = infer.to(torch.complex128)
            right = train.to(torch.complex128)
        elif infer.is_floating_point() or train.is_floating_point():
            left = infer.to(torch.float64)
            right = train.to(torch.float64)
        else:
            left = infer.to(torch.int64)
            right = train.to(torch.int64)
        diff = (left - right).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.to(torch.float64).mean().item())
        nonzero = int(torch.count_nonzero(diff).item())
        if infer.is_floating_point() or train.is_floating_point():
            allclose = bool(
                torch.allclose(
                    infer.to(torch.float64),
                    train.to(torch.float64),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )
            )
        else:
            allclose = equal
    result.update(
        {
            "allclose": allclose,
            "max_abs_diff": json_number(max_abs),
            "mean_abs_diff": json_number(mean_abs),
            "nonzero_count": nonzero,
            "aligned_exact": bool(
                same_shape and same_dtype and equal and max_abs == 0.0
            ),
        }
    )
    if not same_dtype:
        result["status"] = "DTYPE_MISMATCH"
        result["blocker"] = "normalized dtypes differ under the strict alignment rule"
    elif result["aligned_exact"]:
        result["status"] = "EXACT_ALIGNED"
    else:
        result["status"] = "VALUE_DIFFERENCE"
    return result


def compare_pair(
    pair: dict[str, Any],
    infer_json: Path,
    train_json: Path,
    infer_data: Any,
    train_data: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    name = pair.get("name")
    if not isinstance(name, str) or not name:
        raise CompareError("Each pair requires a non-empty name")
    infer, infer_meta = load_side(
        pair.get("infer", {}), infer_json, infer_data, f"{name}.infer"
    )
    train, train_meta = load_side(
        pair.get("train", {}), train_json, train_data, f"{name}.train"
    )
    tensor_result = compare_tensors(infer, train, rtol, atol)
    return {
        "name": name,
        "status": tensor_result["status"],
        "semantic_node": pair.get("semantic_node"),
        "tensor_role": pair.get("tensor_role"),
        "infer": infer_meta,
        "train": train_meta,
        "statistics": compare_stats(infer_meta, train_meta),
        "tensor": tensor_result,
    }


def command_inspect(args: argparse.Namespace) -> int:
    dump_json = args.dump_json.resolve()
    data = load_json(dump_json)
    op_map = extract_op_map(data)
    regex = re.compile(args.pattern) if args.pattern else None
    records = []
    for key, entry in op_map.items():
        if regex and not regex.search(key):
            continue
        for selector, record in iter_records(entry):
            records.append(
                {
                    "key": key,
                    "selector": selector,
                    **record_summary(record),
                }
            )
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break
    result = {
        "dump_json": str(dump_json),
        "operator_count": len(op_map),
        "record_count": len(records),
        "records": records,
    }
    write_json(result, args.output)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise CompareError("Manifest root must be an object")
    infer_json = resolve_path(manifest_path.parent, manifest["infer_dump_json"])
    train_json = resolve_path(manifest_path.parent, manifest["train_dump_json"])
    infer_data = load_json(infer_json)
    train_data = load_json(train_json)
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise CompareError("Manifest requires a non-empty pairs list")
    if not all(isinstance(pair, dict) for pair in pairs):
        raise CompareError("Every manifest pair must be an object")
    pair_names = [pair.get("name") for pair in pairs]
    if any(not isinstance(name, str) or not name for name in pair_names):
        raise CompareError("Every manifest pair requires a non-empty name")
    if len(set(pair_names)) != len(pair_names):
        raise CompareError("Manifest pair names must be unique")
    execution_order_basis = manifest.get("execution_order_basis")
    if len(pairs) > 1 and (
        not isinstance(execution_order_basis, str)
        or not execution_order_basis.strip()
    ):
        raise CompareError(
            "A multi-pair manifest requires execution_order_basis; file order alone "
            "is not evidence of semantic execution order",
            "MISSING_ORDER_BASIS",
        )
    rtol = float(manifest.get("rtol", 1e-5))
    atol = float(manifest.get("atol", 1e-8))
    stop_at_first = bool(manifest.get("stop_at_first_diff", True))
    results = []
    first_failure = None
    first_nonzero = None
    for pair in pairs:
        try:
            result = compare_pair(
                pair, infer_json, train_json, infer_data, train_data, rtol, atol
            )
        except CompareError as exc:
            result = {
                "name": pair.get("name", "<unnamed>"),
                "status": exc.status,
                "semantic_node": pair.get("semantic_node"),
                "tensor_role": pair.get("tensor_role"),
                "error": str(exc),
            }
        results.append(result)
        tensor = result.get("tensor", {})
        if first_nonzero is None and tensor.get("nonzero_count") not in {None, 0}:
            first_nonzero = result["name"]
        if result["status"] != "EXACT_ALIGNED":
            first_failure = first_failure or result["name"]
            if stop_at_first:
                break
    statuses = [item["status"] for item in results]
    status_counts = {
        status: statuses.count(status) for status in sorted(set(statuses))
    }
    all_aligned = len(results) == len(pairs) and all(
        status == "EXACT_ALIGNED" for status in statuses
    )
    known_exactness_statuses = {
        "EXACT_ALIGNED",
        "VALUE_DIFFERENCE",
        "DTYPE_MISMATCH",
        "NONFINITE_TENSOR",
    }
    if all_aligned:
        comparison_status = "ALL_PAIRS_EXACT"
    elif any(status not in known_exactness_statuses for status in statuses) or (
        "SHAPE_MISMATCH" in statuses
    ):
        comparison_status = "NOT_COMPARABLE"
    elif "NONFINITE_TENSOR" in statuses:
        comparison_status = "NONFINITE_TENSOR"
    elif first_nonzero is not None:
        comparison_status = "FIRST_NONZERO_PAIR_FOUND"
    else:
        comparison_status = "EXACTNESS_FAILURE"
    report = {
        "manifest": str(manifest_path),
        "infer_dump_json": str(infer_json),
        "train_dump_json": str(train_json),
        "execution_order_basis": execution_order_basis,
        "coverage": manifest.get("coverage"),
        "pair_count": len(pairs),
        "compared_count": len(results),
        "comparison_status": comparison_status,
        "status_counts": status_counts,
        "all_aligned_exact": all_aligned,
        "first_failure": first_failure,
        "first_nonzero_difference": first_nonzero,
        "first_nonzero_is_candidate_only": True,
        "semantic_first_divergence_requires_analysis_proof": True,
        "results": results,
    }
    write_json(report, args.output)
    return 0 if all_aligned else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect msprobe dump.json records and compare explicitly mapped .pt tensors"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--dump-json", type=Path, required=True)
    inspect_parser.add_argument("--pattern", default="")
    inspect_parser.add_argument("--limit", type=int, default=0)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.set_defaults(handler=command_inspect)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(handler=command_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (CompareError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
