#!/usr/bin/env python3
"""Run a case-specific train/infer micro-probe and emit exact summaries only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_msprobe_dumps import CompareError, compare_tensors  # noqa: E402


class ProbeError(RuntimeError):
    pass


SHA256_VALUE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
REPAIR_UNIT_ID = re.compile(r"^R[0-9]{3,}$")
DIFF_ID = re.compile(r"^D[0-9]{3,}$")


def load_adapter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("alignment_probe_adapter", path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # adapter errors need a concise structured failure
        raise ProbeError(f"Adapter import failed: {exc}") from exc
    return module


def require_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise ProbeError(f"Adapter must define callable {name}")
    return value


def require_metadata(module: ModuleType) -> dict[str, Any]:
    metadata_fn = require_callable(module, "probe_metadata")
    metadata = metadata_fn()
    if not isinstance(metadata, dict):
        raise ProbeError("probe_metadata() must return an object")
    for key in (
        "case_id",
        "repair_unit_id",
        "member_diff_ids",
        "phase",
        "model",
        "topology",
        "source_revisions",
        "upstream_replay_ids",
    ):
        if key not in metadata:
            raise ProbeError(f"probe_metadata() missing {key}")
    if metadata["phase"] != "prefill":
        raise ProbeError("probe_metadata().phase must be prefill")
    if not isinstance(metadata["repair_unit_id"], str) or not REPAIR_UNIT_ID.match(
        metadata["repair_unit_id"]
    ):
        raise ProbeError("probe_metadata().repair_unit_id is invalid")
    member_diff_ids = metadata["member_diff_ids"]
    if not isinstance(member_diff_ids, list) or not member_diff_ids or not all(
        isinstance(item, str) and DIFF_ID.match(item) for item in member_diff_ids
    ):
        raise ProbeError("probe_metadata().member_diff_ids is invalid")
    upstream_replay_ids = metadata["upstream_replay_ids"]
    if not isinstance(upstream_replay_ids, list) or not all(
        isinstance(item, str) and DIFF_ID.match(item) for item in upstream_replay_ids
    ):
        raise ProbeError("probe_metadata().upstream_replay_ids is invalid")
    topology = metadata["topology"]
    if not isinstance(topology, dict) or topology.get("tp") != 1:
        raise ProbeError("probe_metadata().topology.tp must be 1")
    for key in (
        "input_hashes",
        "state_dependencies",
        "semantic_transforms",
        "applicability_limits",
    ):
        value = metadata.get(key)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ProbeError(f"probe_metadata().{key} must be non-empty strings")
    if not all(SHA256_VALUE.match(value) for value in metadata["input_hashes"]):
        raise ProbeError("probe_metadata().input_hashes must contain full sha256 values")
    return metadata


def is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.is_tensor(value))


def compare_tree(actual: Any, expected: Any, path: str = "$") -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, current: str) -> None:
        left_tensor = is_tensor(left)
        right_tensor = is_tensor(right)
        if left_tensor or right_tensor:
            if not (left_tensor and right_tensor):
                leaves.append(
                    {
                        "path": current,
                        "kind": "type_mismatch",
                        "exact": False,
                        "actual_type": type(left).__name__,
                        "expected_type": type(right).__name__,
                    }
                )
                return
            try:
                result = compare_tensors(left, right, rtol=0.0, atol=0.0)
            except CompareError as exc:
                raise ProbeError(f"Tensor compare failed at {current}: {exc}") from exc
            leaves.append(
                {
                    "path": current,
                    "kind": "tensor",
                    "exact": bool(result["aligned_exact"]),
                    "result": result,
                }
            )
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                leaves.append(
                    {
                        "path": current,
                        "kind": "type_mismatch",
                        "exact": False,
                        "actual_type": type(left).__name__,
                        "expected_type": type(right).__name__,
                    }
                )
                return
            if set(left) != set(right):
                leaves.append(
                    {
                        "path": current,
                        "kind": "dict_keys",
                        "exact": False,
                        "actual_keys": sorted(map(str, left)),
                        "expected_keys": sorted(map(str, right)),
                    }
                )
                return
            for key in sorted(left, key=str):
                visit(left[key], right[key], f"{current}.{key}")
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                leaves.append(
                    {
                        "path": current,
                        "kind": "type_mismatch",
                        "exact": False,
                        "actual_type": type(left).__name__,
                        "expected_type": type(right).__name__,
                    }
                )
                return
            if len(left) != len(right):
                leaves.append(
                    {
                        "path": current,
                        "kind": "sequence_length",
                        "exact": False,
                        "actual_length": len(left),
                        "expected_length": len(right),
                    }
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{current}[{index}]")
            return
        exact = bool(left == right)
        if isinstance(left, float) and not math.isfinite(left):
            exact = False
        if isinstance(right, float) and not math.isfinite(right):
            exact = False
        leaves.append(
            {
                "path": current,
                "kind": "scalar",
                "exact": exact,
                "actual_type": type(left).__name__,
                "expected_type": type(right).__name__,
                "actual_finite": not isinstance(left, float) or math.isfinite(left),
                "expected_finite": not isinstance(right, float) or math.isfinite(right),
            }
        )

    visit(actual, expected, path)
    all_exact = bool(leaves) and all(item["exact"] for item in leaves)
    first_failure = next((item["path"] for item in leaves if not item["exact"]), None)
    return {"all_exact": all_exact, "first_failure": first_failure, "leaves": leaves}


def load_fresh_case(adapter: ModuleType, case_dir: Path) -> Any:
    return require_callable(adapter, "load_probe_case")(case_dir)


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    adapter_path = args.adapter.resolve()
    case_dir = args.case_dir.resolve()
    adapter = load_adapter(adapter_path)
    metadata = require_metadata(adapter)
    expected_fn = require_callable(adapter, "load_expected_outputs")
    expected = expected_fn(load_fresh_case(adapter, case_dir))
    if not isinstance(expected, dict) or set(expected) != {"train", "infer"}:
        raise ProbeError("load_expected_outputs() must return exactly train and infer")

    train_output = require_callable(adapter, "run_train_reference")(
        load_fresh_case(adapter, case_dir)
    )
    infer_output = require_callable(adapter, "run_infer_baseline")(
        load_fresh_case(adapter, case_dir)
    )
    comparisons = {
        "train_reference_vs_train_dump": compare_tree(
            train_output, expected["train"]
        ),
        "infer_baseline_vs_infer_dump": compare_tree(
            infer_output, expected["infer"]
        ),
        "infer_baseline_vs_train_reference": compare_tree(
            infer_output, train_output
        ),
    }
    reproduced = (
        comparisons["train_reference_vs_train_dump"]["all_exact"]
        and comparisons["infer_baseline_vs_infer_dump"]["all_exact"]
        and not comparisons["infer_baseline_vs_train_reference"]["all_exact"]
    )

    candidate_exact = None
    if args.command == "verify":
        candidate_output = require_callable(adapter, "run_infer_candidate")(
            load_fresh_case(adapter, case_dir)
        )
        comparisons["infer_candidate_vs_train_reference"] = compare_tree(
            candidate_output, train_output
        )
        candidate_exact = comparisons["infer_candidate_vs_train_reference"][
            "all_exact"
        ]

    if not reproduced:
        status = "PROBE_NOT_REPRODUCED"
        exit_code = 1
    elif args.command == "reproduce":
        status = "PROBE_REPRODUCED"
        exit_code = 0
    elif candidate_exact:
        status = "PROBE_CANDIDATE_EXACT"
        exit_code = 0
    else:
        status = "PROBE_CANDIDATE_DIFFERS"
        exit_code = 1

    return (
        {
            "probe_status": status,
            "adapter": str(adapter_path),
            "case_dir": str(case_dir),
            "metadata": metadata,
            "baseline_reproduced": reproduced,
            "candidate_exact": candidate_exact,
            "comparisons": comparisons,
            "stores_tensor_values": False,
        },
        exit_code,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce and verify one isolated train/infer divergence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("reproduce", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--adapter", type=Path, required=True)
        subparser.add_argument("--case-dir", type=Path, required=True)
        subparser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report, exit_code = run_probe(args)
        output = args.report.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output}: {report['probe_status']}")
        return exit_code
    except (ProbeError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
