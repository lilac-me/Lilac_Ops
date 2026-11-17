#!/usr/bin/env python3
"""Compare two shared-input logits payloads using valid-token cosine similarity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .alignment_utils import input_sha256, safe_torch_load, to_json_compatible
except ImportError:  # Direct script execution.
    from alignment_utils import input_sha256, safe_torch_load, to_json_compatible  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-file", "--hf-file", "--hf", dest="reference_file", required=True
    )
    parser.add_argument(
        "--candidate-file", "--mindspeed-file", "--mindspeed",
        dest="candidate_file", required=True,
    )
    parser.add_argument("--reference-framework", default="transformers")
    parser.add_argument("--candidate-framework", default="mindspeed-llm")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--skip-framework-check", action="store_true")
    parser.add_argument("--candidate-execution-kind")
    parser.add_argument("--minimum-candidate-world-size", type=int, default=1)
    parser.add_argument("--require-candidate-fsdp-wrapped", action="store_true")
    parser.add_argument("--require-candidate-complete-weight-load", action="store_true")
    parser.add_argument("--report-file", default="comparison_report.json")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--token-chunk-size", type=int, default=32)
    parser.add_argument(
        "--max-token-cosines-in-report",
        type=int,
        default=256,
        help="Include per-token positions/cosines only at or below this valid-token count; 0 disables.",
    )
    return parser.parse_args()


def load_output(
    path: str,
    *,
    expected_framework: str | None,
    label: str,
) -> dict[str, Any]:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a dictionary payload.")
    required = {"input_ids", "attention_mask", "logits"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"{path} is missing keys: {sorted(missing)}")
    if expected_framework is not None and payload.get("framework") != expected_framework:
        raise ValueError(
            f"{label} framework={payload.get('framework')!r}, expected {expected_framework!r}"
        )
    return payload


def _as_input_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    value = value.detach().cpu().to(torch.long).contiguous()
    if value.ndim != 2:
        raise ValueError(f"{name} must be [B,S], got {tuple(value.shape)}")
    return value


def validate_candidate_execution(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    execution = candidate.get("metadata", {}).get("execution")
    required_kind = getattr(args, "candidate_execution_kind", None)
    minimum_world_size = getattr(args, "minimum_candidate_world_size", 1)
    require_fsdp = getattr(args, "require_candidate_fsdp_wrapped", False)
    require_complete_weight_load = getattr(args, "require_candidate_complete_weight_load", False)
    if not any((required_kind, minimum_world_size > 1, require_fsdp, require_complete_weight_load)):
        return execution or {}
    if not isinstance(execution, dict):
        raise ValueError("Candidate payload is missing metadata.execution proof")
    if required_kind and execution.get("kind") != required_kind:
        raise ValueError(
            f"Candidate execution kind={execution.get('kind')!r}, expected {required_kind!r}"
        )
    world_size = execution.get("world_size")
    if not isinstance(world_size, int) or world_size < minimum_world_size:
        raise ValueError(
            f"Candidate world_size={world_size!r}, expected >= {minimum_world_size}"
        )
    if require_fsdp:
        wrapped_count = execution.get("fsdp_wrapped_module_count")
        if not isinstance(wrapped_count, int) or wrapped_count <= 0:
            raise ValueError(
                f"Candidate fsdp_wrapped_module_count={wrapped_count!r}, expected > 0"
            )
    if require_complete_weight_load:
        missing_count = execution.get("weight_load_missing_parameter_count")
        if missing_count != 0:
            raise ValueError(
                f"Candidate weight_load_missing_parameter_count={missing_count!r}, expected 0"
            )
    return execution


def validate_inputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    reference_ids = _as_input_tensor(reference["input_ids"], "reference input_ids")
    candidate_ids = _as_input_tensor(candidate["input_ids"], "candidate input_ids")
    reference_mask = _as_input_tensor(reference["attention_mask"], "reference attention_mask")
    candidate_mask = _as_input_tensor(candidate["attention_mask"], "candidate attention_mask")
    if reference_ids.shape != candidate_ids.shape or not torch.equal(reference_ids, candidate_ids):
        raise AssertionError(
            f"{reference_label}/{candidate_label} input_ids differ; "
            f"shapes={tuple(reference_ids.shape)}/{tuple(candidate_ids.shape)}"
        )
    if reference_mask.shape != reference_ids.shape:
        raise ValueError("reference attention_mask shape does not match input_ids.")
    if candidate_mask.shape != candidate_ids.shape or not torch.equal(reference_mask, candidate_mask):
        raise AssertionError(f"{reference_label}/{candidate_label} attention_mask values differ.")
    if not torch.all((reference_mask == 0) | (reference_mask == 1)):
        raise ValueError("attention_mask must contain only 0 and 1.")
    if torch.any(reference_mask.sum(dim=-1) == 0):
        raise ValueError("Every input row must contain at least one valid token.")
    seen_padding = reference_mask.eq(0).cummax(dim=-1).values
    if torch.any(seen_padding & reference_mask.eq(1)):
        raise ValueError("Only no padding or right padding is supported by this baseline.")

    digest = input_sha256(reference_ids, reference_mask)
    for label, payload in ((reference_label, reference), (candidate_label, candidate)):
        recorded = payload.get("input_sha256")
        if recorded != digest:
            raise AssertionError(f"{label} input_sha256={recorded!r}, recomputed={digest!r}")
    return reference_ids, reference_mask, digest


def compare_in_chunks(
    reference_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    valid_positions: torch.Tensor,
    *,
    chunk_size: int,
    topk: int,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    cosines = []
    element_count = 0
    sum_abs = sum_square = max_abs = 0.0
    flat_dot = flat_reference_square = flat_candidate_square = 0.0
    top1_matches = 0
    topk_overlap_sum = 0.0
    worst_token_max_abs = -1.0
    worst_token_position = None

    for start in range(0, reference_valid.shape[0], chunk_size):
        end = min(start + chunk_size, reference_valid.shape[0])
        reference = reference_valid[start:end].float()
        candidate = candidate_valid[start:end].float()
        for label, tensor in ((reference_label, reference), (candidate_label, candidate)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{label} logits contains NaN/Inf in chunk {start}:{end}")
            if torch.any(torch.linalg.vector_norm(tensor, dim=-1) <= 1e-12):
                raise FloatingPointError(f"{label} has a near-zero logits vector in chunk {start}:{end}")

        token_cosines = F.cosine_similarity(reference, candidate, dim=-1, eps=1e-8)
        cosines.append(token_cosines.cpu())
        diff = reference - candidate
        abs_diff = diff.abs()
        element_count += diff.numel()
        sum_abs += torch.sum(abs_diff, dtype=torch.float64).item()
        sum_square += torch.sum(diff.square(), dtype=torch.float64).item()
        max_abs = max(max_abs, abs_diff.max().item())
        flat_dot += torch.sum(reference * candidate, dtype=torch.float64).item()
        flat_reference_square += torch.sum(reference.square(), dtype=torch.float64).item()
        flat_candidate_square += torch.sum(candidate.square(), dtype=torch.float64).item()

        top1_matches += int(reference.argmax(dim=-1).eq(candidate.argmax(dim=-1)).sum())
        reference_topk = reference.topk(topk, dim=-1).indices
        candidate_topk = candidate.topk(topk, dim=-1).indices
        overlap = reference_topk.unsqueeze(-1).eq(candidate_topk.unsqueeze(-2)).any(dim=-1)
        topk_overlap_sum += overlap.float().sum().item() / float(topk)

        per_token_max = abs_diff.amax(dim=-1)
        local_index = int(per_token_max.argmax())
        local_value = per_token_max[local_index].item()
        if local_value > worst_token_max_abs:
            worst_token_max_abs = local_value
            worst_token_position = valid_positions[start + local_index].tolist()

    token_cosine = torch.cat(cosines)
    valid_count = reference_valid.shape[0]
    return {
        "average_token_cosine": token_cosine.mean().item(),
        "minimum_token_cosine": token_cosine.min().item(),
        "maximum_token_cosine": token_cosine.max().item(),
        "p01_token_cosine": torch.quantile(token_cosine, 0.01).item(),
        "p05_token_cosine": torch.quantile(token_cosine, 0.05).item(),
        "median_token_cosine": torch.quantile(token_cosine, 0.50).item(),
        "flatten_cosine": flat_dot / math.sqrt(flat_reference_square * flat_candidate_square),
        "max_abs": max_abs,
        "mean_abs": sum_abs / element_count,
        "rms_error": math.sqrt(sum_square / element_count),
        "top1_match_rate": top1_matches / valid_count,
        f"top{topk}_mean_overlap": topk_overlap_sum / valid_count,
        "worst_token_position": worst_token_position,
        "worst_token_max_abs": worst_token_max_abs,
        "_token_cosine": token_cosine,
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    expected_reference = None if args.skip_framework_check else args.reference_framework
    expected_candidate = None if args.skip_framework_check else args.candidate_framework
    reference_payload = load_output(
        args.reference_file,
        expected_framework=expected_reference,
        label=args.reference_label,
    )
    candidate_payload = load_output(
        args.candidate_file,
        expected_framework=expected_candidate,
        label=args.candidate_label,
    )
    candidate_execution = validate_candidate_execution(candidate_payload, args)
    input_ids, attention_mask, digest = validate_inputs(
        reference_payload,
        candidate_payload,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
    )

    reference_logits = reference_payload["logits"]
    candidate_logits = candidate_payload["logits"]
    if not torch.is_tensor(reference_logits) or not torch.is_tensor(candidate_logits):
        raise TypeError("Both logits values must be tensors.")
    if reference_logits.shape != candidate_logits.shape:
        raise AssertionError(
            f"logits shape differs: reference={tuple(reference_logits.shape)}, "
            f"candidate={tuple(candidate_logits.shape)}"
        )
    if reference_logits.ndim != 3 or reference_logits.shape[:2] != input_ids.shape:
        raise AssertionError(
            f"Expected logits [B,S,V] for input {tuple(input_ids.shape)}, "
            f"got {tuple(reference_logits.shape)}"
        )

    valid_mask = attention_mask.bool()
    valid_positions = torch.nonzero(valid_mask, as_tuple=False)
    if valid_positions.numel() == 0:
        raise ValueError("There are no valid tokens to compare.")
    topk = min(args.topk, reference_logits.shape[-1])
    if topk <= 0:
        raise ValueError("--topk must be positive.")

    metrics = compare_in_chunks(
        reference_logits[valid_mask],
        candidate_logits[valid_mask],
        valid_positions,
        chunk_size=args.token_chunk_size,
        topk=topk,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
    )
    token_cosine = metrics.pop("_token_cosine")
    metrics["first_valid_token_cosine"] = token_cosine[0].item()
    if 0 < args.max_token_cosines_in_report >= token_cosine.numel():
        metrics["token_positions"] = valid_positions.tolist()
        metrics["token_cosines"] = token_cosine.tolist()
    last_cosines = []
    offset = 0
    for length in attention_mask.sum(dim=-1).tolist():
        last_cosines.append(token_cosine[offset + int(length) - 1].item())
        offset += int(length)
    metrics["last_valid_token_cosines"] = last_cosines
    metrics["mean_last_valid_token_cosine"] = sum(last_cosines) / len(last_cosines)
    metrics["average_token_cosine_percent"] = metrics["average_token_cosine"] * 100.0

    passed = math.isfinite(metrics["average_token_cosine"]) and (
        metrics["average_token_cosine"] >= args.threshold
    )
    reference_source = {
        "label": args.reference_label,
        "framework": reference_payload.get("framework"),
        "file": str(Path(args.reference_file).resolve()),
        "metadata": reference_payload.get("metadata", {}),
    }
    candidate_source = {
        "label": args.candidate_label,
        "framework": candidate_payload.get("framework"),
        "file": str(Path(args.candidate_file).resolve()),
        "metadata": candidate_payload.get("metadata", {}),
    }
    report = {
        "format_version": 2,
        "passed": passed,
        "primary_metric": "average_token_cosine",
        "threshold": args.threshold,
        "threshold_percent": args.threshold * 100.0,
        "input_sha256": digest,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(reference_logits.shape),
        "valid_token_count": int(valid_mask.sum()),
        "metrics": metrics,
        "reference": reference_source,
        "candidate": candidate_source,
        "candidate_execution": candidate_execution,
    }
    # Keep legacy report keys for existing Transformers/MindSpeed consumers.
    if args.reference_framework == "transformers" and args.candidate_framework == "mindspeed-llm":
        report["hf"] = reference_source
        report["mindspeed"] = candidate_source
    return to_json_compatible(report), passed


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1].")
    if args.token_chunk_size <= 0:
        raise ValueError("--token-chunk-size must be positive.")
    if args.max_token_cosines_in_report < 0:
        raise ValueError("--max-token-cosines-in-report must be non-negative.")

    report, passed = build_report(args)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    relation = ">=" if passed else "<"
    status = "PASS" if passed else "FAIL"
    metric = report["metrics"]["average_token_cosine"]
    print(f"\n[{status}] average valid-token logits cosine {metric:.6f} {relation} {args.threshold:.6f}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
