#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze Ascend op_statistic.csv against the batch invariant op allowlist."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

# Skills are siblings in both source checkouts and materialized plugin installs.
ENVIRONMENT_SCRIPTS = Path(__file__).resolve().parents[2] / "batch-invariance-environment" / "scripts"
sys.path.insert(0, str(ENVIRONMENT_SCRIPTS))
from check_batch_invariant_env import classify_chip, detect_chip


DEFAULT_ALLOWLIST = (
    Path(__file__).resolve().parents[1] / "references" / "batch_invariant_ops.md"
)

CHIP_FAMILIES = {"A2", "A3", "A5"}

PREFERRED_COLUMNS = {
    "op_type",
    "op type",
    "op name",
    "op_name",
    "name",
    "kernel name",
    "kernel_name",
    "task name",
    "task_name",
}

# Only the non-batch-invariant base operators below are hard failures. A repair
# must inspect the real model/vLLM-Ascend call path and chip implementation; the
# analyzer deliberately does not infer a replacement by appending a suffix.
BLACKLIST_OPS = {
    "SoftmaxV2": "non-batch-invariant SoftmaxV2 base operator observed",
    "MatMulV2": "non-batch-invariant MatMulV2 base operator observed",
    "FusedInferAttentionScore": "non-batch-invariant FusedInferAttentionScore base operator observed",
    "AddRmsNorm": "non-batch-invariant AddRmsNorm base operator observed",
    "ReduceSum": "non-batch-invariant ReduceSum base operator observed",
}
COMMUNICATION_OPS = {
    "allreduceAicpuKernel",
    "allgatherAicpuKernel",
    "hcom_allReduce",
    "hcom_allGather",
}




def soc_version_to_chip_family(soc_version: str) -> str:
    return classify_chip(soc_version)


def resolve_chip_family(chip_type: str) -> tuple[str, str]:
    if chip_type != "auto":
        return soc_version_to_chip_family(chip_type), chip_type

    detected_soc_version, _ = detect_chip()
    if detected_soc_version:
        return soc_version_to_chip_family(detected_soc_version), detected_soc_version

    raise ValueError("BLOCKED_CHIP_DETECTION: cannot determine chip type from SOC_VERSION or npu-smi")


def extract_allowlist_ops(lines: list[str]) -> set[str]:
    ops: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("|") or line.startswith(("-", "*")):
            ops.update(
                token
                for token in re.findall(r"`([^`]+)`", line)
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
            )
            continue

        if line.startswith("```"):
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line):
            ops.add(line)

    return ops


def split_allowlist_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        section_match = re.match(r"^#{2,}\s*(A[235])\b", line.strip(), re.IGNORECASE)
        if section_match:
            current = "A3" if section_match.group(1).upper() == "A2" else section_match.group(1).upper()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return sections


def load_allowlist(path: Path, chip_family: str) -> set[str]:
    chip_family = "A3" if chip_family == "A2" else chip_family
    if not path.is_file():
        raise FileNotFoundError(f"allowlist not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = split_allowlist_sections(lines)
    if not sections:
        return extract_allowlist_ops(lines)
    return extract_allowlist_ops(sections.get(chip_family, []))


def infer_columns(fieldnames: list[str]) -> list[str]:
    fields = [
        col
        for col in fieldnames
        if col and col.strip().lower() in PREFERRED_COLUMNS
    ]
    if fields:
        return fields
    fields = [
        col
        for col in fieldnames
        if col and ("op" in col.lower() or "kernel" in col.lower())
    ]
    if fields:
        return fields
    raise ValueError(f"cannot infer op columns from: {fieldnames}")


def extract_op_name(value: str, allowlist: set[str]) -> str | None:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value)
    if not tokens:
        return None
    for token in reversed(tokens):
        if token in allowlist:
            return token
    return tokens[-1]


def collect_ops(path: Path, allowlist: set[str]) -> set[str]:
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty or invalid csv")
        fields = infer_columns(reader.fieldnames)
        for row in reader:
            for col in fields:
                op_name = extract_op_name((row.get(col) or "").strip(), allowlist)
                if op_name:
                    seen.add(op_name)
    return seen


def classify_blacklisted_ops(unsupported: set[str]) -> dict[str, list[str]]:
    """Return exact blacklist base names after allowlist filtering."""
    matches: dict[str, list[str]] = {}
    for op_name in sorted(unsupported):
        reason = BLACKLIST_OPS.get(op_name)
        if reason is not None:
            matches.setdefault(op_name, []).append(reason)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_csv", type=Path, help="Path to ASCEND_PROFILER_OUTPUT/op_statistic.csv")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--chip-type",
        choices=["auto", *sorted(CHIP_FAMILIES)],
        default="auto",
        help="Chip family used to select allowlist section. A2 currently uses the A3 section.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Whitelist-external operators always "
            "return REVIEW_REQUIRED with exit 2; blacklist matches return exit 3."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--environment-report", type=Path, help="Use the successful preflight JSON report instead of redetecting the chip.")
    args = parser.parse_args()

    try:
        if args.environment_report:
            report = json.loads(args.environment_report.read_text(encoding="utf-8"))
            if report.get("outcome") != "PASS" or report.get("installation_state") != "READY":
                raise ValueError("environment report must be PASS with installation_state=READY")
            detected_soc_version = report["chip_raw"]
            chip_family = soc_version_to_chip_family(detected_soc_version)
            if report.get("chip") != chip_family:
                raise ValueError("environment report chip classification mismatch")
            if args.chip_type != "auto" and args.chip_type != chip_family:
                raise ValueError("--chip-type conflicts with environment report")
        else:
            chip_family, detected_soc_version = resolve_chip_family(args.chip_type)
    except (ValueError, KeyError, OSError) as exc:
        print(f"BLOCKED_CHIP_DETECTION: {exc}", file=sys.stderr)
        return 4
    allowlist = load_allowlist(args.allowlist, chip_family)
    seen = collect_ops(args.profile_csv, allowlist)
    supported_seen = sorted(seen & allowlist)
    unsupported_seen = sorted(seen - allowlist)
    # Blacklist triage applies only after allowlist filtering. A listed
    # BatchInvariant variant is accepted and never enters this branch.
    blacklisted_seen = classify_blacklisted_ops(set(unsupported_seen))
    communication_ops_seen = sorted(seen & COMMUNICATION_OPS)
    moe_ops_seen = sorted(op for op in seen if "Moe" in op or "MoE" in op)

    if blacklisted_seen:
        decision = "FAIL_BLACKLIST"
        batch_invariance_satisfied: bool | None = False
    elif unsupported_seen:
        decision = "REVIEW_REQUIRED"
        batch_invariance_satisfied = None
    else:
        decision = "PASS_TRIAGE"
        batch_invariance_satisfied = True

    result = {
        "profile_csv": str(args.profile_csv),
        "allowlist": str(args.allowlist),
        "chip_type": chip_family,
        "allowlist_family": "A3" if chip_family == "A2" else chip_family,
        "detected_soc_version": detected_soc_version,
        "supported_seen": supported_seen,
        "unsupported_seen": unsupported_seen,
        "blacklisted_seen": blacklisted_seen,
        "blacklist_skipped_single_op_validation": sorted(blacklisted_seen),
        "replacement_requires_code_inspection": sorted(blacklisted_seen),
        "replacement_name_inferred": False,
        "batch_invariance_satisfied": batch_invariance_satisfied,
        "decision": decision,
        "communication_ops_seen": communication_ops_seen,
        "moe_ops_seen": moe_ops_seen,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("chip_type:")
        print(f"  {chip_family}")
        print("detected_soc_version:")
        print(f"  {detected_soc_version or '<none>'}")
        for title, values in (
            ("communication_ops_seen", communication_ops_seen),
            ("moe_ops_seen", moe_ops_seen),
            ("supported_seen", supported_seen),
            ("unsupported_seen", unsupported_seen),
        ):
            print(f"{title}:")
            if values:
                for value in values:
                    print(f"  {value}")
            else:
                print("  <none>")

        print("blacklisted_seen:")
        if blacklisted_seen:
            for reason, values in blacklisted_seen.items():
                print(f"  {reason}: {', '.join(values)}")
        else:
            print("  <none>")
        print("decision:")
        print(f"  {decision}")

    if blacklisted_seen:
        return 3
    if unsupported_seen:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
