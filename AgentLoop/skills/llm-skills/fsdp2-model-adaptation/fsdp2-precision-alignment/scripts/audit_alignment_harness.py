#!/usr/bin/env python3
"""Audit environment ownership and shell safety in a model alignment harness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCE_GUARD = 'if [[ "${BASH_SOURCE[0]}" != "$0" ]]'
GLOBAL_REPO_PYTHONPATH = re.compile(
    r"(?m)^\s*export\s+PYTHONPATH="
    r"(?![^\n]*\$\{PYTHONPATH:\+)[^\n]*(?:\$REPO_ROOT|\$\{REPO_ROOT\})[^\n]*$"
)
ENVIRONMENT_SOURCE = re.compile(
    r"(?m)^\s*(?:source|\.)\s+\S*(?:set_env\.sh|setenv\.bash|/activate)\b"
)
CONDA_ACTIVATE = re.compile(r"(?m)^\s*conda\s+activate\b")
PLACEHOLDER_BASE_PYTHON = re.compile(
    r"""(?mx)^\s*(?:export\s+)?BASE_PYTHON\s*=\s*["']?
    /absolute/path(?:/[^"'\s]*)?["']?\s*$"""
)
DISABLED_MODULE_OUTPUT_SYNC_DEFAULT = re.compile(
    r"(?:SYNCHRONIZE_MODULE_OUTPUTS|synchronize_module_outputs)[^\n]*:-0",
    re.IGNORECASE,
)
RESOLVED_EXECUTABLE_OWNERSHIP = re.compile(
    r"Path\(\s*sys\.executable\s*\)\.resolve\(\)"
)
PRODUCTION_MASK_REJECTION_MARKERS = (
    "rejected_outer_rank4_mask",
    "unexpectedly accepted an outer rank-4 mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment_dir", type=Path)
    parser.add_argument("--report-file", type=Path)
    return parser.parse_args()


def audit(alignment_dir: Path) -> dict[str, object]:
    alignment_dir = alignment_dir.resolve()
    errors: list[str] = []
    checks: dict[str, object] = {}
    if not alignment_dir.is_dir():
        return {
            "format_version": 1,
            "passed": False,
            "alignment_dir": str(alignment_dir),
            "checks": {},
            "errors": [f"Alignment directory does not exist: {alignment_dir}"],
        }

    shell_files = sorted(alignment_dir.rglob("*.sh"))
    checks["shell_entrypoints"] = [
        str(path.relative_to(alignment_dir)) for path in shell_files
    ]
    for path in shell_files:
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(alignment_dir)
        if SOURCE_GUARD not in source:
            errors.append(f"{relative}: shell entrypoint does not reject sourcing")
        if GLOBAL_REPO_PYTHONPATH.search(source):
            errors.append(f"{relative}: globally overwrites PYTHONPATH with REPO_ROOT")
        if ENVIRONMENT_SOURCE.search(source):
            errors.append(f"{relative}: sources an environment/toolkit script")
        if CONDA_ACTIVATE.search(source):
            errors.append(f"{relative}: activates a Conda environment")
        if PLACEHOLDER_BASE_PYTHON.search(source):
            errors.append(
                f"{relative}: assigns a documentation placeholder to BASE_PYTHON"
            )
        if DISABLED_MODULE_OUTPUT_SYNC_DEFAULT.search(source):
            errors.append(
                f"{relative}: exposes module-output synchronization "
                "but disables it by default"
            )
        if "--reuse-reference" in source and "validate" not in source.lower():
            errors.append(
                f"{relative}: reuses reference artifacts without "
                "a compatibility-validation stage"
            )

        is_mcore_launcher = (
            "torch.distributed.run" in source
            or "tests/tools/mcore/precision_alignment/dump_logits.py" in source
            or "scripts/precision_alignment/dump_logits.py" in source
        )
        if is_mcore_launcher:
            if "CANN_SET_ENV" in source:
                errors.append(f"{relative}: MCore launcher selects a CANN environment")
            if 'PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' not in source:
                errors.append(
                    f"{relative}: MCore launcher does not preserve inherited PYTHONPATH"
                )

        if " -m pip " in source or '" -m pip ' in source or "-m pip" in source:
            if "PIP_REQUIRE_VIRTUALENV=true" not in source:
                errors.append(f"{relative}: pip is not protected by PIP_REQUIRE_VIRTUALENV")
            if '"${BASE_PYTHON}" -m pip' in source:
                errors.append(f"{relative}: installs packages through BASE_PYTHON")
            if "PYTHONPATH=" not in source or "PYTHONHOME=" not in source:
                errors.append(f"{relative}: pip does not clear Python path overrides")
            if "--upgrade" not in source:
                errors.append(
                    f"{relative}: pip does not override incompatible inherited packages"
                )

    python_files = sorted(alignment_dir.rglob("*.py"))
    for path in python_files:
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(alignment_dir)
        if RESOLVED_EXECUTABLE_OWNERSHIP.search(source):
            errors.append(
                f"{relative}: resolves sys.executable before venv ownership checks"
            )
        if "expected_venv" in source or "HF_VENV" in source:
            if "sys.prefix" in source and "sys.base_prefix" not in source:
                errors.append(
                    f"{relative}: venv check uses sys.prefix without sys.base_prefix"
                )
        if (
            not path.name.startswith("test_")
            and any(marker in source for marker in PRODUCTION_MASK_REJECTION_MARKERS)
        ):
            errors.append(
                f"{relative}: treats rejection of a production-valid rank-4 DSA mask "
                "as a passing regression"
            )

    text_files = sorted(
        path
        for suffix in ("*.md", "*.sh", "*.py")
        for path in alignment_dir.rglob(suffix)
    )
    global_exports = []
    for path in text_files:
        source = path.read_text(encoding="utf-8")
        if GLOBAL_REPO_PYTHONPATH.search(source):
            global_exports.append(str(path.relative_to(alignment_dir)))
    checks["global_repo_pythonpath_exports"] = global_exports
    if global_exports:
        errors.append(
            "Parent-shell PYTHONPATH overwrite appears in: " + ", ".join(global_exports)
        )

    checks["mcore_environment_policy"] = (
        "inherit-active-device-environment-without-toolkit-selection"
    )
    checks["hf_install_policy"] = "venv-only-when-pip-is-present"
    checks["base_python_policy"] = (
        "reject-placeholders-and-validate-active-interpreter"
    )
    checks["venv_ownership_policy"] = "lexical-launcher-plus-prefixes"
    checks["producer_consumer_policy"] = "production-valid-input-must-be-accepted"
    checks["reference_reuse_policy"] = "validate-producer-contract-not-file-existence"
    checks["known_hf_scheduling_barrier_policy"] = "mandatory-and-recorded"
    return {
        "format_version": 1,
        "passed": not errors,
        "alignment_dir": str(alignment_dir),
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = audit(args.alignment_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
