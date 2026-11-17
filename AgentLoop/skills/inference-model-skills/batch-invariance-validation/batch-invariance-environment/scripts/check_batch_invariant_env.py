#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for vLLM Ascend batch invariant custom ops."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlparse

from chip_detection import get_chip_type

REQUIRED_OPS = (
    "npu_fused_infer_attention_score_batch_invariant",
    "npu_matmul_batch_invariant",
    "npu_reduce_sum_batch_invariant",
)


def classify_chip(chip: str) -> str:
    """Normalize chip names using the agreed A5, A2, then A3 precedence."""
    normalized = chip.strip().lower()
    if chip_is_a5(normalized):
        return "A5"
    if normalized == "a2" or "910b" in normalized or "a2g3" in normalized:
        return "A2"
    if normalized == "a3" or "910" in normalized:
        return "A3"
    raise ValueError(f"unsupported or undetected chip: {chip!r}")


def detect_chip() -> tuple[str, str]:
    configured = os.environ.get("SOC_VERSION", "").strip()
    return (configured, "SOC_VERSION") if configured else (get_chip_type(), "npu-smi")


def resolve_opp_path(explicit: str | None = None) -> tuple[str, str]:
    if explicit and explicit.strip():
        return str(Path(explicit).resolve()), "--opp-path"
    environment = os.environ.get("ASCEND_OPP_PATH", "").strip()
    if environment:
        return str(Path(environment).resolve()), "ASCEND_OPP_PATH"
    return "/usr/local/Ascend/cann-9.0.0/opp", "default"


def find_build_script(pip_output: str, source_root: str = "") -> Path | None:
    roots = [Path(source_root)] if source_root and source_root != "unavailable" else []
    for line in pip_output.splitlines():
        if line.startswith(("Location:", "Editable project location:")):
            roots.append(Path(line.split(":", 1)[1].strip()))
    for root in roots:
        for relative in ("csrc/build_batch_invariant_ops.sh", "vllm_ascend/csrc/build_batch_invariant_ops.sh",
                         "vllm-ascend/csrc/build_batch_invariant_ops.sh"):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def parse_timeout(value: str) -> float:
    text = value.strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    return float(text)


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def validate_tp_confirmation(run_config_path: str) -> bool:
    """Require a user-confirmed TP plan before environment admission."""
    path = Path(run_config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"BLOCKED_TP_PLAN_UNCONFIRMED: cannot read {path}: {exc}")
        return False

    confirmation = config.get("tp_confirmation")
    tp_plan = config.get("tp_plan")
    if not isinstance(tp_plan, dict) or not tp_plan:
        print("BLOCKED_TP_PLAN_UNCONFIRMED: run_config.json is missing tp_plan")
        return False
    if not isinstance(confirmation, dict) or confirmation.get("status") != "confirmed":
        print(
            "BLOCKED_TP_PLAN_UNCONFIRMED: run_config.json must contain "
            "tp_confirmation.status=confirmed"
        )
        return False
    missing = [
        key
        for key in ("source", "confirmed_utc", "user_response")
        if not confirmation.get(key)
    ]
    if missing:
        print(
            "BLOCKED_TP_PLAN_UNCONFIRMED: tp_confirmation is missing "
            + ", ".join(missing)
        )
        return False

    print(
        "TP_PLAN_CONFIRMED: "
        + json.dumps(
            {"tp_plan": tp_plan, "confirmation": confirmation},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return True


def package_provenance(distribution_name: str) -> dict[str, str]:
    """Return package version and best-effort Git provenance."""
    unavailable = "unavailable"
    result = {
        "distribution": distribution_name,
        "version": unavailable,
        "git_commit": unavailable,
        "git_branch": unavailable,
        "git_commit_time": unavailable,
        "source_root": unavailable,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return result

    result["version"] = dist.version
    source_candidates: list[Path] = []
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {}
        vcs_info = direct_url.get("vcs_info") or {}
        result["git_commit"] = vcs_info.get("commit_id") or unavailable
        result["git_branch"] = vcs_info.get("requested_revision") or unavailable
        parsed = urlparse(direct_url.get("url", ""))
        if parsed.scheme == "file":
            source_candidates.append(Path(unquote(parsed.path)))

    source_candidates.append(Path(dist.locate_file("")))
    source_root: Path | None = None
    for candidate in source_candidates:
        resolved = candidate.resolve()
        for parent in (resolved, *resolved.parents):
            if (parent / ".git").exists():
                source_root = parent
                break
        if source_root is not None:
            break

    if source_root is not None:
        result["source_root"] = str(source_root)
        git_fields = {
            "git_commit": ["git", "rev-parse", "HEAD"],
            "git_branch": ["git", "branch", "--show-current"],
            "git_commit_time": ["git", "show", "-s", "--format=%cI", "HEAD"],
        }
        for key, command in git_fields.items():
            completed = run([command[0], "-C", str(source_root), *command[1:]])
            value = completed.stdout.strip()
            if completed.returncode == 0 and value:
                result[key] = value
    return result


def chip_is_a5(chip: str) -> bool:
    """Check whether the detected chip is Ascend 950 (A5)."""
    normalized = chip.strip().lower()
    return normalized == "a5" or "950" in normalized


def pick_platform(chip: str) -> str | None:
    """Map A2, A3, and A5 to the upstream build-script argument."""
    try:
        family = classify_chip(chip)
    except ValueError:
        return None
    return {"A2": "ascend910b", "A3": "ascend910_93", "A5": "ascend950"}.get(family)


def install_batch_invariant_ops(report: dict, timeout_s: float) -> tuple[bool, dict]:
    """Build operators from the matching vLLM Ascend source discovered by preflight."""
    build_script_text = report.get("build_script")
    platform = report.get("build_platform")
    installation = {"attempted": True, "platform": platform, "command": None,
                    "returncode": None, "output": ""}
    if not build_script_text or not platform:
        installation["output"] = "matching vLLM Ascend source build script is unavailable"
        return False, installation
    build_script = Path(str(build_script_text)).resolve()
    command = ["bash", str(build_script), str(platform)]
    installation["command"] = command
    try:
        completed = subprocess.run(
            command, cwd=build_script.parent.parent, check=False, text=True,
            encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout_s,
            env={**os.environ, "VLLM_BATCH_INVARIANT": "1",
                 "COMPILE_CUSTOM_KERNELS": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        installation["output"] = str(exc)
        return False, installation
    installation.update(returncode=completed.returncode, output=completed.stdout)
    return completed.returncode == 0, installation


def check_batch_invariant(opp_path: str, timeout_s: float, run_config_path: str,
                          report: dict | None = None, opp_source: str = "--opp-path") -> int:
    """Check the installed batch invariant environment."""
    report = report if report is not None else {}
    report.update(schema_version="1.0", opp_path=str(Path(opp_path).resolve()),
                  opp_path_source=opp_source, installation_state="NOT_CHECKED",
                  installation_capability="UNKNOWN", checks={})

    def finish(code: int, reason: str | None = None) -> int:
        report.update(exit_code=code, outcome="PASS" if code == 0 else "BLOCKED", reason_code=reason)
        print("ENVIRONMENT_RESULT: " + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return code

    if not validate_tp_confirmation(run_config_path):
        return finish(6, "BLOCKED_TP_PLAN_UNCONFIRMED")

    report["runtime_provenance"] = {}
    for distribution in ("vllm", "vllm-ascend"):
        provenance = package_provenance(distribution)
        report["runtime_provenance"][distribution] = provenance
        print(
            "RUNTIME_PROVENANCE: "
            + json.dumps(provenance, sort_keys=True)
        )

    detected_chip, chip_source = detect_chip()
    report.update(chip_raw=detected_chip, chip_source=chip_source)
    try:
        family = classify_chip(detected_chip)
    except ValueError as exc:
        print(str(exc))
        return finish(7, "BLOCKED_CHIP_DETECTION")
    platform = pick_platform(detected_chip)
    report.update(chip=family, allowlist_family="A3" if family == "A2" else family,
                  build_platform=platform,
                  installation_capability="SUPPORTED" if platform else "UNSUPPORTED")
    print(f"Detected chip: {detected_chip} -> {family}; build platform: {platform}")
    print(f"OPP path: {report['opp_path']} (source: {opp_source})")

    ok = True
    missing_runtime_packages: list[str] = []
    vllm_ascend_output = ""
    for distribution in ("vllm", "vllm-ascend"):
        package_show = run([sys.executable, "-m", "pip", "show", distribution])
        if distribution == "vllm-ascend":
            vllm_ascend_output = package_show.stdout
        if package_show.returncode != 0:
            print(f"ERROR: {distribution} is missing.")
            missing_runtime_packages.append(distribution)
            ok = False
        else:
            print(package_show.stdout.strip())

    build_script = find_build_script(vllm_ascend_output,
                                    report["runtime_provenance"]["vllm-ascend"].get("source_root", ""))
    report["build_script"] = str(build_script) if build_script else None
    report["build_command"] = f"bash {shlex.quote(str(build_script))} {platform}" if build_script and platform else None

    pip_show = run([sys.executable, "-m", "pip", "show", "batch_invariant_ops"])
    report["checks"]["python_package"] = pip_show.returncode == 0
    report["package_provenance"] = package_provenance("batch_invariant_ops")
    if pip_show.returncode != 0:
        print("ERROR: batch_invariant_ops is missing.")
        print(pip_show.stdout.strip())
        ok = False
    else:
        print(pip_show.stdout.strip())

    custom_op_so = (
        Path(opp_path)
        / "vendors"
        / "batch_invariant"
        / "op_api"
        / "lib"
        / "libcust_opapi.so"
    )
    report["custom_op_so"] = str(custom_op_so.resolve())
    report["checks"]["so_exists"] = custom_op_so.is_file()
    if not custom_op_so.is_file():
        print(f"ERROR: custom op SO is missing: {custom_op_so}")
        ok = False
    else:
        print(f"{custom_op_so} exists")

    code = """
import ctypes
import json
import sys
import batch_invariant_ops
import torch
ctypes.CDLL(sys.argv[1])
required = json.loads(sys.argv[2])
missing = []
for name in required:
    try:
        op = getattr(torch.ops.batch_invariant_ops, name)
        assert op.overloads()
        print(name, op)
    except (AttributeError, AssertionError):
        missing.append(name)
print('Missing required operators:', missing)
print('OPERATOR_INVENTORY: ' + json.dumps({
    'missing': missing,
    'discovered': sorted(name for name in dir(torch.ops.batch_invariant_ops) if name.endswith('_batch_invariant')),
}))
sys.exit(1 if missing else 0)
"""
    report["required_operators"] = list(REQUIRED_OPS)
    report["checks"]["so_load_and_torch_registration"] = False
    try:
        check = run([sys.executable, "-c", code, str(custom_op_so), json.dumps(REQUIRED_OPS)], timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"ERROR: torch op registration check timed out after {timeout_s:.0f}s.")
        ok = False
        report["operator_check_output"] = "timeout"
    else:
        report["operator_check_output"] = check.stdout
        for line in check.stdout.splitlines():
            if line.startswith("OPERATOR_INVENTORY: "):
                report["operator_inventory"] = json.loads(line.removeprefix("OPERATOR_INVENTORY: "))
        report["checks"]["so_load_and_torch_registration"] = check.returncode == 0
        print(check.stdout.strip())
        if check.returncode != 0:
            print("ERROR: torch op registration check failed.")
            print("Expected operators: " + ", ".join(REQUIRED_OPS))
            ok = False

    report["missing_runtime_packages"] = missing_runtime_packages
    report["installation_state"] = (
        "READY" if all(report["checks"].values()) else
        "MISSING" if not report["checks"]["python_package"] or not custom_op_so.is_file() else "BROKEN"
    )
    if ok:
        print("[PASS] batch invariant custom op registration is available")
        return finish(0)

    if missing_runtime_packages:
        print()
        print("BLOCKED_RUNTIME_PACKAGE_MISSING")
        print("Missing runtime packages: " + ", ".join(missing_runtime_packages))
        print("Install the missing packages manually, then rerun this preflight.")
        return finish(5, "BLOCKED_RUNTIME_PACKAGE_MISSING")

    # ---------- Recovery guidance ----------
    print()
    print("=" * 60)
    print("Recovery guidance")
    print("=" * 60)
    show_recovery_guidance(detected_chip)
    return finish(1, "BLOCKED_OPERATOR_INSTALLATION")


def show_recovery_guidance(chip: str | None = None) -> None:
    """Print recovery steps following the SKILL.md flow."""
    # Step 1: check vLLM Ascend installation
    chip = detect_chip()[0] if chip is None else chip
    vllm_show = run([sys.executable, "-m", "pip", "show", "vllm-ascend"])
    if vllm_show.returncode != 0:
        print("vLLM Ascend is NOT installed.")
        print()
        print("==> Install vLLM Ascend manually, then rerun this preflight.")
        return

    # Step 2a: vLLM Ascend is installed — find build script
    print("vLLM Ascend is installed.")
    build_script = find_build_script(vllm_show.stdout, package_provenance("vllm-ascend").get("source_root", ""))
    if build_script is None:
        print("ERROR: build_batch_invariant_ops.sh not found in the installed or editable source locations")
        print()
        print("==> Manual: the batch invariant operator package must be installed manually.")
        print("    After installation, rerun this script before starting Phase 1/2.")
        return

    print(f"Found: {build_script}")

    # Map chip to platform argument
    platform = pick_platform(chip) if chip else None

    print()
    print("==> Build batch invariant ops manually")
    if platform is not None:
        print(f"    Detected chip: {chip} → platform = {platform}")
        print(f"    bash {shlex.quote(str(build_script))} {platform}")
    else:
        if chip:
            print(f"    Detected chip: {chip} (no automatic mapping)")
        else:
            print("    Chip type could not be auto-detected.")
        print("    Choose the platform argument by chip type:")
        print("      ascend910b     → A2")
        print("      ascend910_93   → A3")
        print("      ascend950      → A5")
        print(f"    bash {build_script} <platform>")

    print()
    print("    After building, rerun this script to confirm the environment is ready.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-config",
        help="run_config.json containing a confirmed tp_plan and tp_confirmation",
    )
    parser.add_argument("--detect-chip", action="store_true", help="Only detect and normalize the chip; no environment admission or installation checks.")
    parser.add_argument(
        "--opp-path",
        default=None,
        help="Ascend OPP path. Defaults to ASCEND_OPP_PATH or CANN 9.0.0 OPP.",
    )
    parser.add_argument(
        "--timeout",
        default="120s",
        help="Timeout for torch op registration check, e.g. 120s, 2m, 30000ms.",
    )
    parser.add_argument("--report-json", type=Path, help="Write structured preflight results to this file.")
    parser.add_argument(
        "--auto-install", action=argparse.BooleanOptionalAction, default=True,
        help="Automatically build missing batch-invariant operators from matching vLLM Ascend source.",
    )
    args = parser.parse_args()
    if args.detect_chip:
        raw, source = detect_chip()
        try:
            family = classify_chip(raw)
        except ValueError:
            print(json.dumps({"chip_raw": raw, "chip_source": source, "chip": None,
                              "reason_code": "BLOCKED_CHIP_DETECTION"}))
            return 7
        print(json.dumps({"chip_raw": raw, "chip_source": source, "chip": family,
                          "build_platform": pick_platform(raw)}, ensure_ascii=False))
        return 0
    if not args.run_config:
        parser.error("--run-config is required for environment admission")
    opp_path, source = resolve_opp_path(args.opp_path)
    report: dict = {}
    result = check_batch_invariant(
        opp_path,
        parse_timeout(args.timeout),
        args.run_config,
        report, source,
    )
    if (result != 0 and args.auto_install
            and report.get("reason_code") == "BLOCKED_OPERATOR_INSTALLATION"):
        installed, installation = install_batch_invariant_ops(report, parse_timeout(args.timeout))
        first_report = dict(report)
        report.clear()
        report["auto_installation"] = installation
        report["pre_install_check"] = first_report
        if installed:
            post_report: dict = {}
            result = check_batch_invariant(
                opp_path, parse_timeout(args.timeout), args.run_config,
                post_report, source,
            )
            installation_copy = report["auto_installation"]
            first_copy = report["pre_install_check"]
            report.clear()
            report.update(post_report, auto_installation=installation_copy,
                          pre_install_check=first_copy)
        else:
            report.update(exit_code=1, outcome="BLOCKED",
                          reason_code="BLOCKED_OPERATOR_INSTALLATION")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
