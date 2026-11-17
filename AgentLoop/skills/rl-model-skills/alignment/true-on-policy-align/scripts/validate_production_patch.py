#!/usr/bin/env python3
"""Fail closed when a production patch or clean runtime retains diagnostic material."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


DIAGNOSTIC_PATTERNS = (
    ("diagnostic_env", r"TRUE_ON_POLICY[^\s\"']*(?:CAPTURE|REPLAY|FORCE|DUMP|FINGERPRINT)"),
    ("runtime_hook", r"register_(?:forward_)?(?:pre_)?hook\s*\("),
    ("tensor_write", r"torch\.save\s*\("),
    ("replay_code", r"(?:ReplayStore|replay_chain\.json|tensor_replay)"),
    ("forced_response", r"(?:FORCE_RESPONSE|forced_response_capture)"),
    ("diagnostic_fingerprint", r"(?i)(?:weight[_-]?sync[_-]?fingerprint|_write_[A-Za-z0-9_]*fingerprints?|diagnostic[_-]?fingerprint)"),
)
PRIVATE_PATTERNS = (
    ("private_ipv4", r"(?<!\d)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?!\d)"),
    ("private_posix_path", r"(?<![A-Za-z0-9])/(?:opt|home|root|mnt|data)/[^\s\"')>]+"),
    ("private_windows_path", r"(?i)\b[A-Z]:\\(?:Users|Work|Data)\\[^\s\"']+"),
    ("embedded_url_credentials", r"https?://[^\s/:@]+:[^\s/@]+@"),
    ("credential_assignment", r"(?i)\b(?:password|passwd|secret|api[_-]?key|(?:access|auth|bearer|hf)[_-]?token)\b[\"']?\s*[:=]\s*[^\s,;}]+"),
    ("private_key", r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
DIAGNOSTIC_ENV = re.compile(
    r"TRUE_ON_POLICY[^\s]*(?:CAPTURE|REPLAY|FORCE|DUMP|FINGERPRINT)",
    re.IGNORECASE,
)


class PatchAuditError(ValueError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PatchAuditError(f"JSON root must be an object: {path}")
    return value


def active_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "off", "none", "null"}


def scan_patch(
    text: str,
    *,
    allow_patterns: list[str] | None = None,
    check_private: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        allow = [re.compile(pattern) for pattern in (allow_patterns or [])]
    except re.error as exc:
        raise PatchAuditError(f"invalid allow pattern: {exc}") from exc
    patterns = list(DIAGNOSTIC_PATTERNS)
    if check_private:
        patterns.extend(PRIVATE_PATTERNS)
    compiled = [(name, re.compile(pattern)) for name, pattern in patterns]
    violations: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    current_file = "unknown"
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        content = line[1:]
        for rule, pattern in compiled:
            match = pattern.search(content)
            if not match:
                continue
            finding = {
                "rule": rule,
                "file": current_file,
                "patch_line": line_number,
                "match": match.group(0)[:160],
                "line_sha256": "sha256:"
                + hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
            if any(candidate.search(f"{current_file}:{content}") for candidate in allow):
                allowed.append(finding)
            else:
                violations.append(finding)
    return violations, allowed


def validate_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if profile.get("truth_role") != "production_alignment_profile":
        violations.append({"rule": "profile_truth_role", "detail": "invalid truth_role"})
    if profile.get("active") is not True:
        violations.append({"rule": "profile_activation", "detail": "active must be true"})
    for key in ("diagnostic_capture_enabled", "diagnostic_replay_enabled"):
        if profile.get(key) is not False:
            violations.append({"rule": "profile_diagnostic_flag", "detail": f"{key} must be false"})
    features = profile.get("features")
    if not isinstance(features, list) or not all(
        isinstance(item, str) and item.strip() for item in features
    ):
        violations.append({"rule": "profile_features", "detail": "features must be a string list"})
    if not isinstance(profile.get("activation_evidence"), str) or not profile["activation_evidence"].strip():
        violations.append({"rule": "profile_evidence", "detail": "activation_evidence is required"})
    return violations


def audit(
    patch_path: Path,
    *,
    allow_patterns: list[str] | None = None,
    check_private: bool = True,
    env: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    require_clean_runtime: bool = False,
) -> dict[str, Any]:
    try:
        raw = patch_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PatchAuditError(f"cannot read UTF-8 patch {patch_path}: {exc}") from exc
    violations, allowed = scan_patch(
        text,
        allow_patterns=allow_patterns,
        check_private=check_private,
    )
    added_line_violations = len(violations)
    clean_env_checked = env is not None
    if env is not None:
        for key, value in env.items():
            if isinstance(key, str) and DIAGNOSTIC_ENV.search(key) and active_value(value):
                violations.append(
                    {"rule": "active_diagnostic_environment", "name": key, "value": "<redacted>"}
                )
    profile_checked = profile is not None
    if profile is not None:
        violations.extend(validate_profile(profile))
    if require_clean_runtime and not clean_env_checked:
        violations.append({"rule": "clean_runtime_evidence_missing", "detail": "--env-json is required"})
    if require_clean_runtime and not profile_checked:
        violations.append({"rule": "production_profile_missing", "detail": "--profile-json is required"})
    return {
        "schema_version": 1,
        "truth_role": "production_patch_hygiene_gate",
        "status": "pass" if not violations else "fail",
        "patch": str(patch_path),
        "patch_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "violation_count": len(violations),
        "added_line_violations": added_line_violations,
        "violations": violations,
        "allowed_findings": allowed,
        "clean_runtime_env_checked": clean_env_checked,
        "production_profile_checked": profile_checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--skip-private-scan", action="store_true")
    parser.add_argument("--env-json", type=Path)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--require-clean-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(
            args.patch.resolve(),
            allow_patterns=args.allow_pattern,
            check_private=not args.skip_private_scan,
            env=load_object(args.env_json.resolve()) if args.env_json else None,
            profile=load_object(args.profile_json.resolve()) if args.profile_json else None,
            require_clean_runtime=args.require_clean_runtime,
        )
    except PatchAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
