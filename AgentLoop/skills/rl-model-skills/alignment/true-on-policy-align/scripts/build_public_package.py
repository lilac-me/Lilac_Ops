#!/usr/bin/env python3
"""Build or verify an allowlisted, sanitized public alignment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


BLOCKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".dump",
    ".core",
    ".log",
}
SENSITIVE_PATTERNS = (
    ("private_ipv4", re.compile(r"(?<!\d)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?!\d)")),
    ("private_posix_path", re.compile(r"(?<![A-Za-z0-9])/(?:opt|home|root|mnt|data)/[^\s\"')>]+")),
    ("private_windows_path", re.compile(r"\b[A-Z]:\\(?:Users|Work|Data)\\[^\s\"']+", re.IGNORECASE)),
    ("url_credentials", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@")),
    ("credential_assignment", re.compile(r"\b(?:password|passwd|secret|api[_-]?key|(?:access|auth|bearer|hf)[_-]?token)\b[\"']?\s*[:=]\s*(?![\"']?(?:<redacted>|redacted|null|none|\*+)[\"']?(?:\s|[,;}]))[^\s,;}]+", re.IGNORECASE)),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class PackageError(ValueError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageError("JSON root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def safe_relative(raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise PackageError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise PackageError(f"{label} must stay within its declared root")
    return path


def scan_text(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PackageError(f"non-UTF-8 file is not allowed in public package: {path}") from exc
    findings: list[dict[str, Any]] = []
    for rule, pattern in SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append({"rule": rule, "match": match.group(0)[:160]})
    return findings


def build(
    source_root: Path,
    allowlist_path: Path,
    output_dir: Path,
    *,
    max_file_bytes: int,
) -> dict[str, Any]:
    allowlist = load_object(allowlist_path)
    if allowlist.get("schema_version") != 1:
        raise PackageError("allowlist.schema_version must be 1")
    entries = allowlist.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PackageError("allowlist.entries must be a non-empty list")
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise PackageError(f"source root does not exist: {source_root}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PackageError(f"refusing to overwrite existing output directory: {output_dir}")
    normalized: list[tuple[Path, Path]] = []
    destinations: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PackageError(f"entries[{index}] must be an object")
        source_rel = safe_relative(entry.get("source"), f"entries[{index}].source")
        destination_rel = safe_relative(
            entry.get("destination", entry.get("source")),
            f"entries[{index}].destination",
        )
        source = (source_root / source_rel).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise PackageError(f"source escapes root: {source_rel}") from exc
        if not source.is_file() or source.is_symlink():
            raise PackageError(f"source must be a regular non-symlink file: {source_rel}")
        if source.suffix.lower() in BLOCKED_SUFFIXES:
            raise PackageError(f"blocked raw artifact type: {source_rel}")
        size = source.stat().st_size
        if size > max_file_bytes:
            raise PackageError(f"file exceeds max_file_bytes: {source_rel}")
        findings = scan_text(source)
        if findings:
            raise PackageError(f"sensitive content in {source_rel}: {findings}")
        destination_key = destination_rel.as_posix()
        if destination_key in destinations:
            raise PackageError(f"duplicate destination: {destination_key}")
        destinations.add(destination_key)
        normalized.append((source, destination_rel))

    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for source, destination_rel in normalized:
        target = output_dir / destination_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": destination_rel.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    manifest = {
        "schema_version": 1,
        "truth_role": "sanitized_public_package_manifest",
        "file_count": len(records),
        "files": sorted(records, key=lambda item: item["path"]),
    }
    manifest_path = output_dir / "sha256_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {**manifest, "manifest": str(manifest_path), "status": "pass"}


def verify(package_dir: Path, manifest_path: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise PackageError(f"package directory does not exist: {package_dir}")
    manifest = load_object(manifest_path.resolve())
    if manifest.get("schema_version") != 1:
        raise PackageError("manifest.schema_version must be 1")
    if manifest.get("truth_role") != "sanitized_public_package_manifest":
        raise PackageError("manifest truth_role is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PackageError("manifest.files must be a non-empty list")
    if manifest.get("file_count") != len(files):
        raise PackageError("manifest.file_count must match manifest.files")
    mismatches: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise PackageError(f"manifest.files[{index}] must be an object")
        relative = safe_relative(entry.get("path"), f"manifest.files[{index}].path")
        if relative.as_posix() in expected_paths:
            raise PackageError(f"duplicate manifest path: {relative.as_posix()}")
        expected_paths.add(relative.as_posix())
        if not isinstance(entry.get("sha256"), str) or not SHA256.match(entry["sha256"]):
            raise PackageError(f"manifest.files[{index}].sha256 is invalid")
        path = (package_dir / relative).resolve()
        try:
            path.relative_to(package_dir)
        except ValueError as exc:
            raise PackageError(f"manifest path escapes package: {relative}") from exc
        observed = sha256(path) if path.is_file() else None
        if observed != entry.get("sha256"):
            mismatches.append(
                {"path": relative.as_posix(), "expected": entry.get("sha256"), "observed": observed}
            )
        if path.is_file():
            if path.suffix.lower() in BLOCKED_SUFFIXES:
                mismatches.append({"path": relative.as_posix(), "reason": "blocked suffix"})
            findings = scan_text(path)
            if findings:
                mismatches.append(
                    {"path": relative.as_posix(), "reason": "sensitive content", "findings": findings}
                )
    try:
        manifest_relative = manifest_path.resolve().relative_to(package_dir).as_posix()
    except ValueError as exc:
        raise PackageError("manifest must be inside package_dir") from exc
    actual_paths = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_paths - expected_paths - {manifest_relative})
    for path in unexpected:
        mismatches.append({"path": path, "reason": "unexpected file not in manifest"})
    return {
        "schema_version": 1,
        "truth_role": "public_package_hash_verification",
        "status": "pass" if not mismatches else "fail",
        "file_count": len(files),
        "mismatches": mismatches,
        "unexpected_files": unexpected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-root", type=Path, required=True)
    build_parser.add_argument("--allowlist", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--package-dir", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "build":
            result = build(
                args.source_root,
                args.allowlist,
                args.output_dir,
                max_file_bytes=args.max_file_bytes,
            )
        else:
            result = verify(args.package_dir, args.manifest)
    except (OSError, PackageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
