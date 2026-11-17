#!/usr/bin/env python3
"""Synchronize the Claude/Codex plugin manifests, shared marketplace and bundle lock."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    if not SEMVER.fullmatch(args.version):
        raise SystemExit(f"invalid semver: {args.version}")
    if args.release:
        if git("status", "--porcelain").stdout.strip():
            raise SystemExit("release mode requires a clean worktree")
        tag = git("describe", "--tags", "--exact-match", "HEAD")
        if tag.returncode or tag.stdout.strip() not in {args.version, f"v{args.version}"}:
            raise SystemExit(f"release mode requires HEAD tag v{args.version} or {args.version}")

    manifest_path = PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
    claude_manifest_path = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
    claude_marketplace_path = REPO_ROOT / ".claude-plugin" / "marketplace.json"
    manifest = read(manifest_path)
    claude_manifest = read(claude_manifest_path)
    claude_marketplace = read(claude_marketplace_path)
    claude_matches = [
        item for item in claude_marketplace.get("plugins", []) if item.get("name") == "ecosystem-code-repository-npu-migration"
    ]
    if len(claude_matches) != 1:
        raise SystemExit("shared marketplace must contain ecosystem-code-repository-npu-migration exactly once")
    manifest["version"] = args.version
    claude_manifest["version"] = args.version
    claude_matches[0]["version"] = args.version
    write(manifest_path, manifest)
    write(claude_manifest_path, claude_manifest)
    write(claude_marketplace_path, claude_marketplace)
    return subprocess.run(
        [sys.executable, str(PLUGIN_ROOT / "scripts" / "build_bundle_lock.py"), "--write"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
