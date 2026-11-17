#!/usr/bin/env python3
"""Build or check the deterministic plugin bundle lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from plugin_bundle import build_bundle_payload


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[1]
CANONICAL_SKILLS_ROOT = REPO_ROOT / "skills" / "ecosystem-code-repository-npu-migration"


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PLUGIN_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def comparable(value: dict) -> dict:
    result = dict(value)
    result.pop("source", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    # Skill sources live under the repo-level canonical domain directory, so
    # release cleanliness has to cover both the plugin tree and that root.
    dirty = bool(
        git_output("status", "--porcelain", "--", str(PLUGIN_ROOT), str(CANONICAL_SKILLS_ROOT))
    )
    if args.release and dirty:
        print("release bundle requires a clean plugin worktree", file=sys.stderr)
        return 2
    payload = build_bundle_payload(
        PLUGIN_ROOT,
        source_commit=git_output("rev-parse", "HEAD"),
        source_dirty=dirty,
    )
    lock_path = PLUGIN_ROOT / "bundle.lock.json"
    if args.write:
        lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(lock_path)
        return 0
    try:
        current = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = None
    if not isinstance(current, dict) or comparable(current) != comparable(payload):
        print("bundle.lock.json is stale; run build_bundle_lock.py --write", file=sys.stderr)
        return 2
    print(lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
