#!/usr/bin/env python3
"""Inventory and explicitly quarantine legacy loose ecosystem-code-repository-npu-migration skill installs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path


LEGACY_NAMES = (
    "ecosystem-code-repository-npu-migration-goal", "ecosystem-code-repository-npu-migration-custom-workflow", "ecosystem-code-repository-npu-migration-env-knowledge",
    "ecosystem-code-repository-npu-migration-env-setup", "ecosystem-code-repository-npu-migration-local-assets", "ecosystem-code-repository-npu-migration-vllm-ascend-env",
    "ecosystem-code-repository-npu-migration-feature-model", "ecosystem-code-repository-npu-migration-repair", "ecosystem-code-repository-npu-migration-test-repair",
    "ecosystem-code-repository-npu-migration-report-pr", "ecosystem-code-repository-npu-migration-docs-support",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    home = args.home.expanduser().resolve()
    stamp = args.timestamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    roots = {"codex": home / ".codex" / "skills"}
    found: list[dict[str, str]] = []
    moved: list[dict[str, str]] = []
    for host, root in roots.items():
        backup = root.parent / f"ecosystem-code-repository-npu-migration-legacy-backup-{stamp}"
        for name in LEGACY_NAMES:
            source = root / name
            if not source.is_dir():
                continue
            destination = backup / name
            experience = source / "references" / "experience"
            record = {"host": host, "source": str(source), "backup": str(destination)}
            if experience.is_dir():
                record["experience_quarantine"] = str(backup / "experience-quarantine" / name)
            found.append(record)
            if args.apply:
                backup.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise RuntimeError(f"backup destination already exists: {destination}")
                if experience.is_dir():
                    quarantine = backup / "experience-quarantine" / name
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(experience, quarantine)
                shutil.move(str(source), str(destination))
                moved.append(record)
    result = {
        "status": "PASS",
        "reason_code": "LEGACY_INSTALL_INVENTORIED" if not args.apply else "LEGACY_INSTALL_QUARANTINED",
        "apply": args.apply,
        "home": str(home),
        "legacy_install_count": len(found),
        "found": found,
        "moved": moved,
        "global_install_modified": bool(args.apply and moved),
    }
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
