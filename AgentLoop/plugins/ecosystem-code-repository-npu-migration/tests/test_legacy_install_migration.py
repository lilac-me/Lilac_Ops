#!/usr/bin/env python3
"""Verify legacy install migration is read-only unless --apply is explicit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migrate_legacy_install.py"


def invoke(home: Path, *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(home), "--timestamp", "test", "--json", *extra],
        text=True, capture_output=True, check=False,
    )
    if proc.returncode:
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="npu-legacy-install-") as value:
        home = Path(value)
        old = home / ".codex" / "skills" / "ecosystem-code-repository-npu-migration-custom-workflow"
        old_alias = home / ".codex" / "skills" / "ecosystem-code-repository-npu-migration-test-repair"
        experience = old / "references" / "experience" / "positive"
        experience.mkdir(parents=True)
        (old / "SKILL.md").write_text("legacy", encoding="utf-8")
        (experience / "local.json").write_text("{}", encoding="utf-8")
        old_alias.mkdir(parents=True)
        (old_alias / "SKILL.md").write_text("legacy alias", encoding="utf-8")
        dry = invoke(home)
        assert dry["legacy_install_count"] == 2 and old.exists() and old_alias.exists()
        applied = invoke(home, "--apply")
        assert applied["global_install_modified"] is True
        assert not old.exists() and not old_alias.exists()
        backup = home / ".codex" / "ecosystem-code-repository-npu-migration-legacy-backup-test"
        assert (backup / "ecosystem-code-repository-npu-migration-custom-workflow" / "SKILL.md").is_file()
        assert (backup / "ecosystem-code-repository-npu-migration-test-repair" / "SKILL.md").is_file()
        assert (backup / "experience-quarantine" / "ecosystem-code-repository-npu-migration-custom-workflow" / "positive" / "local.json").is_file()
    print("SELF_TEST_PASS")
    return 0


def test_legacy_install_cleanup() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
