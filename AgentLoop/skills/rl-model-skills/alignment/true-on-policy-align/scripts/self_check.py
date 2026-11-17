#!/usr/bin/env python3
"""Run the dependency-free structural and regression checks for this Skill."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TEMPLATES = ROOT / "references" / "templates"


def run(command: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def check_python_sources() -> list[str]:
    errors: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")) + sorted((ROOT / "tests").glob("test_*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    return errors


def check_json_templates() -> list[str]:
    errors: list[str] = []
    for path in sorted(TEMPLATES.glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    return errors


def check_local_links() -> list[str]:
    errors: list[str] = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in sorted(ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            target = raw.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)} -> {raw}")
    return errors


def main() -> int:
    checks: list[dict[str, Any]] = []
    static_errors = {
        "python_sources": check_python_sources(),
        "json_templates": check_json_templates(),
        "local_links": check_local_links(),
    }
    checks.append({"name": "static", "errors": static_errors})
    with tempfile.TemporaryDirectory(prefix="true-on-policy-self-check-") as directory:
        commands = [
            [sys.executable, str(SCRIPTS / "validate_difference_packet.py"), "--packet", str(TEMPLATES / "difference_packet.json")],
            [sys.executable, str(SCRIPTS / "validate_difference_packet.py"), "--packet", str(TEMPLATES / "state_difference_packet.json")],
            [sys.executable, str(SCRIPTS / "validate_state_card.py"), "--state", str(TEMPLATES / "state_card.json")],
            [sys.executable, str(SCRIPTS / "generate_alignment_report.py"), "--input", str(TEMPLATES / "alignment_report_source.zh-CN.json"), "--output-dir", str(Path(directory) / "reports")],
            [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-p", "test_*.py"],
        ]
        for command in commands:
            checks.append(run(command))
    failed = any(static_errors.values()) or any(
        check.get("returncode", 0) != 0 for check in checks[1:]
    )
    result = {"valid": not failed, "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
