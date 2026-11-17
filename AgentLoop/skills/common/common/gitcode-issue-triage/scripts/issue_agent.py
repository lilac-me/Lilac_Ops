"""Locate and run the mindspeed-scan-issue-agent CLI from a checkout or installed plugin."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_entrypoint() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        source_entry = parent / "plugins" / "mindspeed-scan-issue-agent" / "scripts" / "issue_agent.py"
        if source_entry.exists():
            return source_entry
        installed_entry = parent / "scripts" / "issue_agent.py"
        if (parent / ".codex-plugin" / "plugin.json").exists() and installed_entry.exists():
            return installed_entry
    raise FileNotFoundError("mindspeed-scan-issue-agent plugin entrypoint was not found")


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(find_entrypoint()), *sys.argv[1:]]))
