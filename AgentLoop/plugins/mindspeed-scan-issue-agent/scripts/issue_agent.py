from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from issue_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
