"""Version lint (R-COR-004 / R-COM-007): best-effort version annotation and
配套表 consistency checks. Unsupported deep checks degrade gracefully.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r'(?:版本|version|v\d+\.\d+|\d+\.\d+\.\d+)', re.IGNORECASE)


def check(doc_path: Path, repo: Path) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    # R-COM-007: document should annotate an applicable version range
    if not VERSION_RE.search(text):
        issues.append({
            'id': f'lint-ver-{len(issues) + 1}',
            'rule_id': 'R-COM-007',
            'dimension': 'completeness',
            'severity': 'minor',
            'location': {'file': str(doc_path)},
            'evidence': '文档未标注适用版本',
            'suggestion': '标注适用版本或版本范围',
            'source': 'lint:version',
            'status': 'open',
        })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('version_lint: use doc_lint.py lint')
    sys.exit(2)
