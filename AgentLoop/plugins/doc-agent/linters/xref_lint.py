"""Xref lint (R-COM-005): relative markdown link targets inside the repo."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MD_LINK = re.compile(r'\[([^\]]*)\]\(([^)#]+)(?:#[^)]*)?\)')


def check(doc_path: Path, repo: Path) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    for m in MD_LINK.finditer(text):
        target = m.group(2).strip()
        if target.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        resolved = (doc_path.parent / target).resolve()
        if not resolved.exists() and target.endswith('.md'):
            issues.append({
                'id': f'lint-xref-{len(issues) + 1}',
                'rule_id': 'R-COM-005',
                'dimension': 'completeness',
                'severity': 'major',
                'location': {'file': str(doc_path),
                             'line': text[:m.start()].count('\n') + 1},
                'evidence': f'交叉引用目标不存在：{target}',
                'suggestion': '修正引用路径',
                'source': 'lint:xref',
                'status': 'open',
            })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('xref_lint: use doc_lint.py lint')
    sys.exit(2)
