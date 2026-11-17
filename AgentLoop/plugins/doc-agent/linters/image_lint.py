"""Image lint (R-ACC-007 / R-COM-008): relative image paths, alt text and
figure_requests consistency. R-COR-006 (真实性) is a semantic evidence-chain
check by reviewers, not a machine check."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)\)')
PENDING_RE = re.compile(r'待人工插图')


def check(doc_path: Path, repo: Path, figures_path: Path | None = None) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    figure_items = []
    if figures_path and figures_path.is_file():
        try:
            figure_items = json.loads(figures_path.read_text(encoding='utf-8')).get('items', [])
        except (OSError, json.JSONDecodeError):
            pass
    figure_nos = {str(i.get('no')) for i in figure_items}
    for num, line in enumerate(text.split('\n'), 1):
        for m in IMG_RE.finditer(line):
            alt, path = m.group(1), m.group(2)
            if path.startswith(('http://', 'https://', '#')):
                continue
            target = (repo / path) if not path.startswith('/') else Path(path)
            if not target.exists():
                issues.append({
                    'id': f'lint-img-{len(issues) + 1}',
                    'rule_id': 'R-ACC-007',
                    'dimension': 'accessibility',
                    'severity': 'major',
                    'location': {'file': str(doc_path), 'line': num},
                    'evidence': f'图片路径不存在：{path}',
                    'suggestion': '修正路径或补充图片交付',
                    'source': 'lint:image',
                    'status': 'open',
                })
            if alt.strip() == '':
                issues.append({
                    'id': f'lint-img-{len(issues) + 1}',
                    'rule_id': 'R-COM-008',
                    'dimension': 'completeness',
                    'severity': 'major',
                    'location': {'file': str(doc_path), 'line': num},
                    'evidence': '图片缺少替代文本',
                    'suggestion': '补充替代文本草案',
                    'source': 'lint:image',
                    'status': 'open',
                })
            if PENDING_RE.search(alt) and figure_nos:
                no_match = re.search(r'图\s*([\w\-]+)[：:]', alt)
                if no_match and no_match.group(1) not in figure_nos:
                    issues.append({
                        'id': f'lint-img-{len(issues) + 1}',
                        'rule_id': 'R-COM-008',
                        'dimension': 'completeness',
                        'severity': 'major',
                        'location': {'file': str(doc_path), 'line': num},
                        'evidence': f"占位图号 {no_match.group(1)} 未登记于 figure_requests.json",
                        'suggestion': '在 figure_requests.json 登记该插图诉求',
                        'source': 'lint:image',
                        'status': 'open',
                    })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('image_lint: use doc_lint.py lint')
    sys.exit(2)
