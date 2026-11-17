"""Placeholder lint (R-COM-006): TODO/FIXME/unfilled angle placeholders."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TODO_RE = re.compile(r'\b(TODO|FIXME|XXX)\b', re.IGNORECASE)
ANGLE_RE = re.compile(r'<[^>\s]{1,60}>')
FENCE_START = re.compile(r'^```')
# Common inline HTML tags that are content, not placeholders
HTML_TAGS = {'br', 'hr', 'img', 'div', 'span', 'p', 'a', 'b', 'i', 'u', 'em',
             'strong', 'code', 'pre', 'table', 'tr', 'td', 'th', 'center',
             'font', 'li', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
             'video', 'sup', 'sub', 'small', 'big', 'tt', 'strike', 's',
             'blockquote', 'caption', 'thead', 'tbody', 'col', 'colgroup',
             'section', 'article', 'header', 'footer', 'nav', 'main', 'aside',
             'iframe', 'input', 'button', 'form', 'label', 'select', 'option',
             'textarea', 'style', 'script', 'title', 'meta', 'link', 'head',
             'body', 'html', 'audio', 'source', 'track', 'details', 'summary',
             'figure', 'figcaption', 'mark', 'ruby', 'rt', 'rp', 'wbr', 'bdi',
             'bdo', 'data', 'time', 'kbd', 'samp', 'var', 'abbr', 'cite', 'q',
             'dfn', 'ins', 'del', 'address', 'dl', 'dt', 'dd', 'fieldset',
             'legend', 'meter', 'progress', 'output', 'picture', 'canvas',
             'noscript', 'template', 'slot', 'dialog', 'menu', 'menuitem'}


def _is_html_tag(token: str) -> bool:
    inner = token[1:-1].strip()
    if inner.startswith('/'):
        inner = inner[1:]
    name = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)(?:\s.*)?$', inner)
    return bool(name) and name.group(1).lower() in HTML_TAGS


def check(doc_path: Path) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    in_code = False
    for num, line in enumerate(text.split('\n'), 1):
        if FENCE_START.match(line.strip()):
            in_code = not in_code
            continue
        if in_code:
            continue
        if TODO_RE.search(line):
            issues.append({
                'id': f'lint-ph-{len(issues) + 1}',
                'rule_id': 'R-COM-006',
                'dimension': 'completeness',
                'severity': 'critical',
                'location': {'file': str(doc_path), 'line': num},
                'evidence': f'占位符: {TODO_RE.search(line).group(1)}',
                'suggestion': '填充内容或删除占位',
                'source': 'lint:placeholder',
                'status': 'open',
            })
        for m in ANGLE_RE.finditer(line):
            token = m.group(0)
            if token.startswith('<http'):
                continue
            if _is_html_tag(token):
                continue
            issues.append({
                'id': f'lint-ph-{len(issues) + 1}',
                'rule_id': 'R-COM-006',
                'dimension': 'completeness',
                'severity': 'critical',
                'location': {'file': str(doc_path), 'line': num},
                'evidence': f'未填充占位符: {token}',
                'suggestion': '给出实际值或明确说明',
                'source': 'lint:placeholder',
                'status': 'open',
            })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('placeholder_lint: use doc_lint.py lint')
    sys.exit(2)
