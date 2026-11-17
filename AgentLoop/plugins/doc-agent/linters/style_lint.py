"""Style lint (R-CLA-003 / R-CLA-005): paragraph length and heading density."""

from __future__ import annotations

import re
import sys
from pathlib import Path

H_RE = re.compile(r'^#{1,4}\s')
FENCE = re.compile(r'^```')


def check(doc_path: Path) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    lines = text.split('\n')
    # paragraph length: consecutive non-blank, non-heading, non-fence lines
    para_start = None
    para_chars = 0
    in_code = False
    last_heading_line = 0
    char_count = 0

    def flush(issues: list[dict], start: int, length: int):
        # sentence heuristic: count 。！？!?. followed by nothing or space/end
        sentences = len(re.findall(r'[。！？!?](?:$|\s|[”"』])', '\n'.join(
            lines[start:start + length])))
        if sentences > 5:
            issues.append({
                'id': f'lint-style-{len(issues) + 1}',
                'rule_id': 'R-CLA-003',
                'dimension': 'clarity',
                'severity': 'minor',
                'location': {'file': str(doc_path), 'line': start + 1},
                'evidence': f'单段约 {sentences} 句（>5）',
                'suggestion': '拆分段落',
                'source': 'lint:style',
                'status': 'open',
            })

    for num, line in enumerate(lines, 1):
        if FENCE.match(line.strip()):
            in_code = not in_code
            if para_start is not None:
                flush(issues, para_start, num - para_start)
                para_start = None
            continue
        if in_code:
            continue
        if H_RE.match(line):
            if para_start is not None:
                flush(issues, para_start, num - para_start)
                para_start = None
            gap = char_count - last_heading_line
            last_heading_line = char_count
            continue
        if line.strip() == '':
            if para_start is not None:
                flush(issues, para_start, num - para_start)
                para_start = None
            continue
        if para_start is None:
            para_start = num
        char_count += len(line)
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('style_lint: use doc_lint.py lint')
    sys.exit(2)
