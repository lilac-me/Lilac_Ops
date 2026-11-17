"""Term lint (R-CLA-008): typo / simplified-traditional mix / term case
heuristics. Best-effort with a small built-in dictionary; only minor hints.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 简体 -> 繁体 小映射表（仅常见字，避免误报）
SC_TC_PAIRS = [('为', '爲'), ('后', '後'), ('发', '發'), ('达', '達'), ('类', '類'),
               ('时', '時'), ('说', '說'), ('学', '學'), ('动', '動'), ('当', '當'),
               ('点', '點'), ('门', '門'), ('开', '開'), ('关', '關'), ('设', '設'),
               ('计', '計'), ('参', '參'), ('数', '數'), ('据', '據'), ('库', '庫')]
# 常见错别字（低误报的几组）
TYPO_PAIRS = [('做为', '作为'), ('帐号', '账号'), ('登陆', '登录'), ('安装包安装', '安装包安装'),
              ('默认值缺省', '默认值')]
TERM_CASE = [('npu', 'NPU'), ('ascend', '昇腾'), ('mindSpeed', 'MindSpeed'),
             ('cann', 'CANN'), ('vllm', 'vLLM')]


def check(doc_path: Path) -> list[dict]:
    issues = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    for sc, tc in SC_TC_PAIRS:
        if sc in text and tc in text:
            issues.append({
                'id': f'lint-term-{len(issues) + 1}',
                'rule_id': 'R-CLA-008',
                'dimension': 'clarity',
                'severity': 'minor',
                'location': {'file': str(doc_path), 'line': text.index(tc) and
                             text[:text.index(tc)].count('\n') + 1},
                'evidence': f'简繁混用：{sc} 与 {tc} 并存',
                'suggestion': '全文统一为简体',
                'source': 'lint:term',
                'status': 'open',
            })
            break  # 一次命中即提示，避免刷屏
    for line_num, line in enumerate(text.split('\n'), 1):
        for bad, good in TERM_CASE:
            for m in re.finditer(rf'(?<![A-Za-z]){re.escape(bad)}(?![A-Za-z])', line):
                issues.append({
                    'id': f'lint-term-{len(issues) + 1}',
                    'rule_id': 'R-CLA-008',
                    'dimension': 'clarity',
                    'severity': 'minor',
                    'location': {'file': str(doc_path), 'line': line_num},
                    'evidence': f'术语大小写：{m.group(0)} 应为 {good}',
                    'suggestion': f'统一为 {good}',
                    'source': 'lint:term',
                    'status': 'open',
                })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('term_lint: use doc_lint.py lint')
    sys.exit(2)
