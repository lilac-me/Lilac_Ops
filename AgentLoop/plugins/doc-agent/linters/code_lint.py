"""Code lint (R-COR-002): static checks on fenced code blocks."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r'^```(\w*)\s*$')


def _blocks(text: str) -> list[tuple[str, int]]:
    blocks = []
    in_code = False
    lang = ''
    start = 0
    buf: list[str] = []
    for num, line in enumerate(text.split('\n'), 1):
        m = FENCE_RE.match(line.strip())
        if m:
            if in_code:
                blocks.append(('\n'.join(buf), start, lang))
                buf, in_code = [], False
            else:
                in_code, lang, start = True, m.group(1), num
            continue
        if in_code:
            buf.append(line)
    return blocks


def _py_check(code: str, repo: Path, issues: list[dict], file: str, start: int) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        issues.append({
            'id': f'lint-code-{len(issues) + 1}',
            'rule_id': 'R-COR-002',
            'dimension': 'correctness',
            'severity': 'major',
            'location': {'file': file, 'line': start + (exc.lineno or 0)},
            'evidence': f'Python 语法错误: {exc.msg}',
            'suggestion': '修正代码块语法',
            'source': 'lint:code',
            'status': 'open',
        })
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split('.')[0]
            if not any(repo.glob(f'{top}.py')) and not (repo / top).is_dir():
                issues.append({
                    'id': f'lint-code-{len(issues) + 1}',
                    'rule_id': 'R-COR-002',
                    'dimension': 'correctness',
                    'severity': 'major',
                    'location': {'file': file, 'line': start},
                    'evidence': f'import {top} 在目标仓未找到（静态启发，仅提示）',
                    'suggestion': '确认依赖名称或在文档中声明外部依赖',
                    'source': 'lint:code',
                    'status': 'open',
                })


def _shell_check(code: str, issues: list[dict], file: str, start: int) -> None:
    if code.count('"') % 2 or code.count("'") % 2:
        issues.append({
            'id': f'lint-code-{len(issues) + 1}',
            'rule_id': 'R-COR-002',
            'dimension': 'correctness',
            'severity': 'major',
            'location': {'file': file, 'line': start},
            'evidence': 'shell 代码块引号不配对（基础检查）',
            'suggestion': '检查命令引号',
            'source': 'lint:code',
            'status': 'open',
        })


def check(doc_path: Path, repo: Path) -> list[dict]:
    issues: list[dict] = []
    text = doc_path.read_text(encoding='utf-8', errors='ignore')
    for code, start, lang in _blocks(text):
        lang = (lang or '').lower()
        if lang in ('python', 'py'):
            _py_check(code, repo, issues, str(doc_path), start)
        elif lang in ('bash', 'sh', 'shell', 'cmd'):
            _shell_check(code, issues, str(doc_path), start)
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('code_lint: use doc_lint.py lint')
    sys.exit(2)
