"""Nav lint (R-ACC-001/003/006): README reachability via link-graph BFS."""

from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path

MD_REL_LINK = re.compile(r'\[([^\]]*)\]\(([^)#]+)(?:#[^)]*)?\)')
H2_RE = re.compile(r'^##\s+(.+?)\s*$', re.M)


def _en_out_of_scope(repo: Path, resolved: Path) -> bool:
    try:
        rel = resolved.relative_to(repo)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[:2] == ('docs', 'en')


def _doc_links(path: Path, repo: Path) -> list[str]:
    text = path.read_text(encoding='utf-8', errors='ignore')
    targets = []
    for m in MD_REL_LINK.finditer(text):
        target = m.group(2).strip()
        if target.startswith(('http://', 'https://')):
            continue
        resolved = (path.parent / target).resolve()
        if _en_out_of_scope(repo, resolved):
            continue  # en tree is out of doc-agent scope
        if resolved.suffix.lower() in ('.md', '.rst') or resolved.is_dir():
            targets.append(str(resolved.relative_to(repo)))
    return targets


def check(doc_path: Path, repo: Path) -> list[dict]:
    issues = []
    readme = repo / 'README.md'
    if not readme.is_file():
        return issues
    rel_doc = str(doc_path.resolve().relative_to(repo.resolve()))
    queue = deque(['README.md'])
    visited = {'README.md'}
    steps = {'README.md': 0}
    while queue:
        cur = queue.popleft()
        cur_path = (repo / cur).resolve()
        if not cur_path.is_file():
            continue
        for target in _doc_links(cur_path, repo):
            if target in visited:
                continue
            visited.add(target)
            steps[target] = steps[cur] + 1
            queue.append(target)
    if rel_doc not in visited:
        issues.append({
            'id': f'lint-nav-{len(issues) + 1}',
            'rule_id': 'R-ACC-001',
            'dimension': 'accessibility',
            'severity': 'critical',
            'location': {'file': rel_doc},
            'evidence': '从 README 链接图不可达目标文档',
            'suggestion': '在 README 对应入口章节登记该文档链接',
            'source': 'lint:nav',
            'status': 'open',
        })
    elif steps[rel_doc] > 5:
        issues.append({
            'id': f'lint-nav-{len(issues) + 1}',
            'rule_id': 'R-ACC-003',
            'dimension': 'accessibility',
            'severity': 'major',
            'location': {'file': rel_doc},
            'evidence': f'README 到目标文档需 {steps[rel_doc]} 步（>5）',
            'suggestion': '增加更直接的入口链接',
            'source': 'lint:nav',
            'status': 'open',
        })
    # R-ACC-006: README entry links pointing to missing files
    readme_text = readme.read_text(encoding='utf-8', errors='ignore')
    for m in MD_REL_LINK.finditer(readme_text):
        target = m.group(2).strip()
        if target.startswith(('http://', 'https://', '#')):
            continue
        resolved = (readme.parent / target).resolve()
        if _en_out_of_scope(repo, resolved):
            continue  # en targets are the translation team's concern
        if not resolved.exists() and target.endswith('.md'):
            issues.append({
                'id': f'lint-nav-{len(issues) + 1}',
                'rule_id': 'R-ACC-006',
                'dimension': 'accessibility',
                'severity': 'major',
                'location': {'file': 'README.md',
                             'line': readme_text[:m.start()].count('\n') + 1},
                'evidence': f'README 入口链接目标不存在：{target}',
                'suggestion': '修正链接或补充目标文件',
                'source': 'lint:nav',
                'status': 'open',
            })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('nav_lint: use doc_lint.py lint')
    sys.exit(2)
