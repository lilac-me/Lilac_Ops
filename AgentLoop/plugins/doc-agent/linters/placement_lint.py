"""Placement lint (R-ACC-008): for self-developed repos, Chinese
documentation must live under docs/zh/ at the repo root.

The placement rules come from the `placement` section of
templates/repo_manifest.yaml, with three scopes per rule:
  out_of_scope  — skipped entirely (e.g. docs/en/** handled by the
                  translation team, outside doc-agent scope)
  soft_scope    — sporadic files acceptable; reported as minor hints
                  (e.g. examples/tests/docker)
  exceptions    — conventional repo-root files (README*/LICENSE etc.)
Everything else outside docs/zh/ is a major violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

HIDDEN = {'.git', '.github', '.gitee', '.gitcode', '.idea', '.vscode', '.hg', '.svn'}


def _rel(parts) -> str:
    return '/'.join(parts)


def _in_scope(parts: tuple, patterns: list) -> bool:
    """Match a path prefix against scope patterns like 'examples/**'."""
    for pattern in patterns:
        prefix = pattern.rstrip('/**').split('/')
        if tuple(parts[:len(prefix)]) == tuple(prefix):
            return True
    return False


def check(repo: Path, manifest: dict, enabled: bool) -> list[dict]:
    issues = []
    if not enabled:
        return issues
    placement_rules = [r for r in manifest.get('placement', [])
                       if r.get('applies_to') == 'self_developed']
    if not placement_rules:
        return issues
    out_of_scope = []
    soft_scope = []
    exceptions = []
    for rule in placement_rules:
        out_of_scope.extend(rule.get('out_of_scope') or [])
        soft_scope.extend(rule.get('soft_scope') or [])
        exceptions.extend(rule.get('exceptions') or [])
    for doc in sorted(list(repo.rglob('*.md')) + list(repo.rglob('*.rst'))):
        rel = doc.relative_to(repo)
        parts = rel.parts
        if any(part in HIDDEN for part in parts):
            continue
        if _in_scope(parts, out_of_scope):
            continue
        # docs/zh/ at any depth below it is the required location
        if len(parts) >= 2 and parts[0] == 'docs' and parts[1] == 'zh':
            continue
        if len(parts) == 1:  # repo-root file: exemptions apply
            if any(doc.match(pattern) for pattern in exceptions):
                continue
        if _in_scope(parts, soft_scope):
            severity = 'minor'
            evidence = f'零星资料位于 {_rel(parts)}（可接受；如属中文资料主体请移入 docs/zh/）'
        else:
            severity = 'major'
            evidence = f'中文资料未存放在 docs/zh/：{_rel(parts)}（自研仓强制要求）'
        issues.append({
            'id': f'lint-placement-{len(issues) + 1}',
            'rule_id': 'R-ACC-008',
            'dimension': 'accessibility',
            'severity': severity,
            'location': {'file': _rel(parts)},
            'evidence': evidence,
            'suggestion': '将中文资料移入仓库根目录下 docs/zh/ 内',
            'source': 'lint:placement',
            'status': 'open',
        })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('placement_lint: use doc_lint.py lint')
    sys.exit(2)
