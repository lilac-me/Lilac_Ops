"""Manifest lint (R-COM-001): check a repo has all required documents
per the repository-level manifest (templates/repo_manifest.yaml).

Detection: a document is present if any `files` glob matches a non-empty
file, or any .md file has an H1/H2 heading matching one of the entry's
headings. Conditional entries apply only when their `when` capability is
enabled; entries with `name_when_models` keep their requirement but switch
their display title.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'engine'))

from yamlite import load as yaml_load  # noqa: E402

H_RE = re.compile(r'^#{1,2}\s+(.+?)\s*$', re.M)
CAPABILITIES = {'repo_has_models', 'repo_has_toolchain'}
# English doc tree is handled by a dedicated translation team and is out of
# doc-agent scope: never used to satisfy Chinese-document presence checks.
OUT_OF_SCOPE_PREFIX = ('docs', 'en')


def _out_of_scope(rel: Path) -> bool:
    return len(rel.parts) >= 2 and rel.parts[:2] == OUT_OF_SCOPE_PREFIX


def load_manifest(path: Path) -> dict:
    return yaml_load(path.read_text(encoding='utf-8'))


def _active(entries: list, capabilities: set) -> list:
    active = []
    for entry in entries:
        when = entry.get('when')
        if when and when not in capabilities:
            continue
        active.append(entry)
    return active


def _headings_of(entry: dict) -> list:
    value = entry.get('heading') or []
    if isinstance(value, str):
        return [value]
    return list(value)


def _display_name(entry: dict, capabilities: set) -> str:
    if 'repo_has_models' in capabilities and entry.get('name_when_models'):
        return entry['name_when_models']
    return entry.get('name', '')


def _detect(repo: Path, entry: dict) -> str | None:
    """Return detection evidence, or None when not found.

    `files` globs match recursively from the repo root (relative patterns
    behave like `**/<pattern>`), so nested doc trees such as docs/zh/...
    are covered. `heading` may be a string or a list; any match counts.
    """
    for pattern in entry.get('files') or []:
        # rglob matches at any depth (root files included), so nested doc
        # trees such as docs/zh/... are covered by every pattern; docs/en
        # matches are ignored (out of doc-agent scope).
        for p in repo.rglob(pattern):
            if _out_of_scope(p.relative_to(repo)):
                continue
            if p.is_file() and p.stat().st_size > 0:
                return f'file match: {p.relative_to(repo)} ({pattern})'
    for heading in _headings_of(entry):
        for md in repo.rglob('*.md'):
            if _out_of_scope(md.relative_to(repo)):
                continue
            try:
                text = md.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            for m in H_RE.finditer(text):
                title = m.group(1).strip()
                if title == heading or title.startswith(heading + '（'):
                    return f'heading match: {md.relative_to(repo)}#{title}'
    return None


def check(repo: Path, manifest: dict, capabilities: set,
          include: list, exclude: list) -> list[dict]:
    issues = []
    required_entries = _active(manifest.get('required', []), capabilities)
    wanted = {e['id']: e for e in required_entries}
    for opt in _active(manifest.get('optional', []), capabilities):
        if opt['id'] in include:
            wanted[opt['id']] = opt
    for eid in exclude:
        wanted.pop(eid, None)

    n = 0
    for eid, entry in wanted.items():
        evidence = _detect(repo, entry)
        if evidence is None:
            n += 1
            display = _display_name(entry, capabilities)
            issues.append({
                'id': f'lint-manifest-{n}',
                'rule_id': 'R-COM-001',
                'dimension': 'completeness',
                'severity': 'critical',
                # Multiple missing manifest entries share one rule and file;
                # the entry id keeps their lifecycle identities distinct.
                'location': {'file': str(repo), 'anchor': eid},
                'evidence': f"缺少必备资料：{display}（{eid}；files/heading 均未匹配）",
                'suggestion': f'补齐 {display} 资料（内容按目标仓场景灵活撰写）',
                'source': 'lint:manifest',
                'status': 'open',
            })
    return issues


if __name__ == '__main__':  # pragma: no cover
    print('manifest_lint: use doc_lint.py lint')
    sys.exit(2)
