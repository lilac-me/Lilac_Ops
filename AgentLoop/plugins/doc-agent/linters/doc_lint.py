"""Unified doc lint CLI for doc-agent.

Commands:
  lint    --repo R --doc D [--manifest M] [--capabilities models,toolchain]
          [--include ID[,ID] [--exclude ID[,ID]]] [--rules RULES]
          [--figures F] [--no-network] --out OUT.json
  rules   [--rules RULES]        validate rules.yaml
  link    <repo|url> [...]       compat link scanner (see link_lint.py)

All deterministic rules in rules.yaml are executed through their declared
linter module; every issue binds a rule_id from rules.yaml. A missing or
failed declared linter marks the report failed and prevents scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PLUGIN_ROOT / 'engine'
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ENGINE_DIR))

import yamlite  # noqa: E402
import gates  # noqa: E402

DEFAULT_MANIFEST = PLUGIN_ROOT / 'templates' / 'repo_manifest.yaml'
DEFAULT_RULES = PLUGIN_ROOT / 'rules' / 'rules.yaml'

LINTER_MODULES = {
    'link': 'link_lint',
    'code': 'code_lint',
    'manifest': 'manifest_lint',
    'placeholder': 'placeholder_lint',
    'term': 'term_lint',
    'image': 'image_lint',
    'nav': 'nav_lint',
    'style': 'style_lint',
    'xref': 'xref_lint',
    'version': 'version_lint',
    'placement': 'placement_lint',
}


def _load_rules(rules_path: Path) -> dict:
    data = yamlite.load(rules_path.read_text(encoding='utf-8'))
    return {r['id']: r for r in data.get('rules', [])}


def _run_linter(name: str, issues: list[dict], args, doc_path: Path, repo: Path) -> str | None:
    """Run one configured linter and return an error instead of failing open."""
    import importlib
    mod_name = LINTER_MODULES.get(name)
    if mod_name is None:
        return f'no linter module for kind {name!r}'
    try:
        importlib.import_module(mod_name)
    except ImportError as exc:
        return f'linter {name} unavailable ({exc})'
    try:
        if name == 'link':
            from link_lint import emit_issues, scan
            results = scan(str(repo), check_links=not args.no_network)
            issues.extend(emit_issues(results))
        elif name == 'manifest':
            from manifest_lint import check, load_manifest
            manifest_path = Path(args.manifest or DEFAULT_MANIFEST)
            if not manifest_path.is_file():
                return f'manifest not found: {manifest_path}'
            manifest = load_manifest(manifest_path)
            capabilities = set()
            for cap in (args.capabilities or '').split(','):
                cap = cap.strip()
                if cap:
                    capabilities.add(f'repo_has_{cap}')
            include = [x.strip() for x in (args.include or '').split(',') if x.strip()]
            exclude = [x.strip() for x in (args.exclude or '').split(',') if x.strip()]
            issues.extend(check(repo, manifest, capabilities, include, exclude))
        elif name == 'placement':
            from placement_lint import check
            from manifest_lint import load_manifest
            manifest_path = Path(args.manifest or DEFAULT_MANIFEST)
            if not manifest_path.is_file():
                return f'manifest not found: {manifest_path}'
            manifest = load_manifest(manifest_path)
            issues.extend(check(repo, manifest, args.self_developed))
        elif name == 'code':
            from code_lint import check
            issues.extend(check(doc_path, repo))
        elif name == 'placeholder':
            from placeholder_lint import check
            issues.extend(check(doc_path))
        elif name == 'term':
            from term_lint import check
            issues.extend(check(doc_path))
        elif name == 'image':
            from image_lint import check
            figures = Path(args.figures) if args.figures else None
            issues.extend(check(doc_path, repo, figures))
        elif name == 'nav':
            from nav_lint import check
            issues.extend(check(doc_path, repo))
        elif name == 'style':
            from style_lint import check
            issues.extend(check(doc_path))
        elif name == 'xref':
            from xref_lint import check
            issues.extend(check(doc_path, repo))
        elif name == 'version':
            from version_lint import check
            issues.extend(check(doc_path, repo))
    except Exception as exc:  # noqa: BLE001 - report failure without a traceback
        return f'linter {name} failed ({exc})'
    return None


def _apply_rule_metadata(issues: list[dict], rules: dict) -> None:
    for issue in issues:
        rule = rules.get(issue.get('rule_id', ''), {})
        if rule:
            # linters may refine severity per hit (e.g. soft-scope hints);
            # only fill missing metadata from the rule.
            issue.setdefault('severity', rule.get('severity'))
            issue.setdefault('dimension', rule.get('dimension'))


def _write_lint_report(args, doc_path: Path, digest: str,
                       issues: list[dict], errors: list[str]) -> int:
    """Write one contract-shaped report and return the documented exit code."""
    out = {
        'doc': str(doc_path),
        'doc_sha256': digest,
        'depth': args.depth,
        'status': 'failed' if errors else 'completed',
        'issues': issues,
        'errors': errors,
    }
    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'ERROR: cannot write lint report: {exc}', file=sys.stderr)
        return 2
    sev = {}
    for issue in issues:
        sev[issue['severity']] = sev.get(issue['severity'], 0) + 1
    print(f'LINT: {out["status"]}; {len(issues)} issue(s) {sev or "none"} -> {out_path}')
    if errors:
        return 2
    return 0 if not issues else 1


def _cmd_lint(args) -> int:
    repo = Path(args.repo).resolve()
    doc_path = Path(args.doc).resolve()
    if not repo.is_dir():
        print(f'ERROR: repo not found: {args.repo}', file=sys.stderr)
        return 2
    if not doc_path.is_file():
        print(f'ERROR: doc not found: {args.doc}', file=sys.stderr)
        return 2
    try:
        initial_digest = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    except OSError as exc:
        print(f'ERROR: cannot read target document: {exc}', file=sys.stderr)
        return 2
    try:
        doc_rel = doc_path.relative_to(repo)
        if len(doc_rel.parts) >= 2 and doc_rel.parts[:2] == ('docs', 'en'):
            error = ('target doc is under docs/en and outside doc-agent scope; '
                     'it must be reviewed by the translation team')
            print(f'ERROR: {error}', file=sys.stderr)
            return _write_lint_report(
                args, doc_path, initial_digest, [], [error]
            )
    except ValueError:
        pass
    rules_path = Path(args.rules or DEFAULT_RULES)
    if not rules_path.is_file():
        print(f'ERROR: rules not found: {rules_path}', file=sys.stderr)
        return 2
    try:
        rules = _load_rules(rules_path)
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: cannot load rules: {exc}', file=sys.stderr)
        return 2
    kinds = sorted({r.get('linter') for r in rules.values()
                    if r.get('kind') == 'deterministic' and r.get('linter')})
    issues: list[dict] = []
    errors: list[str] = []
    for kind in kinds:
        error = _run_linter(kind, issues, args, doc_path, repo)
        if error:
            errors.append(error)
            print(f'ERROR: {error}', file=sys.stderr)
    _apply_rule_metadata(issues, rules)
    for index, issue in enumerate(issues):
        ok, detail = gates.validate_value(issue, 'issue', f'issues[{index}]')
        if not ok:
            errors.append(detail)
        rule = rules.get(issue.get('rule_id'))
        if rule is None:
            errors.append(f'issues[{index}]: unknown rule_id {issue.get("rule_id")!r}')
        elif issue.get('dimension') != rule.get('dimension'):
            errors.append(
                f'issues[{index}]: dimension {issue.get("dimension")!r} does not match '
                f'rule {issue.get("rule_id")} ({rule.get("dimension")!r})'
            )
        if issue.get('status') != 'open':
            errors.append(f'issues[{index}]: deterministic lint status must be open')
        if not str(issue.get('source', '')).startswith('lint:'):
            errors.append(f'issues[{index}]: deterministic source must start with lint:')
    # sort by severity then file/line
    order = {'critical': 0, 'major': 1, 'minor': 2}
    issues.sort(key=lambda i: (order.get(i.get('severity'), 9),
                               i.get('location', {}).get('file', ''),
                               i.get('location', {}).get('line', 0)))
    try:
        final_digest = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    except OSError as exc:
        final_digest = initial_digest
        errors.append(f'target document became unreadable during lint: {exc}')
    if final_digest != initial_digest:
        errors.append('target document changed during lint; rerun lint')
    return _write_lint_report(
        args, doc_path, final_digest, issues, errors
    )


def _cmd_rules(args) -> int:
    rules_path = Path(args.rules or DEFAULT_RULES)
    if not rules_path.is_file():
        print(f'ERROR: rules not found: {rules_path}', file=sys.stderr)
        return 2
    data = yamlite.load(rules_path.read_text(encoding='utf-8'))
    rules = data.get('rules', [])
    ids = [r.get('id') for r in rules]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append('duplicate rule ids')
    for r in rules:
        if not r.get('id') or not re.match(r'^R-(COR|COM|CLA|ACC)-\d{3}$', r.get('id', '')):
            problems.append(f"malformed id: {r.get('id')!r}")
        for field in ('dimension', 'title', 'kind', 'severity', 'weight', 'primary'):
            if field not in r:
                problems.append(f"{r.get('id')}: missing field '{field}'")
        if r.get('kind') == 'deterministic' and not r.get('linter'):
            problems.append(f"{r.get('id')}: deterministic rule missing linter")
        if r.get('kind') == 'semantic' and 'linter' in r:
            problems.append(f"{r.get('id')}: semantic rule should not declare linter")
        if r.get('dimension') not in ('correctness', 'completeness', 'clarity', 'accessibility'):
            problems.append(f"{r.get('id')}: bad dimension {r.get('dimension')!r}")
    scoring = data.get('scoring', {})
    for key in ('weights', 'floors', 'pass_total', 'max_rounds'):
        if key not in scoring:
            problems.append(f'scoring missing {key}')
    if problems:
        print('RULES: problems found')
        for p in problems:
            print(f'  [FAIL] {p}')
        return 1
    print(f'RULES: ok ({len(ids)} rules)')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='doc_lint', description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_lint = sub.add_parser('lint')
    p_lint.add_argument('--repo', required=True)
    p_lint.add_argument('--doc', required=True)
    p_lint.add_argument('--manifest', default=None)
    p_lint.add_argument('--capabilities', default=None,
                        help='comma list: models, toolchain')
    p_lint.add_argument('--include', default=None,
                        help='comma list of optional manifest ids to require')
    p_lint.add_argument('--exclude', default=None,
                        help='comma list of manifest ids to waive')
    p_lint.add_argument('--self-developed', action='store_true',
                        help='target repo is a self-developed Ascend repo; '
                             'enforces docs/zh placement (R-ACC-008)')
    p_lint.add_argument('--rules', default=None)
    p_lint.add_argument('--figures', default=None)
    p_lint.add_argument('--depth', choices=['L0', 'L1', 'L2'], default='L0',
                        help='recorded only; deterministic linters run in every depth')
    p_lint.add_argument('--no-network', action='store_true')
    p_lint.add_argument('--out', required=True)

    p_rules = sub.add_parser('rules')
    p_rules.add_argument('--rules', default=None)

    p_link = sub.add_parser('link')
    p_link.add_argument('repo')
    p_link.add_argument('-o', '--output', default=None)
    p_link.add_argument('-q', '--quiet', action='store_true')
    p_link.add_argument('--no-check-links', action='store_true')
    p_link.add_argument('--no-cleanup', action='store_true')
    p_link.add_argument('--timeout', type=int, default=10)
    p_link.add_argument('--concurrency', type=int, default=8)

    args = parser.parse_args(argv)
    if args.cmd == 'lint':
        return _cmd_lint(args)
    if args.cmd == 'rules':
        return _cmd_rules(args)
    if args.cmd == 'link':
        from link_lint import main as link_main
        return link_main(argv[1:] if argv else [])
    return 2


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    raise SystemExit(main())
