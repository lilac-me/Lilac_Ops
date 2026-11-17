"""Link lint (R-COR-001): scan a repo's docs for URLs, classify them and
verify reachability. Condensed port of the earlier scan_links.py core.

CLI (compat): link <repo|url> [-o report.xlsx] [-q] [--no-check-links]
[--no-cleanup] [--timeout N] [--concurrency N]

Library: scan(repo, check_links, timeout, concurrency) -> list[dict]
         emit_issues(results) -> list[issue]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MD_LINK_RE = re.compile(r'\[([^\]]*)\]\(\s*(https?://[^)\s]+?)(?:\s+"[^"]*")?\s*\)')
MD_REF_RE = re.compile(r'^\s*\[([^\]]*)\]:\s*(https?://\S+)', re.MULTILINE)
AUTO_LINK_RE = re.compile(r'<(https?://[^>]+)>')
HTML_HREF_RE = re.compile(r"""href=["'](https?://[^"']+)["']""", re.IGNORECASE)
RST_LINK_RE = re.compile(r'`[^`]*\s+<(https?://[^>]+)>`_')
RST_CITATION_RE = re.compile(r'\.\.\s+_([^:]+):\s*(https?://\S+)')
BARE_URL_RE = re.compile(r'https?://[^\s<>"\'\]}]+')

CODE_REPO_DOMAINS = ['gitcode.com', 'github.com', 'gitee.com', 'gitlab.com',
                     'bitbucket.org', 'codeberg.org', 'sourceforge.net',
                     'git.woa.com', 'gitea.com']
ASCEND_DOMAIN = 'hiascend.com'
SKIP_DIRS = {'.git', '.hg', '.svn', 'node_modules', '__pycache__',
             'venv', '.venv', 'env', '.idea', '.vscode'}

DEFAULT_TIMEOUT = 10
DEFAULT_CONCURRENCY = 8
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/120.0 Safari/537.36 doc-agent-link-lint')


def clean_url(url: str) -> str:
    while url and url[-1] in '.,;:!?。，；：！？':
        url = url[:-1]
    return url


def _is_inside_md_link(line: str, pos: int) -> bool:
    for m in MD_LINK_RE.finditer(line):
        if m.start(2) <= pos < m.end(2):
            return True
    return False


def extract_urls(text: str, source_file: str) -> list[dict]:
    results = []
    in_code = False
    fence = ''
    fence_len = 0
    for num, line in enumerate(text.split('\n'), 1):
        s = line.strip()
        if s.startswith('```'):
            n = 3
            while n < len(s) and s[n] == '`':
                n += 1
            if in_code and fence == '`' and n >= fence_len:
                in_code = False
                continue
            if not in_code:
                in_code, fence, fence_len = True, '`', n
                continue
        elif s.startswith('~~~'):
            n = 3
            while n < len(s) and s[n] == '~':
                n += 1
            if in_code and fence == '~' and n >= fence_len:
                in_code = False
                continue
            if not in_code:
                in_code, fence, fence_len = True, '~', n
                continue
        if in_code:
            continue
        urls = set()
        for m in MD_LINK_RE.finditer(line):
            url = m.group(2)
            url_end = m.end(2)
            if url_end < len(line) and line[url_end] == ')':
                if m.end() < len(line) and line[m.end()] == ')':
                    url += ')'
            urls.add(url)
        for m in RST_LINK_RE.finditer(line):
            urls.add(m.group(1))
        for m in RST_CITATION_RE.finditer(line):
            urls.add(clean_url(m.group(2)))
        for m in MD_REF_RE.finditer(line):
            urls.add(clean_url(m.group(2)))
        for m in AUTO_LINK_RE.finditer(line):
            urls.add(m.group(1))
        for m in HTML_HREF_RE.finditer(line):
            urls.add(m.group(1))
        for m in BARE_URL_RE.finditer(line):
            if _is_inside_md_link(line, m.start()):
                continue
            url = clean_url(m.group(0))
            while url.endswith(')') and url.count('(') < url.count(')'):
                url = url[:-1]
            if url and len(url) > 10:
                urls.add(url)
        for url in urls:
            results.append({'file': source_file, 'line': num, 'url': url})
    return results


def classify(url: str, repo_identity: dict | None) -> str:
    host = urlparse(url).netloc.lower()
    if ASCEND_DOMAIN in host:
        return '昇腾社区网址'
    if repo_identity and repo_identity.get('host') in host:
        return '本代码仓链接'
    if any(d in host for d in CODE_REPO_DOMAINS):
        return '其他代码仓链接'
    return '其他链接'


def check_url(url: str, timeout: int) -> tuple[str, str]:
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            try:
                req2 = urllib.request.Request(url, headers={'User-Agent': USER_AGENT,
                                                            'Range': 'bytes=0-1023'})
                with urllib.request.urlopen(req2, timeout=timeout) as resp:
                    return '有效', f'HTTP {resp.status}'
            except Exception as exc2:
                return '检查失败', str(exc2)
        return '失效', f'HTTP {exc.code}'
    except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
        return '检查失败', str(exc)
    if 200 <= code < 400:
        return '有效', f'HTTP {code}'
    return '失效', f'HTTP {code}'


def _clone(url: str) -> tuple[Path, bool]:
    tmp = Path(tempfile.mkdtemp(prefix='doc-link-'))
    for depth in (1, 0):
        cmd = ['git', 'clone', '--depth', str(depth), url, str(tmp)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            if proc.returncode == 0:
                return tmp, True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return tmp, False


def _repo_identity(path: Path, url: str | None) -> dict | None:
    if url:
        parsed = urlparse(url)
        return {'host': parsed.netloc.lower()}
    try:
        proc = subprocess.run(['git', 'config', '--get', 'remote.origin.url'],
                              cwd=str(path), capture_output=True, timeout=10)
        if proc.returncode == 0:
            parsed = urlparse(proc.stdout.decode('utf-8', 'ignore').strip())
            return {'host': parsed.netloc.lower()}
    except OSError:
        pass
    return None


def scan(repo: str, check_links: bool, timeout: int = DEFAULT_TIMEOUT,
         concurrency: int = DEFAULT_CONCURRENCY) -> list[dict]:
    path = Path(repo)
    cleanup = False
    identity = None
    if re.match(r'^(https?://|git@|ssh://|git://)', repo):
        path, ok = _clone(repo)
        cleanup = True
        identity = _repo_identity(path, repo)
        if not ok:
            print('WARN: clone incomplete; scanning what exists', file=sys.stderr)
    else:
        identity = _repo_identity(path, None)
    results = []
    for md in path.rglob('*'):
        if not md.is_file() or md.suffix.lower() not in ('.md', '.rst'):
            continue
        if any(part in SKIP_DIRS for part in md.parts):
            continue
        rel = md.relative_to(path)
        # docs/en is handled by a dedicated translation team and is out of
        # doc-agent scope globally.
        if len(rel.parts) >= 2 and rel.parts[:2] == ('docs', 'en'):
            continue
        try:
            text = md.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        for entry in extract_urls(text, str(md.relative_to(path))):
            entry['repo'] = path.name
            entry['category'] = classify(entry['url'], identity)
            results.append(entry)
    if check_links:
        urls = sorted({r['url'] for r in results})
        statuses = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for url, (status, detail) in zip(urls, pool.map(lambda u: check_url(u, timeout), urls)):
                statuses[url] = (status, detail)
        for r in results:
            status, detail = statuses.get(r['url'], ('未检查', ''))
            r['link_status'] = status
            r['link_detail'] = detail
    else:
        for r in results:
            r['link_status'] = '未检查'
            r['link_detail'] = '--no-check-links'
    if cleanup:
        shutil.rmtree(path, ignore_errors=True)
    return results


def emit_issues(results: list[dict]) -> list[dict]:
    issues = []
    for i, r in enumerate(results, 1):
        if r.get('link_status') == '失效':
            issues.append({
                'id': f'lint-link-{i}',
                'rule_id': 'R-COR-001',
                'dimension': 'correctness',
                'severity': 'critical',
                'location': {'file': r['file'], 'line': r.get('line')},
                'evidence': f"URL: {r['url']} ({r.get('link_detail')})",
                'suggestion': '修复或替换失效链接；确无替代时删除链接',
                'source': 'lint:link',
                'status': 'open',
            })
    return issues


def _write_xlsx(results: list[dict], output: str) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        print('Error: openpyxl not available. Install it with: pip install openpyxl',
              file=sys.stderr)
        sys.exit(1)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'URL分类报告'
    headers = ['代码仓名', '文件名', '行号', 'URL', '类别', '有效性', '状态详情']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
    for row, r in enumerate(results, 2):
        ws.cell(row=row, column=1, value=r.get('repo', ''))
        ws.cell(row=row, column=2, value=r.get('file', ''))
        ws.cell(row=row, column=3, value=r.get('line', ''))
        ws.cell(row=row, column=4, value=r.get('url', ''))
        ws.cell(row=row, column=5, value=r.get('category', ''))
        ws.cell(row=row, column=6, value=r.get('link_status', ''))
        ws.cell(row=row, column=7, value=r.get('link_detail', ''))
    wb.save(output)
    print(f'Report saved to: {output}')


def _summary(results: list[dict]) -> None:
    from collections import Counter
    cats = Counter(r['category'] for r in results)
    print(f'Total URLs: {len(results)}')
    for cat, n in cats.most_common():
        print(f'  {cat}: {n}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='doc_lint link',
                                     description='链接扫描与可达性检查')
    parser.add_argument('repo')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-check-links', action='store_true')
    parser.add_argument('--no-cleanup', action='store_true')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)
    results = scan(args.repo, not args.no_check_links, args.timeout, args.concurrency)
    if args.output and args.output.endswith('.xlsx'):
        _write_xlsx(results, args.output)
    elif not args.quiet:
        _summary(results)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    raise SystemExit(main())
