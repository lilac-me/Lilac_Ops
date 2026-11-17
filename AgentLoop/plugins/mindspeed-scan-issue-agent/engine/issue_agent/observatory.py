from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .storage import Storage


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GitCode Issue Observatory</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f4f6f8;color:#172b4d}
header{background:#172b4d;color:white;padding:20px 28px}main{padding:24px;max-width:1400px;margin:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.card,section{background:white;border:1px solid #dfe1e6;border-radius:10px;padding:16px;margin-bottom:18px}
.n{font-size:30px;font-weight:700;color:#0052cc}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #ebecf0}th{background:#f7f8f9}.muted{color:#6b778c}.bad{color:#b71c1c}.ok{color:#1b5e20}
</style></head><body><header><h1>GitCode Repository Issue Observatory</h1><div>跨仓、只读、本地控制面</div></header>
<main><div id="status" class="muted">正在加载…</div><div class="cards" id="cards"></div>
<section><h2>仓库</h2><table><thead><tr><th>仓库</th><th>最近更新</th></tr></thead><tbody id="repos"></tbody></table></section>
<section><h2>最近 Issue</h2><table><thead><tr><th>仓库</th><th>#</th><th>标题</th><th>作者</th><th>动作</th></tr></thead><tbody id="issues"></tbody></table></section>
<section><h2>草稿</h2><table><thead><tr><th>ID</th><th>仓库</th><th>Issue</th><th>模式</th><th>正文</th></tr></thead><tbody id="drafts"></tbody></table></section></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){try{const d=await (await fetch('/api/dashboard')).json();
document.getElementById('status').innerHTML='<span class="ok">● 服务正常</span> · 数据仅来自本机 SQLite';
const counts=[['仓库',d.repositories.length],['扫描轮次',d.runs.length],['Issue',d.issues.length],['草稿',d.drafts.length],['投递记录',d.deliveries.length]];
document.getElementById('cards').innerHTML=counts.map(x=>`<div class="card"><div class="muted">${x[0]}</div><div class="n">${x[1]}</div></div>`).join('');
document.getElementById('repos').innerHTML=d.repositories.map(x=>`<tr><td>${esc(x.full_name)}</td><td>${esc(x.updated_at)}</td></tr>`).join('');
document.getElementById('issues').innerHTML=d.issues.map(x=>`<tr><td>${esc(x.repo_key)}</td><td>${x.number}</td><td>${esc(x.title)}</td><td>${esc(x.author)}</td><td>${esc(x.last_action)}</td></tr>`).join('');
document.getElementById('drafts').innerHTML=d.drafts.map(x=>`<tr><td>${x.id}</td><td>${esc(x.repo_key)}</td><td>${x.issue_number}</td><td>${esc(x.mode)}</td><td>${esc(x.body).slice(0,180)}</td></tr>`).join('');
}catch(e){document.getElementById('status').innerHTML='<span class="bad">加载失败：</span>'+esc(e)}}load();setInterval(load,10000);
</script></body></html>"""


def database_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return sorted(path.rglob("state.sqlite3"))
    return [path]


def combined_dashboard(source: str | Path) -> dict:
    merged = {"repositories": [], "runs": [], "issues": [], "drafts": [], "deliveries": []}
    for path in database_paths(source):
        with Storage(path) as storage:
            data = storage.dashboard()
        for key in merged:
            merged[key].extend(data[key])
    return merged


def handler_for(db_path: str | Path):
    path = Path(db_path).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/health":
                self._send(200, "application/json", b'{"status":"ok"}')
                return
            if route == "/api/dashboard":
                data = combined_dashboard(path)
                self._send(200, "application/json; charset=utf-8",
                           json.dumps(data, ensure_ascii=False).encode("utf-8"))
                return
            if route == "/":
                self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            self._send(404, "application/json", b'{"error":"not found"}')

    return Handler


def serve(db_path: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Observatory binds to loopback only")
    server = ThreadingHTTPServer((host, port), handler_for(db_path))
    actual = server.server_address[1]
    print(f"Observatory: http://127.0.0.1:{actual}", flush=True)
    server.serve_forever()
