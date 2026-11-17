from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .local_settings import LocalSettings, delete_token, get_token, read_token, save_token, write_token
from .models import RepositoryRef
from .observatory import serve
from .service import IssueAgent, digest
from .storage import Storage
from .web_observatory import WebObservatory


def default_data_dir() -> Path:
    configured = os.getenv("GITCODE_ISSUE_AGENT_DATA", "")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "9-17-scan-issue-agent"
    return Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "9-17-scan-issue-agent"


def load_config(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("repositories"), list):
        raise ValueError("config must contain a repositories array")
    return data


def repo_from_config(row: dict) -> RepositoryRef:
    return RepositoryRef.parse(str(row["name"]), str(row.get("code_url") or ""), str(row.get("default_branch") or ""))


def add_common_scan(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True, help="GitCode owner/repo or repository URL")
    parser.add_argument("--code-url", default="", help="Override Git clone URL")
    parser.add_argument("--default-branch", default="")
    parser.add_argument("--mode", choices=["dry-run", "shadow", "template-auto"], default="dry-run")
    parser.add_argument("--issue", type=int)
    parser.add_argument("--max-issues", type=int, default=0, help="0 means unlimited (default)")
    parser.add_argument("--workspace", default="", help="Use an existing local Git checkout for shadow analysis")
    parser.add_argument("--maintainer", action="append", default=[], help="Additional trusted login; repeatable")
    parser.add_argument("--allow-public-degraded", action="store_true",
                        help="Allow an incomplete public-API batch preview without a read token")
    parser.add_argument("--allow-post", action="store_true", help="Request real posting; repository policy and one-time approval still apply")
    parser.add_argument("--data-dir", default=str(default_data_dir()))


def scan_one(args) -> dict:
    if args.issue is not None and args.issue < 1:
        raise ValueError("--issue must be positive")
    if args.allow_post and args.mode != "template-auto":
        raise ValueError("--allow-post is valid only with --mode template-auto")
    if args.issue is None and not read_token() and not args.allow_public_degraded:
        raise ValueError(
            "batch scan requires a read token for reliable membership; configure it in the "
            "local Observatory settings page, or explicitly accept an incomplete preview "
            "with --allow-public-degraded"
        )
    repo = RepositoryRef.parse(args.repo, args.code_url, args.default_branch)
    return IssueAgent(args.data_dir).scan(repo, args.mode, args.issue, args.max_issues,
                                          args.workspace, args.maintainer, args.allow_post)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="issue-agent", description="Cross-repository GitCode Issue Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Show safe configuration status")
    check.add_argument("--data-dir", default=str(default_data_dir()))
    settings = sub.add_parser("settings", help="Local settings JSON API; secrets accepted only on stdin")
    settings.add_argument("--data-dir", default=str(default_data_dir()))
    scan = sub.add_parser("scan", help="Scan one repository")
    add_common_scan(scan)
    many = sub.add_parser("scan-all", help="Scan every repository in a JSON config")
    many.add_argument("--config", required=True)
    many.add_argument("--data-dir", default=str(default_data_dir()))
    many.add_argument("--mode", choices=["dry-run", "shadow", "template-auto"], default="")
    many.add_argument("--allow-post", action="store_true")
    many.add_argument("--allow-public-degraded", action="store_true")
    init = sub.add_parser("init-config", help="Write a safe example JSON configuration")
    init.add_argument("path")
    web = sub.add_parser("serve", help="Start the local read-only Observatory")
    web_source = web.add_mutually_exclusive_group()
    web_source.add_argument("--db", help="Path to one repository state.sqlite3")
    web_source.add_argument("--data-dir", default=str(default_data_dir()), help="Aggregate every repository database")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    visual = sub.add_parser("visualize", help="Install and start the full MindSpeed Issue Observatory")
    visual.add_argument("--repo", required=True, help="Previously selected GitCode owner/repo or URL")
    visual.add_argument("--port", type=int, default=5173)
    visual.add_argument("--sync-interval-minutes", type=int, default=60)
    visual.add_argument("--data-dir", default=str(default_data_dir()))
    visual.add_argument("--refresh", action="store_true", help="Fast-forward the managed Observatory checkout")
    visual.add_argument("--no-open", action="store_true", help="Do not open the local URL in a browser")
    visual.add_argument("--allow-public-degraded", action="store_true",
                        help="Allow anonymous incomplete GitCode visibility")
    visual_status = sub.add_parser("visualize-status", help="Show full Observatory service status")
    visual_status.add_argument("--data-dir", default=str(default_data_dir()))
    visual_stop = sub.add_parser("visualize-stop", help="Stop the managed full Observatory service")
    visual_stop.add_argument("--data-dir", default=str(default_data_dir()))
    schedule = sub.add_parser("schedule", help="Run configured repositories at a fixed local interval")
    schedule.add_argument("--config", required=True)
    schedule.add_argument("--data-dir", default=str(default_data_dir()))
    schedule.add_argument("--interval-seconds", type=int, default=3600)
    schedule.add_argument("--allow-post", action="store_true",
                          help="Request guarded posting for repositories configured as template-auto")
    schedule.add_argument("--allow-public-degraded", action="store_true")
    schedule.add_argument("--once", action="store_true")
    return parser


def scan_config(config_path: str, data_dir: str, override_mode: str = "", allow_post: bool = False,
                allow_public_degraded: bool = False) -> list[dict]:
    config = load_config(config_path)
    if not read_token() and not allow_public_degraded:
        raise ValueError(
            "multi-repository scan requires a read token; configure it in the local "
            "Observatory settings page, or explicitly accept incomplete results "
            "with --allow-public-degraded"
        )
    reports = []
    for row in config["repositories"]:
        repo = repo_from_config(row)
        mode = override_mode or str(row.get("mode") or "dry-run")
        reports.append(IssueAgent(data_dir).scan(repo, mode=mode,
                                                 max_issues=int(row.get("max_issues", 0)),
                                                 local_path=str(row.get("workspace") or ""),
                                                 configured_maintainers=list(row.get("maintainers") or []),
                                                 allow_post=allow_post, scheduled=True))
    return reports


def settings_command(data_dir: str) -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("settings request must be an object")
    op = payload.get("operation")
    settings = LocalSettings(data_dir)
    if op == "status":
        return {"config": settings.load(), "read_token_configured": bool(read_token()),
                "write_token_configured": bool(write_token()),
                "secure_storage": os.name == "nt", "token_source": {
                    "read": "credential_manager" if get_token("read") else "legacy_environment" if read_token() else "none",
                    "write": "credential_manager" if get_token("write") else "legacy_environment" if write_token() else "none"}}
    if op == "save_config":
        return {"config": settings.save(payload.get("config") or {})}
    if op == "save_token":
        save_token(str(payload.get("kind") or ""), str(payload.get("token") or ""))
        return {"configured": True, "kind": payload["kind"]}
    if op == "delete_token":
        delete_token(str(payload.get("kind") or ""))
        return {"configured": False, "kind": payload["kind"]}
    if op in {"drafts", "approve_draft"}:
        repo = RepositoryRef.parse(str(payload.get("repo") or ""))
        if repo.full_name not in settings.load()["repositories"]:
            raise ValueError("repository is not trusted")
        with Storage(Path(data_dir) / "repos" / repo.slug / "state.sqlite3") as storage:
            if op == "drafts":
                return {"drafts": storage.recent_template_drafts(repo.key)}
            if not settings.policy(repo).get("interactive_post"):
                raise ValueError("interactive posting is disabled for this repository")
            draft_id = int(payload.get("draft_id") or 0)
            draft = storage.get_draft(draft_id)
            if not draft or draft["repo_key"] != repo.key or draft["mode"] != "template-auto":
                raise ValueError("template draft not found in selected repository")
            body_hash = digest(draft["body"])
            expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            storage.grant_post(repo.key, int(draft["issue_number"]), body_hash, expiry)
            return {"approved": True, "repo": repo.full_name,
                    "issue": draft["issue_number"], "body_hash": body_hash,
                    "expires_at": expiry}
    raise ValueError("unsupported settings operation")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            config = LocalSettings(args.data_dir).load()
            result = {"data_dir": str(Path(args.data_dir).expanduser().resolve()),
                      "read_token": bool(read_token()), "batch_scan_ready": bool(read_token()),
                      "write_token": bool(write_token()),
                      "trusted_repositories": list(config["repositories"]),
                      "default_repository": config["default_repository"]}
        elif args.command == "settings":
            result = settings_command(args.data_dir)
        elif args.command == "scan":
            result = scan_one(args)
        elif args.command == "scan-all":
            result = scan_config(args.config, args.data_dir, args.mode, args.allow_post,
                                 args.allow_public_degraded)
        elif args.command == "init-config":
            target = Path(args.path).expanduser().resolve()
            if target.exists():
                raise ValueError(f"refusing to overwrite {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            sample = {"repositories": [{"name": "example-org/example-repo", "mode": "dry-run",
                                         "max_issues": 0, "maintainers": []}]}
            target.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result = {"created": str(target)}
        elif args.command == "serve":
            serve(args.db or args.data_dir, args.host, args.port)
            return 0
        elif args.command == "visualize":
            # First use deliberately opens the local setup page even without a token.
            # Collection remains credential-scoped and labels anonymous visibility.
            result = WebObservatory(args.data_dir).start(
                RepositoryRef.parse(args.repo), port=args.port,
                sync_minutes=args.sync_interval_minutes,
                open_browser=not args.no_open, refresh=args.refresh,
            )
        elif args.command == "visualize-status":
            result = WebObservatory(args.data_dir).status()
        elif args.command == "visualize-stop":
            result = WebObservatory(args.data_dir).stop()
        else:
            if args.interval_seconds < 10 and not args.once:
                raise ValueError("--interval-seconds must be at least 10")
            while True:
                result = scan_config(args.config, args.data_dir, allow_post=args.allow_post,
                                     allow_public_degraded=args.allow_public_degraded)
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                if args.once:
                    return 0
                time.sleep(args.interval_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
