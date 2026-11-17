from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Issue, RepositoryRef


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NullStorage:
    def deliveries(self, repo_key: str, issue_number: int) -> list[dict[str, Any]]:
        return []

    def cursor(self, repo_key: str, issue_number: int) -> dict[str, Any]:
        return {}

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


class Storage:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS repositories(
          repo_key TEXT PRIMARY KEY, full_name TEXT NOT NULL, code_url TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scan_runs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, repo_key TEXT NOT NULL, mode TEXT NOT NULL,
          started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, summary_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS issues(
          repo_key TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL, title TEXT NOT NULL,
          author TEXT NOT NULL, updated_at TEXT, snapshot_json TEXT NOT NULL, last_action TEXT,
          PRIMARY KEY(repo_key, number));
        CREATE TABLE IF NOT EXISTS event_cursors(
          repo_key TEXT NOT NULL, issue_number INTEGER NOT NULL, last_seen_comment_id TEXT,
          last_handled_comment_id TEXT, maintainer_intervened INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL, PRIMARY KEY(repo_key, issue_number));
        CREATE TABLE IF NOT EXISTS drafts(
          id INTEGER PRIMARY KEY AUTOINCREMENT, repo_key TEXT NOT NULL, issue_number INTEGER NOT NULL,
          mode TEXT NOT NULL, trigger_id TEXT NOT NULL, body TEXT NOT NULL, evidence_json TEXT NOT NULL,
          created_at TEXT NOT NULL, review_state TEXT NOT NULL DEFAULT 'pending');
        CREATE TABLE IF NOT EXISTS delivery_ledger(
          idempotency_key TEXT PRIMARY KEY, repo_key TEXT NOT NULL, issue_number INTEGER NOT NULL,
          status TEXT NOT NULL, remote_id TEXT, body_hash TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS post_grants(
          repo_key TEXT NOT NULL, issue_number INTEGER NOT NULL, body_hash TEXT NOT NULL,
          expires_at TEXT NOT NULL, consumed_at TEXT,
          PRIMARY KEY(repo_key,issue_number,body_hash));
        CREATE TABLE IF NOT EXISTS audit_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, repo_key TEXT NOT NULL, action TEXT NOT NULL,
          detail_json TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register_repo(self, repo: RepositoryRef) -> None:
        self.db.execute("INSERT INTO repositories VALUES(?,?,?,?) ON CONFLICT(repo_key) DO UPDATE SET full_name=excluded.full_name,code_url=excluded.code_url,updated_at=excluded.updated_at",
                        (repo.key, repo.full_name, repo.code_url, now()))
        self.db.commit()

    def start_run(self, repo_key: str, mode: str) -> int:
        cur = self.db.execute("INSERT INTO scan_runs(repo_key,mode,started_at,status) VALUES(?,?,?,'running')",
                              (repo_key, mode, now()))
        self.db.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, summary: dict[str, Any]) -> None:
        self.db.execute("UPDATE scan_runs SET finished_at=?,status=?,summary_json=? WHERE id=?",
                        (now(), status, json.dumps(summary, ensure_ascii=False), run_id))
        self.db.commit()

    def upsert_issue(self, repo_key: str, issue: Issue, action: str) -> None:
        snapshot = {k: v for k, v in issue.__dict__.items() if k != "raw"}
        self.db.execute("""INSERT INTO issues VALUES(?,?,?,?,?,?,?,?)
          ON CONFLICT(repo_key,number) DO UPDATE SET state=excluded.state,title=excluded.title,
          author=excluded.author,updated_at=excluded.updated_at,snapshot_json=excluded.snapshot_json,last_action=excluded.last_action""",
                        (repo_key, issue.number, issue.state, issue.title, issue.author, issue.updated_at,
                         json.dumps(snapshot, ensure_ascii=False), action))
        self.db.commit()

    def cursor(self, repo_key: str, issue_number: int) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM event_cursors WHERE repo_key=? AND issue_number=?",
                              (repo_key, issue_number)).fetchone()
        return dict(row) if row else {}

    def deliveries(self, repo_key: str, issue_number: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM delivery_ledger WHERE repo_key=? AND issue_number=?",
            (repo_key, issue_number)).fetchall()]

    def clear_bot_retirement(self, repo_key: str, issue_number: int) -> None:
        self.db.execute("UPDATE event_cursors SET maintainer_intervened=0 WHERE repo_key=? AND issue_number=?",
                        (repo_key, issue_number))
        self.db.commit()

    def update_cursor(self, repo_key: str, issue_number: int, last_seen: str, last_handled: str,
                      maintainer_intervened: bool = False) -> None:
        self.db.execute("""INSERT INTO event_cursors VALUES(?,?,?,?,?,?)
          ON CONFLICT(repo_key,issue_number) DO UPDATE SET
          last_seen_comment_id=excluded.last_seen_comment_id,
          last_handled_comment_id=excluded.last_handled_comment_id,
          maintainer_intervened=max(event_cursors.maintainer_intervened,excluded.maintainer_intervened),
          updated_at=excluded.updated_at""",
                        (repo_key, issue_number, last_seen, last_handled, int(maintainer_intervened), now()))
        self.db.commit()

    def save_draft(self, repo_key: str, issue_number: int, mode: str, trigger_id: str,
                   body: str, evidence: dict[str, Any]) -> int:
        cur = self.db.execute("INSERT INTO drafts(repo_key,issue_number,mode,trigger_id,body,evidence_json,created_at) VALUES(?,?,?,?,?,?,?)",
                              (repo_key, issue_number, mode, trigger_id, body,
                               json.dumps(evidence, ensure_ascii=False), now()))
        self.db.commit()
        return int(cur.lastrowid)

    def get_draft(self, draft_id: int) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        return dict(row) if row else None

    def recent_template_drafts(self, repo_key: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute("""SELECT id,issue_number,body,created_at FROM drafts
          WHERE repo_key=? AND mode='template-auto' ORDER BY id DESC LIMIT ?""",
                               (repo_key, limit)).fetchall()
        return [dict(row) for row in rows]

    def grant_post(self, repo_key: str, issue_number: int, body_hash: str,
                   expires_at: str) -> None:
        self.db.execute("""INSERT INTO post_grants VALUES(?,?,?,?,NULL)
          ON CONFLICT(repo_key,issue_number,body_hash) DO UPDATE SET
          expires_at=excluded.expires_at,consumed_at=NULL""",
                        (repo_key, issue_number, body_hash, expires_at))
        self.db.commit()

    def consume_post_grant(self, repo_key: str, issue_number: int, body_hash: str) -> bool:
        with self.db:
            return self._consume_post_grant(repo_key, issue_number, body_hash)

    def _consume_post_grant(self, repo_key: str, issue_number: int, body_hash: str) -> bool:
        cur = self.db.execute("""UPDATE post_grants SET consumed_at=?
          WHERE repo_key=? AND issue_number=? AND body_hash=?
          AND consumed_at IS NULL AND expires_at>?""",
                              (now(), repo_key, issue_number, body_hash, now()))
        return cur.rowcount == 1

    def claim_delivery(self, key: str, repo_key: str, issue_number: int, body_hash: str,
                       require_grant: bool = False) -> bool:
        """Atomically claim a delivery and, when needed, consume its approval.

        Reserve the SQLite writer before checking other claims. The transaction
        ends before any network request; a crashed owner leaves a blocking ledger
        entry for reconciliation rather than allowing another process to repost.
        """
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute("""SELECT 1 FROM delivery_ledger
              WHERE idempotency_key=? OR (repo_key=? AND issue_number=?
              AND status IN ('posting','pending_delivery')) LIMIT 1""",
                                       (key, repo_key, issue_number)).fetchone()
            if existing:
                return False
            if require_grant and not self._consume_post_grant(repo_key, issue_number, body_hash):
                return False
            self.db.execute("INSERT INTO delivery_ledger VALUES(?,?,?,?,?,?,?)",
                            (key, repo_key, issue_number, "posting", "", body_hash, now()))
            return True

    def ledger(self, key: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM delivery_ledger WHERE idempotency_key=?", (key,)).fetchone()
        return dict(row) if row else {}

    def put_ledger(self, key: str, repo_key: str, issue_number: int, status: str,
                   remote_id: str, body_hash: str) -> None:
        self.db.execute("""INSERT INTO delivery_ledger VALUES(?,?,?,?,?,?,?)
          ON CONFLICT(idempotency_key) DO UPDATE SET status=excluded.status,remote_id=excluded.remote_id,updated_at=excluded.updated_at""",
                        (key, repo_key, issue_number, status, remote_id, body_hash, now()))
        self.db.commit()

    def audit(self, repo_key: str, action: str, detail: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO audit_events(repo_key,action,detail_json,created_at) VALUES(?,?,?,?)",
                        (repo_key, action, json.dumps(detail, ensure_ascii=False), now()))
        self.db.commit()

    def dashboard(self) -> dict[str, Any]:
        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(row) for row in self.db.execute(sql).fetchall()]
        return {
            "repositories": rows("SELECT * FROM repositories ORDER BY full_name"),
            "runs": rows("SELECT * FROM scan_runs ORDER BY id DESC LIMIT 50"),
            "issues": rows("SELECT repo_key,number,state,title,author,updated_at,last_action FROM issues ORDER BY updated_at DESC LIMIT 200"),
            "drafts": rows("SELECT id,repo_key,issue_number,mode,trigger_id,body,created_at,review_state FROM drafts ORDER BY id DESC LIMIT 100"),
            "deliveries": rows("SELECT * FROM delivery_ledger ORDER BY updated_at DESC LIMIT 100"),
        }


class ReadOnlyStorage(Storage):
    """Read existing state without schema migrations or persistent writes."""
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
