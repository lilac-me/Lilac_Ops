from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analyzer import analyze_issue, shadow_draft, template_reply
from .gitcode import GitCodeClient, GitCodeError, MemberSnapshot
from .local_settings import LocalSettings, read_token, write_token
from .models import Candidate, Comment, Issue, RepositoryRef
from .storage import NullStorage, ReadOnlyStorage, Storage, now


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def is_open(state: str) -> bool:
    return state.lower() in {"open", "opened", "reopen", "reopened"}


class IssueAgent:
    def __init__(self, data_dir: str | Path, client_factory=GitCodeClient):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.client_factory = client_factory

    @staticmethod
    def _member_state(login: str, hint: bool | None, members: MemberSnapshot,
                      configured: set[str]) -> str:
        if hint is True or login.lower() in configured or login.lower() in members.members:
            return "member"
        if hint is False or members.complete:
            return "non_member"
        return "unknown"

    def _candidate(self, repo: RepositoryRef, issue: Issue, comments: list[Comment],
                   members: MemberSnapshot, configured: set[str], storage: Storage | NullStorage) -> Candidate:
        if not is_open(issue.state):
            return Candidate(issue, comments, "none", "", issue.author, "unknown", "skip", "issue is not open")
        deliveries = storage.deliveries(repo.key, issue.number)
        # A remote ID and unchanged body must both match our confirmed ledger.
        bot_ids = {str(row["remote_id"]) for row in deliveries if row["status"] == "confirmed"
                   for comment in comments if str(row["remote_id"]) == comment.id
                   and row["body_hash"] == digest(comment.body)}
        for comment in comments:
            if comment.id in bot_ids:
                continue
            if self._member_state(comment.author, comment.member_hint, members, configured) == "member":
                return Candidate(issue, comments, "none", comment.id, comment.author, "member", "retire",
                                 "a repository member has commented")
        if any(row["status"] in {"posting", "pending_delivery"} for row in deliveries):
            return Candidate(issue, comments, "none", "", issue.author, "unknown", "block",
                             "previous delivery requires reconciliation")
        cursor = storage.cursor(repo.key, issue.number)
        stale_bot_retirement = (cursor.get("maintainer_intervened") and
                                str(cursor.get("last_seen_comment_id") or "") in bot_ids)
        if cursor.get("maintainer_intervened") and not stale_bot_retirement:
            return Candidate(issue, comments, "none", "", issue.author, "unknown", "retire",
                             "maintainer previously intervened")
        if stale_bot_retirement and isinstance(storage, Storage) and not isinstance(storage, ReadOnlyStorage):
            storage.clear_bot_retirement(repo.key, issue.number)
        if comments and comments[-1].id in bot_ids:
            return Candidate(issue, comments, "none", comments[-1].id, comments[-1].author,
                             "unknown", "skip", "awaiting user update after automated reply")
        actor_comment = comments[-1] if comments else None
        actor = actor_comment.author if actor_comment else issue.author
        hint = actor_comment.member_hint if actor_comment else issue.member_hint
        state = self._member_state(actor, hint, members, configured)
        trigger_id = actor_comment.id if actor_comment else f"issue:{issue.number}"
        trigger = "new_non_member_comment" if actor_comment else "new_issue"
        if state != "non_member":
            return Candidate(issue, comments, trigger, trigger_id, actor, state, "block",
                             "member identity is not safely resolved" if state == "unknown" else "actor is a repository member")
        if str(cursor.get("last_handled_comment_id") or "") == trigger_id:
            return Candidate(issue, comments, trigger, trigger_id, actor, state, "skip", "event already handled")
        return Candidate(issue, comments, trigger, trigger_id, actor, state, "process", "eligible non-member event")

    def scan(self, repo: RepositoryRef, mode: str = "dry-run", issue_number: int | None = None,
             max_issues: int = 0, local_path: str = "", configured_maintainers: list[str] | None = None,
             allow_post: bool = False, scheduled: bool = False) -> dict[str, Any]:
        if mode not in {"dry-run", "shadow", "template-auto"}:
            raise ValueError("mode must be dry-run, shadow, or template-auto")
        dry = mode == "dry-run"
        repo_dir = self.data_dir / "repos" / repo.slug
        db_path = repo_dir / "state.sqlite3"
        storage: Storage | NullStorage = (ReadOnlyStorage(db_path) if db_path.exists() else NullStorage()) if dry else Storage(db_path)
        client = self.client_factory(repo, read_token=read_token(), write_token=write_token())
        configured = {x.strip().lower() for x in (configured_maintainers or []) if x.strip()}
        run_id = None
        if not dry:
            storage.register_repo(repo)
            run_id = storage.start_run(repo.key, mode)
        report: dict[str, Any] = {"repo": asdict(repo), "mode": mode, "started_at": now(),
                                  "member_source": {}, "counts": {}, "items": [], "warnings": []}
        try:
            issues = [client.get_issue(issue_number)] if issue_number else client.list_issues("open")
            members = client.list_members()
            members.members |= configured
            report["member_source"] = {"source": members.source, "complete": members.complete,
                                       "count": len(members.members), "error": members.error}
            selected = issues if max_issues == 0 else issues[:max_issues]
            for issue in selected:
                try:
                    comments = client.list_comments(issue.number)
                except GitCodeError as exc:
                    comments = []
                    item = {"issue": issue.number, "title": issue.title, "action": "block",
                            "reason": f"comments could not be read: {exc}"}
                    report["items"].append(item)
                    continue
                candidate = self._candidate(repo, issue, comments, members, configured, storage)
                item = candidate.public()
                shadow_only = mode == "shadow" and candidate.action == "block" and candidate.member_state == "unknown"
                if shadow_only:
                    item["action"] = "shadow_only"
                    item["reason"] = "membership is unknown; read-only analysis is allowed but publishing remains blocked"
                if candidate.action == "process" or shadow_only:
                    follow_up = candidate.trigger == "new_non_member_comment"
                    if mode == "dry-run":
                        item["proposed_action"] = "shadow-analysis or template draft"
                    elif mode == "shadow":
                        evidence = analyze_issue(repo, issue, repo_dir / "worktrees", local_path)
                        body = shadow_draft(issue, evidence)
                        item.update(evidence=evidence, draft=body,
                                    draft_id=storage.save_draft(repo.key, issue.number, mode,
                                                                candidate.trigger_comment_id, body, evidence))
                        storage.update_cursor(repo.key, issue.number, comments[-1].id if comments else "",
                                              "" if shadow_only else candidate.trigger_comment_id)
                    else:
                        body = template_reply(issue, follow_up)
                        item["draft"] = body
                        draft_id = storage.save_draft(repo.key, issue.number, mode,
                                                      candidate.trigger_comment_id, body, {"kind": "template"})
                        item["draft_id"] = draft_id
                        if allow_post:
                            item["delivery"] = self._publish(client, repo, candidate, body, storage,
                                                               configured, scheduled=scheduled)
                            if item["delivery"]["status"] == "confirmed":
                                remote_id = str(item["delivery"].get("remote_id") or candidate.trigger_comment_id)
                                storage.update_cursor(repo.key, issue.number, remote_id, candidate.trigger_comment_id)
                        else:
                            item["delivery"] = {"status": "not_requested"}
                if not dry:
                    storage.upsert_issue(repo.key, issue, candidate.action)
                    if candidate.action == "retire":
                        storage.update_cursor(repo.key, issue.number,
                                              comments[-1].id if comments else "", "", True)
                report["items"].append(item)
            actions: dict[str, int] = {}
            for item in report["items"]:
                action = str(item.get("action") or "unknown")
                actions[action] = actions.get(action, 0) + 1
            report["counts"] = {"visible_open": len(issues), "selected": len(selected), "actions": actions}
            report["finished_at"] = now()
            if not dry:
                reports = repo_dir / "reports"
                reports.mkdir(parents=True, exist_ok=True)
                (reports / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                storage.finish_run(int(run_id), "success", report["counts"])
            return report
        except Exception as exc:
            if not dry and run_id is not None:
                storage.finish_run(int(run_id), "failed", {"error": str(exc)})
            raise
        finally:
            if isinstance(storage, Storage):
                storage.close()

    def _publish(self, client: GitCodeClient, repo: RepositoryRef, candidate: Candidate, body: str,
                 storage: Storage, configured: set[str], scheduled: bool = False) -> dict[str, Any]:
        policy = LocalSettings(self.data_dir).policy(repo)
        if scheduled and not policy.get("scheduled_post"):
            return {"status": "blocked", "reason": "scheduled posting is disabled for this repository"}
        if not scheduled and not policy.get("interactive_post"):
            return {"status": "blocked", "reason": "interactive posting is disabled for this repository"}
        if not getattr(client, "write_token", write_token()):
            return {"status": "blocked", "reason": "GitCode write token is not configured"}
        body_hash = digest(body)
        key = digest(repo.key, str(candidate.issue.number), candidate.trigger_comment_id, body_hash)
        previous = storage.ledger(key)
        if previous:
            return {"status": previous["status"], "remote_id": previous.get("remote_id"), "deduplicated": True}
        fresh_issue = client.get_issue(candidate.issue.number)
        fresh_comments = client.list_comments(candidate.issue.number)
        fresh_members = client.list_members()
        fresh_members.members |= configured
        fresh = self._candidate(repo, fresh_issue, fresh_comments, fresh_members, configured, storage)
        if fresh.action != "process" or fresh.trigger_comment_id != candidate.trigger_comment_id:
            return {"status": "blocked", "reason": "issue, comments, or membership changed before publish"}
        if not storage.claim_delivery(key, repo.key, candidate.issue.number, body_hash,
                                      require_grant=not scheduled):
            previous = storage.ledger(key)
            if previous:
                return {"status": previous["status"], "remote_id": previous.get("remote_id"),
                        "deduplicated": True}
            return {"status": "blocked", "reason": "delivery is already in progress or this reply lacks an unexpired one-time approval"}
        try:
            created = client.post_comment(candidate.issue.number, body)
        except GitCodeError as exc:
            storage.put_ledger(key, repo.key, candidate.issue.number, "pending_delivery", "", body_hash)
            return {"status": "pending_delivery", "reason": str(exc)}
        remote_id = str(created.get("id") or "")
        try:
            confirmed = [c for c in client.list_comments(candidate.issue.number) if c.id == remote_id or c.body.strip() == body.strip()]
        except GitCodeError:
            confirmed = []
        status = "confirmed" if confirmed else "pending_delivery"
        storage.put_ledger(key, repo.key, candidate.issue.number, status, remote_id, body_hash)
        storage.audit(repo.key, "comment_post", {"issue": candidate.issue.number, "status": status, "remote_id": remote_id})
        return {"status": status, "remote_id": remote_id}
