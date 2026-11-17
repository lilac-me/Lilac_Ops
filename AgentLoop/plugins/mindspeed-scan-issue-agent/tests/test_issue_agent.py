from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN / "engine"))

from issue_agent.gitcode import MemberSnapshot  # noqa: E402
from issue_agent.models import Comment, Issue, RepositoryRef  # noqa: E402
from issue_agent.observatory import handler_for  # noqa: E402
from issue_agent.service import IssueAgent  # noqa: E402
from issue_agent.service import digest  # noqa: E402
from issue_agent.local_settings import LocalSettings  # noqa: E402
from issue_agent.storage import Storage  # noqa: E402
from issue_agent.cli import main, scan_one  # noqa: E402
from issue_agent.web_observatory import WebObservatory  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402


def issue(number=1, comments=0, hint=False):
    return Issue(number, "open", "ParserError in load_config", "load_config fails for config.json",
                 "external-user", comments_count=comments, member_hint=hint)


class FakeClient:
    issues = [issue()]
    comments = {1: []}
    members = MemberSnapshot({"maintainer"}, True, "fixture")
    posted = []

    def __init__(self, repo, read_token="", write_token=""):
        self.repo = repo

    def list_issues(self, state="open"):
        return list(self.issues)

    def get_issue(self, number):
        return next(x for x in self.issues if x.number == number)

    def list_comments(self, number):
        return list(self.comments.get(number, []))

    def list_members(self):
        return MemberSnapshot(set(self.members.members), self.members.complete, self.members.source, self.members.error)

    def post_comment(self, number, body):
        remote = Comment("9001", body, "bot")
        self.comments.setdefault(number, []).append(remote)
        self.posted.append((number, body))
        return {"id": "9001", "body": body}


class RepositoryRefTests(unittest.TestCase):
    def test_parse_owner_repo_and_url(self):
        a = RepositoryRef.parse("Org/Repo")
        b = RepositoryRef.parse("https://gitcode.com/Org/Repo.git")
        self.assertEqual(a.full_name, "Org/Repo")
        self.assertEqual(a.key, b.key)

    def test_rejects_ambiguous_path(self):
        with self.assertRaises(ValueError):
            RepositoryRef.parse("RepoOnly")


class AgentTests(unittest.TestCase):
    def setUp(self):
        FakeClient.issues = [issue()]
        FakeClient.comments = {1: []}
        FakeClient.members = MemberSnapshot({"maintainer"}, True, "fixture")
        FakeClient.posted = []
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = RepositoryRef.parse("Org/Repo")
        self.agent = IssueAgent(self.tmp.name, client_factory=FakeClient)

    def tearDown(self):
        self.tmp.cleanup()

    def seed_bot_reply(self, retired=False):
        db = Path(self.tmp.name) / "repos" / self.repo.slug / "state.sqlite3"
        with Storage(db) as store:
            store.put_ledger("bot", self.repo.key, 1, "confirmed", "10", digest("template"))
            store.update_cursor(self.repo.key, 1, "10", "issue:1", retired)
        FakeClient.comments = {1: [Comment("10", "template", "maintainer", member_hint=True)]}
        return db

    def test_member_bot_waits_then_user_update_is_candidate(self):
        self.seed_bot_reply()
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "skip")
        FakeClient.comments[1].append(Comment("11", "new reproduction", "external-user", member_hint=False))
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "process")

    def test_handled_followup_skipped_in_readonly_scan(self):
        db = self.seed_bot_reply()
        FakeClient.comments[1].append(Comment("11", "new reproduction", "external-user", member_hint=False))
        with Storage(db) as store:
            store.update_cursor(self.repo.key, 1, "11", "11")
        before = db.read_bytes()
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "skip")
        self.assertEqual(db.read_bytes(), before)

    def test_old_bot_retirement_recovered_but_human_still_retires(self):
        db = self.seed_bot_reply(retired=True)
        FakeClient.comments[1].append(Comment("11", "new reproduction", "external-user", member_hint=False))
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "process")
        self.agent.scan(self.repo, mode="template-auto")
        with Storage(db) as store:
            self.assertFalse(store.cursor(self.repo.key, 1)["maintainer_intervened"])
        FakeClient.comments[1].append(Comment("12", "I'll investigate", "maintainer", member_hint=True))
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "retire")

    def test_edited_bot_comment_is_not_automatically_exempt(self):
        self.seed_bot_reply()
        FakeClient.comments[1][0].body = "human edited response"
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "retire")

    def test_ledger_is_scoped_and_uncertain_delivery_blocks(self):
        db = self.seed_bot_reply()
        other = RepositoryRef.parse("Other/Repo")
        self.assertEqual(self.agent.scan(other)["items"][0]["action"], "retire")
        FakeClient.comments = {1: []}
        with Storage(db) as store:
            store.put_ledger("uncertain", self.repo.key, 1, "pending_delivery", "", digest("reply"))
        self.assertEqual(self.agent.scan(self.repo)["items"][0]["action"], "block")

    def test_templates_have_no_hidden_markers(self):
        from issue_agent.analyzer import template_reply
        for follow_up in (False, True):
            self.assertNotIn("<!--", template_reply(issue(), follow_up))

    def test_dry_run_is_zero_write(self):
        report = self.agent.scan(self.repo, mode="dry-run", max_issues=5)
        self.assertEqual(report["items"][0]["action"], "process")
        self.assertFalse(any(Path(self.tmp.name).rglob("*")))

    def test_batch_cli_requires_read_token_unless_degraded_is_explicit(self):
        keys = ("GITCODE_READ_TOKEN", "GITCODE_TOKEN")
        old = {key: os.environ.pop(key, None) for key in keys}
        args = Namespace(issue=None, allow_post=False, mode="dry-run",
                         allow_public_degraded=False, repo="Org/Repo", code_url="",
                         default_branch="", max_issues=5, workspace="", maintainer=[],
                         data_dir=self.tmp.name)
        try:
            with self.assertRaisesRegex(ValueError, "batch scan requires"):
                scan_one(args)
        finally:
            for key, value in old.items():
                if value is not None:
                    os.environ[key] = value

    def test_unknown_membership_blocks(self):
        FakeClient.members = MemberSnapshot(set(), False, "fixture", "denied")
        FakeClient.issues = [issue(hint=None)]
        report = self.agent.scan(self.repo, mode="dry-run")
        self.assertEqual(report["items"][0]["action"], "block")

    def test_shadow_allows_unknown_identity_without_publish(self):
        FakeClient.members = MemberSnapshot(set(), False, "fixture", "denied")
        FakeClient.issues = [issue(hint=None)]
        checkout = Path(self.tmp.name) / "unknown-checkout"
        checkout.mkdir()
        subprocess.run(["git", "init"], cwd=checkout, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        report = self.agent.scan(self.repo, mode="shadow", local_path=str(checkout))
        self.assertEqual(report["items"][0]["action"], "shadow_only")
        self.assertEqual(FakeClient.posted, [])

    def test_maintainer_comment_retires(self):
        FakeClient.comments = {1: [Comment("8", "taking this", "maintainer", member_hint=True)]}
        report = self.agent.scan(self.repo, mode="dry-run")
        self.assertEqual(report["items"][0]["action"], "retire")

    def test_shadow_persists_draft_and_commit(self):
        checkout = Path(self.tmp.name) / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init"], cwd=checkout, check=True, stdout=subprocess.PIPE)
        (checkout / "parser.py").write_text("def load_config():\n    raise ParserError()\n", encoding="utf-8")
        subprocess.run(["git", "add", "parser.py"], cwd=checkout, check=True, stdout=subprocess.PIPE)
        env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
               "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com"}
        subprocess.run(["git", "commit", "-m", "fixture"], cwd=checkout, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        report = self.agent.scan(self.repo, mode="shadow", local_path=str(checkout))
        item = report["items"][0]
        self.assertTrue(item["evidence"]["commit"] != "unknown")
        self.assertTrue(item["draft_id"] > 0)
        self.assertTrue(list(Path(self.tmp.name).rglob("state.sqlite3")))

    def test_real_post_is_blocked_without_environment_gates(self):
        old = {key: os.environ.pop(key, None) for key in
               ("GITCODE_ALLOW_REAL_POST", "GITCODE_POST_TARGET", "GITCODE_WRITE_TOKEN")}
        try:
            report = self.agent.scan(self.repo, mode="template-auto", allow_post=True)
            self.assertEqual(report["items"][0]["delivery"]["status"], "blocked")
            self.assertEqual(FakeClient.posted, [])
        finally:
            for key, value in old.items():
                if value is not None:
                    os.environ[key] = value

    def test_confirmed_post_does_not_reply_to_itself_next_run(self):
        LocalSettings(self.tmp.name).save({"default_repository": "Org/Repo",
            "repositories": {"Org/Repo": {"interactive_post": True, "scheduled_post": False}}})
        old = os.environ.get("GITCODE_WRITE_TOKEN")
        os.environ["GITCODE_WRITE_TOKEN"] = "fixture-only"
        try:
            preview = self.agent.scan(self.repo, mode="template-auto")
            from datetime import datetime, timedelta, timezone
            expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            with Storage(Path(self.tmp.name) / "repos" / self.repo.slug / "state.sqlite3") as storage:
                storage.grant_post(self.repo.key, 1, digest(preview["items"][0]["draft"]), expiry)
            first = self.agent.scan(self.repo, mode="template-auto", allow_post=True)
            self.assertEqual(first["items"][0]["delivery"]["status"], "confirmed")
            second = self.agent.scan(self.repo, mode="template-auto", allow_post=True)
            self.assertEqual(second["items"][0]["action"], "skip")
            self.assertEqual(len(FakeClient.posted), 1)
        finally:
            if old is None: os.environ.pop("GITCODE_WRITE_TOKEN", None)
            else: os.environ["GITCODE_WRITE_TOKEN"] = old

    def test_interactive_post_requires_matching_one_time_draft_grant(self):
        LocalSettings(self.tmp.name).save({"default_repository": "Org/Repo",
            "repositories": {"Org/Repo": {"interactive_post": True, "scheduled_post": False}}})
        keys = ("GITCODE_ALLOW_REAL_POST", "GITCODE_POST_TARGET", "GITCODE_WRITE_TOKEN")
        old = {key: os.environ.pop(key, None) for key in keys}
        os.environ["GITCODE_WRITE_TOKEN"] = "fixture-only"
        try:
            preview = self.agent.scan(self.repo, mode="template-auto")
            draft = preview["items"][0]
            blocked = self.agent.scan(self.repo, mode="template-auto", allow_post=True)
            self.assertEqual(blocked["items"][0]["delivery"]["status"], "blocked")
            self.assertEqual(FakeClient.posted, [])
            db = Path(self.tmp.name) / "repos" / self.repo.slug / "state.sqlite3"
            with Storage(db) as storage:
                from datetime import datetime, timedelta, timezone
                expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                storage.grant_post(self.repo.key, 1, digest(draft["draft"]), expiry)
            posted = self.agent.scan(self.repo, mode="template-auto", allow_post=True)
            self.assertEqual(posted["items"][0]["delivery"]["status"], "confirmed")
            self.assertEqual(len(FakeClient.posted), 1)
        finally:
            for key, value in old.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value

    def test_trusted_repository_settings_are_exact_and_non_secret(self):
        settings = LocalSettings(self.tmp.name)
        with self.assertRaises(ValueError):
            settings.save({"default_repository": "Other/Repo", "repositories": {}})
        saved = settings.ensure_repository(self.repo)
        self.assertFalse(saved["repositories"][self.repo.full_name]["interactive_post"])
        self.assertNotIn("token", settings.path.read_text(encoding="utf-8").lower())


class ObservatoryTests(unittest.TestCase):
    def test_health_and_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite3"
            Storage(db).close()
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(db))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                self.assertEqual(json.loads(urlopen(base + "/health").read()), {"status": "ok"})
                dashboard = json.loads(urlopen(base + "/api/dashboard").read())
                self.assertIn("repositories", dashboard)
            finally:
                server.shutdown()
                server.server_close()


class FullWebObservatoryTests(unittest.TestCase):
    def test_non_gitcode_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = WebObservatory(tmp)
            with patch.object(WebObservatory, "install", return_value={"installed": True}), \
                 patch.object(WebObservatory, "_healthy", return_value=True), \
                 patch("issue_agent.web_observatory._port_available", return_value=True), \
                 patch("issue_agent.web_observatory.subprocess.Popen") as process:
                process.return_value.pid = 1234
                process.return_value.poll.return_value = None
                result = manager.start(RepositoryRef.parse("Org/Repo"), open_browser=False)
                self.assertEqual(result["repo"], "Org/Repo")
                self.assertIn("agent-settings", result["settings_url"])

    def test_visualize_opens_setup_without_token(self):
        keys = ("GITCODE_READ_TOKEN", "GITCODE_TOKEN")
        old = {key: os.environ.pop(key, None) for key in keys}
        try:
            with patch.object(WebObservatory, "start", return_value={"running": True}) as start:
                self.assertEqual(main(["visualize", "--repo", "Ascend/MindSpeed-MM", "--no-open"]), 0)
                start.assert_called_once()
        finally:
            for key, value in old.items():
                if value is not None:
                    os.environ[key] = value

    def test_visualize_passes_selected_repository_and_port(self):
        old = os.environ.get("GITCODE_READ_TOKEN")
        os.environ["GITCODE_READ_TOKEN"] = "fixture-only"
        try:
            with patch.object(WebObservatory, "start", return_value={"running": True}) as start:
                self.assertEqual(main(["visualize", "--repo", "Ascend/MindSpeed-MM",
                                       "--port", "5187", "--no-open"]), 0)
                repo = start.call_args.args[0]
                self.assertEqual(repo.full_name, "Ascend/MindSpeed-MM")
                self.assertEqual(start.call_args.kwargs["port"], 5187)
                self.assertFalse(start.call_args.kwargs["open_browser"])
        finally:
            if old is None:
                os.environ.pop("GITCODE_READ_TOKEN", None)
            else:
                os.environ["GITCODE_READ_TOKEN"] = old


if __name__ == "__main__":
    unittest.main()
