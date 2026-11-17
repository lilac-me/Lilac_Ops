from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from issue_agent.gitcode import GitCodeError, MemberSnapshot
from issue_agent.local_settings import LocalSettings
from issue_agent.models import Candidate, Comment, Issue, RepositoryRef
from issue_agent.service import IssueAgent, digest
from issue_agent.storage import Storage


class DeliveryClaimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = RepositoryRef.parse("fixture/repo")
        self.db = Path(self.tmp.name) / "state.sqlite3"
        with Storage(self.db):
            pass
        LocalSettings(self.tmp.name).save({"default_repository": self.repo.full_name,
            "repositories": {self.repo.full_name: {"interactive_post": True, "scheduled_post": True}}})
        self.issue = Issue(1, "open", "example", "details", "reporter", member_hint=False)
        self.candidate = Candidate(self.issue, [], "new_issue", "issue:1", "reporter",
                                   "non_member", "process", "fixture")
        self.body = "approved template"
        self.body_hash = digest(self.body)
        self.key = digest(self.repo.key, "1", "issue:1", self.body_hash)
        self.token_patch = patch("issue_agent.service.write_token", return_value="fixture-only")
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)

    def client(self, fail=False):
        issue = self.issue
        class Client:
            write_token = "fixture-only"
            def __init__(self):
                self.comments = []
                self.posts = 0
            def get_issue(self, number):
                return issue
            def list_comments(self, number):
                return list(self.comments)
            def list_members(self):
                return MemberSnapshot(set(), True, "fixture")
            def post_comment(self, number, body):
                self.posts += 1
                if fail:
                    raise GitCodeError("uncertain result")
                self.comments.append(Comment("remote-1", body, "bot"))
                return {"id": "remote-1"}
        return Client()

    def grant(self, seconds=300):
        with Storage(self.db) as store:
            store.grant_post(self.repo.key, 1, self.body_hash,
                             (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat())

    def run_concurrently(self, scheduled):
        client = self.client()
        barrier = threading.Barrier(2)
        original = Storage.claim_delivery
        def synchronized(store, *args, **kwargs):
            # Force both workers past the non-atomic preliminary reads.
            barrier.wait(timeout=10)
            return original(store, *args, **kwargs)
        def worker():
            with Storage(self.db) as store:
                return IssueAgent(self.tmp.name)._publish(
                    client, self.repo, self.candidate, self.body, store, set(), scheduled=scheduled)
        with patch.object(Storage, "claim_delivery", synchronized):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker) for _ in range(2)]
                results = [future.result(timeout=15) for future in futures]
        self.assertEqual(client.posts, 1)
        self.assertEqual(sum(bool(result.get("deduplicated")) for result in results), 1)
        with Storage(self.db) as store:
            self.assertEqual(len(store.deliveries(self.repo.key, 1)), 1)
            self.assertEqual(store.ledger(self.key)["status"], "confirmed")

    def test_scheduled_workers_publish_once(self):
        self.run_concurrently(scheduled=True)

    def test_interactive_workers_consume_one_grant_and_publish_once(self):
        self.grant()
        self.run_concurrently(scheduled=False)
        with Storage(self.db) as store:
            self.assertFalse(store.consume_post_grant(self.repo.key, 1, self.body_hash))

    def test_missing_or_expired_grant_does_not_leave_a_claim(self):
        with Storage(self.db) as store:
            self.assertFalse(store.claim_delivery(self.key, self.repo.key, 1, self.body_hash, True))
            self.assertEqual(store.ledger(self.key), {})
        self.grant(seconds=-1)
        with Storage(self.db) as store:
            self.assertFalse(store.claim_delivery(self.key, self.repo.key, 1, self.body_hash, True))
            self.assertEqual(store.ledger(self.key), {})
        self.grant()
        with Storage(self.db) as store:
            self.assertTrue(store.claim_delivery(self.key, self.repo.key, 1, self.body_hash, True))

    def test_unresolved_issue_blocks_different_keys_but_not_other_issues(self):
        with Storage(self.db) as store:
            self.assertTrue(store.claim_delivery(self.key, self.repo.key, 1, self.body_hash))
            self.assertFalse(store.claim_delivery("different", self.repo.key, 1, "other-body"))
            self.assertTrue(store.claim_delivery("other-issue", self.repo.key, 2, "other-body"))
            store.put_ledger(self.key, self.repo.key, 1, "pending_delivery", "", self.body_hash)
            self.assertFalse(store.claim_delivery("different", self.repo.key, 1, "other-body"))

    def test_failed_post_leaves_pending_delivery_and_is_not_reposted(self):
        client = self.client(fail=True)
        with Storage(self.db) as store:
            agent = IssueAgent(self.tmp.name)
            for _ in range(2):
                result = agent._publish(client, self.repo, self.candidate, self.body,
                                        store, set(), scheduled=True)
                self.assertEqual(result["status"], "pending_delivery")
            self.assertEqual(client.posts, 1)


if __name__ == "__main__":
    unittest.main()
