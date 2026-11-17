import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from issue_agent.analyzer import RepositoryWorkspace, analyze_issue
from issue_agent.models import Issue, RepositoryRef


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream"
        self.source.mkdir()
        self.git(self.source, "init", "-b", "main")
        self.commit("main-v1")
        self.git(self.source, "checkout", "-b", "release")
        self.commit("release-v1")
        self.git(self.source, "checkout", "main")
        self.workspace = RepositoryWorkspace(self.root / "cache")

    @staticmethod
    def git(path, *args):
        env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
               "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
               "GIT_TERMINAL_PROMPT": "0"}
        return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                              text=True, env=env, timeout=30).stdout.strip()

    def commit(self, text):
        (self.source / "example.py").write_text(text + "\n", encoding="utf-8")
        self.git(self.source, "add", "example.py")
        self.git(self.source, "commit", "-m", text)
        return self.git(self.source, "rev-parse", "HEAD")

    def repo(self, branch=""):
        return RepositoryRef.parse("fixture/repo", code_url=str(self.source), default_branch=branch)

    def test_first_clone_honors_selected_branch(self):
        path, commit, notes = self.workspace.prepare(self.repo("release"))
        self.assertEqual(commit, self.git(self.source, "rev-parse", "release"))
        self.assertEqual((path / "example.py").read_text(), "release-v1\n")
        self.assertEqual(notes, [])

    def test_cached_checkout_advances_to_latest_remote_commit(self):
        _, first, _ = self.workspace.prepare(self.repo())
        latest = self.commit("main-v2")
        path, current, notes = self.workspace.prepare(self.repo())
        self.assertNotEqual(first, current)
        self.assertEqual(current, latest)
        self.assertEqual((path / "example.py").read_text(), "main-v2\n")
        self.assertEqual(notes, [])

    def test_existing_cache_can_switch_to_selected_branch(self):
        self.workspace.prepare(self.repo())
        path, current, _ = self.workspace.prepare(self.repo("release"))
        self.assertEqual(current, self.git(self.source, "rev-parse", "release"))
        self.assertEqual((path / "example.py").read_text(), "release-v1\n")

    def test_remote_default_branch_change_is_honored(self):
        self.workspace.prepare(self.repo())
        self.git(self.source, "checkout", "release")
        _, current, _ = self.workspace.prepare(self.repo())
        self.assertEqual(current, self.git(self.source, "rev-parse", "release"))

    def test_user_workspace_is_not_switched_or_modified(self):
        head = self.git(self.source, "rev-parse", "HEAD")
        (self.source / "example.py").write_text("user changes\n")
        _, current, _ = self.workspace.prepare(self.repo("release"), str(self.source))
        self.assertEqual(current, head)
        self.assertEqual(self.git(self.source, "branch", "--show-current"), "main")
        self.assertEqual((self.source / "example.py").read_text(), "user changes\n")

    def test_dirty_managed_cache_is_preserved(self):
        path, _, _ = self.workspace.prepare(self.repo())
        (path / "example.py").write_text("local changes\n")
        with self.assertRaisesRegex(ValueError, "local changes"):
            self.workspace.prepare(self.repo("release"))
        self.assertEqual((path / "example.py").read_text(), "local changes\n")

    def test_missing_branch_does_not_silently_search_old_code(self):
        self.workspace.prepare(self.repo())
        issue = Issue(1, "open", "example.py failure", "details", "reporter")
        evidence = analyze_issue(self.repo("missing"), issue, self.root / "cache")
        self.assertEqual(evidence["commit"], "unknown")
        self.assertEqual(evidence["matches"], [])
        self.assertIn("repository preparation failed", evidence["limitations"][0])


if __name__ == "__main__":
    unittest.main()
