from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .models import Issue, RepositoryRef


TEXT_GLOBS = ["*.py", "*.go", "*.rs", "*.java", "*.js", "*.ts", "*.tsx", "*.cpp", "*.cc", "*.c", "*.h", "*.md", "*.yaml", "*.yml", "*.json", "*.toml"]


class RepositoryWorkspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _git(path: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=180).stdout.strip()

    def prepare(self, repo: RepositoryRef, local_path: str = "") -> tuple[Path, str, list[str]]:
        notes: list[str] = []
        path = Path(local_path).expanduser().resolve() if local_path else self.root / repo.slug
        if local_path:
            if not (path / ".git").exists():
                notes.append("local workspace is not a Git checkout")
        else:
            branch = repo.default_branch
            if branch:
                subprocess.run(["git", "check-ref-format", f"refs/heads/{branch}"], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            if not path.exists():
                command = ["git", "clone", "--filter=blob:none"]
                if branch:
                    command.extend(["--branch", branch])
                subprocess.run([*command, "--", repo.code_url, str(path)], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            else:
                if self._git(path, "status", "--porcelain"):
                    raise ValueError("managed code cache has local changes; refusing to update it")
                if self._git(path, "remote", "get-url", "origin") != repo.code_url:
                    raise ValueError("managed code cache origin does not match the requested repository")
                if not branch:
                    remote_head = self._git(path, "ls-remote", "--symref", "origin", "HEAD")
                    for line in remote_head.splitlines():
                        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
                            branch = line.split("\t", 1)[0][len("ref: refs/heads/"):]
                            break
                    if not branch:
                        raise ValueError("cannot resolve the remote default branch; specify --default-branch")
                # Fetch the requested branch explicitly, including when it differs
                # from the branch originally cloned into this managed cache.
                self._git(path, "fetch", "origin", f"refs/heads/{branch}")
                self._git(path, "checkout", "--detach", "FETCH_HEAD")
        commit = "unknown"
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode == 0:
            commit = result.stdout.strip()
        return path, commit, notes


def search_terms(issue: Issue) -> list[str]:
    text = issue.title + "\n" + issue.body
    terms: list[str] = []
    patterns = [
        r"\b[A-Za-z_][A-Za-z0-9_]{3,}\.(?:py|go|rs|java|js|ts|cpp|cc|c|h)\b",
        r"\b[A-Za-z_][A-Za-z0-9_]{3,}(?:Error|Exception|Failure)\b",
        r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_-]{3,}(?![A-Za-z0-9_])",
    ]
    stop = {"issue", "error", "feature", "skill", "please", "when", "with", "from", "this", "that", "have", "does", "using"}
    for pattern in patterns:
        for term in re.findall(pattern, text):
            if term.lower() not in stop and term not in terms:
                terms.append(term)
            if len(terms) >= 8:
                return terms
    for phrase in re.findall(r"[\u4e00-\u9fff]{4,16}", text):
        for start in range(0, len(phrase), 6):
            term = phrase[start:start + 6]
            if len(term) >= 4 and term not in terms:
                terms.append(term)
            if len(terms) >= 8:
                return terms
    return terms


def analyze_issue(repo: RepositoryRef, issue: Issue, workspace_root: str | Path,
                  local_path: str = "") -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "repo": repo.full_name,
        "issue": issue.number,
        "code_url": repo.code_url,
        "commit": "unknown",
        "terms": [],
        "matches": [],
        "verdict": "insufficient_evidence",
        "reproduction": {"status": "not_run", "command": ""},
        "limitations": [],
    }
    try:
        path, commit, notes = RepositoryWorkspace(workspace_root).prepare(repo, local_path)
        evidence["commit"] = commit
        evidence["limitations"].extend(notes)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        evidence["limitations"].append(f"repository preparation failed: {exc}")
        return evidence

    terms = search_terms(issue)
    evidence["terms"] = terms
    rg = shutil.which("rg")
    if not rg:
        evidence["limitations"].append("ripgrep is unavailable")
        return evidence
    for term in terms:
        command = [rg, "-n", "-F", "--max-count", "3"]
        for glob in TEXT_GLOBS:
            command.extend(["--glob", glob])
        command.extend(["--glob", "!node_modules/**", "--glob", "!.git/**",
                        "--glob", "!.venv/**", "--glob", "!vendor/**"])
        command.extend([term, "."])
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", timeout=30, cwd=path)
        for line in result.stdout.splitlines()[:3]:
            try:
                file_path, line_no, snippet = line.split(":", 2)
                relative = str(Path(file_path))
            except (ValueError, OSError):
                continue
            evidence["matches"].append({"term": term, "path": relative.replace("\\", "/"),
                                        "line": int(line_no), "snippet": snippet.strip()[:300]})
        if len(evidence["matches"]) >= 20:
            break
    if evidence["matches"]:
        evidence["limitations"].append("code matches are navigation evidence, not proof of root cause")
    else:
        evidence["limitations"].append("no high-signal issue terms were found in the checked source")
    return evidence


def template_reply(issue: Issue, follow_up: bool) -> str:
    if follow_up:
        return "感谢补充信息，我们已记录本次更新并会继续核查；有进展会在本 Issue 中同步。"
    title = issue.title.strip().replace("\n", " ")[:80]
    return (f"您好，感谢您提交 Issue。我们已收到关于「{title}」的信息，将结合仓库代码和当前版本进一步核查；"
            "有进展会在本 Issue 中同步。")


def shadow_draft(issue: Issue, evidence: dict[str, Any]) -> str:
    matches = evidence.get("matches") or []
    if not matches:
        return ("已完成当前仓库快照的初步检索，但现有信息不足以确认根因。建议补充完整报错、版本/commit、"
                "最小复现步骤及相关配置。该草稿仅供维护者审阅，尚未发布。")
    refs = "、".join(f"{m['path']}:{m['line']}" for m in matches[:5])
    return (f"已在 commit {evidence.get('commit', 'unknown')} 中定位到可能相关的代码位置：{refs}。"
            "这些匹配仅用于导航，尚不能单独证明根因；建议维护者结合调用链和最小复现继续确认。该草稿尚未发布。")
