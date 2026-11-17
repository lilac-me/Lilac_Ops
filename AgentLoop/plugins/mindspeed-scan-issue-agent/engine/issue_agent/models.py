from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class RepositoryRef:
    provider: str
    host: str
    owner: str
    repo: str
    code_url: str
    default_branch: str = ""

    @classmethod
    def parse(cls, value: str, code_url: str = "", default_branch: str = "") -> "RepositoryRef":
        value = value.strip().rstrip("/")
        host = "gitcode.com"
        if "://" in value:
            parsed = urlparse(value)
            host = parsed.hostname or host
            parts = [p for p in parsed.path.strip("/").split("/") if p]
        else:
            parts = [p for p in value.split("/") if p]
        if len(parts) != 2:
            raise ValueError("repository must be owner/repo or a two-segment GitCode URL")
        owner, repo = parts
        if repo.endswith(".git"):
            repo = repo[:-4]
        if not owner or not repo:
            raise ValueError("repository owner and name must be non-empty")
        return cls(
            provider="gitcode",
            host=host.lower(),
            owner=owner,
            repo=repo,
            code_url=code_url or f"https://{host}/{owner}/{repo}.git",
            default_branch=default_branch,
        )

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.host}:{self.owner}:{self.repo}".lower()

    @property
    def slug(self) -> str:
        return f"{self.host}--{self.owner}--{self.repo}".replace("/", "--")


@dataclass
class Comment:
    id: str
    body: str
    author: str
    created_at: str = ""
    member_hint: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Issue:
    number: int
    state: str
    title: str
    body: str
    author: str
    updated_at: str = ""
    created_at: str = ""
    html_url: str = ""
    comments_count: int = 0
    labels: list[str] = field(default_factory=list)
    member_hint: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Candidate:
    issue: Issue
    comments: list[Comment]
    trigger: str
    trigger_comment_id: str
    actor: str
    member_state: str
    action: str
    reason: str

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["issue"].pop("raw", None)
        for comment in data["comments"]:
            comment.pop("raw", None)
        return data


def data_root(base: str | Path, repo: RepositoryRef) -> Path:
    return Path(base).expanduser().resolve() / "repos" / repo.slug
