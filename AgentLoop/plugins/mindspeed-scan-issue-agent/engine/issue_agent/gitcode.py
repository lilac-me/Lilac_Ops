from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Comment, Issue, RepositoryRef


class GitCodeError(RuntimeError):
    pass


@dataclass
class MemberSnapshot:
    members: set[str]
    complete: bool
    source: str
    error: str = ""


class GitCodeClient:
    def __init__(self, repo: RepositoryRef, read_token: str = "", write_token: str = "", timeout: int = 30):
        self.repo = repo
        self.read_token = read_token
        self.write_token = write_token
        self.timeout = timeout
        self.base = "https://api.gitcode.com/api/v5"

    def _request(self, method: str, path: str, query: dict[str, Any] | None = None,
                 body: dict[str, Any] | None = None, write: bool = False) -> Any:
        params = {k: v for k, v in (query or {}).items() if v is not None and v != ""}
        token = self.write_token if write else self.read_token
        if token:
            params["access_token"] = token
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode(params)
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=payload, method=method,
                          headers={"Accept": "application/json", "Content-Type": "application/json",
                                   "User-Agent": "mindspeed-scan-issue-agent/0.2"})
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise GitCodeError(f"GitCode HTTP {exc.code}: {detail}") from exc
                last_error = exc
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == 2:
                    break
            time.sleep(0.5 * (2 ** attempt))
        raise GitCodeError(f"GitCode request failed: {last_error}")

    def _paginate(self, path: str, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for page in range(1, 1001):
            batch = self._request("GET", path, {**(query or {}), "page": page, "per_page": 100})
            if not isinstance(batch, list):
                raise GitCodeError(f"expected a list from {path}")
            for row in batch:
                key = str(row.get("id") or row.get("number") or row.get("iid") or "")
                if key and key not in seen:
                    seen.add(key)
                    result.append(row)
            if len(batch) < 100:
                break
        return result
    @staticmethod
    def _login(user: Any) -> str:
        return str((user or {}).get("login") or (user or {}).get("username") or "").strip()

    @staticmethod
    def _member_hint(user: Any, row: dict[str, Any]) -> bool | None:
        for value in (row.get("is_member"), (user or {}).get("is_member")):
            if isinstance(value, bool):
                return value
        role = (user or {}).get("member_role") or row.get("member_role")
        return True if role else None

    def list_issues(self, state: str = "open") -> list[Issue]:
        path = f"/repos/{self.repo.owner}/{self.repo.repo}/issues"
        return [self._issue(row) for row in self._paginate(path, {"state": state, "sort": "updated", "direction": "desc"})]

    def get_issue(self, number: int) -> Issue:
        path = f"/repos/{self.repo.owner}/{self.repo.repo}/issues/{number}"
        row = self._request("GET", path)
        if not isinstance(row, dict):
            raise GitCodeError("invalid issue response")
        return self._issue(row)

    def _issue(self, row: dict[str, Any]) -> Issue:
        user = row.get("user") or row.get("author") or {}
        labels = []
        for label in row.get("labels") or []:
            labels.append(str(label.get("name") if isinstance(label, dict) else label))
        return Issue(
            number=int(row.get("number") or row.get("iid") or 0),
            state=str(row.get("state") or row.get("issue_state") or "unknown"),
            title=str(row.get("title") or ""),
            body=str(row.get("body") or row.get("description") or ""),
            author=self._login(user),
            updated_at=str(row.get("updated_at") or ""),
            created_at=str(row.get("created_at") or ""),
            html_url=str(row.get("html_url") or ""),
            comments_count=int(row.get("comments") or row.get("comments_count") or 0),
            labels=labels,
            member_hint=self._member_hint(user, row),
            raw=row,
        )

    def list_comments(self, number: int) -> list[Comment]:
        path = f"/repos/{self.repo.owner}/{self.repo.repo}/issues/{number}/comments"
        rows = self._paginate(path, {"order": "asc"})
        comments = []
        for row in rows:
            user = row.get("user") or row.get("author") or {}
            comments.append(Comment(
                id=str(row.get("id") or ""),
                body=str(row.get("body") or ""),
                author=self._login(user),
                created_at=str(row.get("created_at") or ""),
                member_hint=self._member_hint(user, row),
                raw=row,
            ))
        return comments

    def list_members(self) -> MemberSnapshot:
        if not self.read_token:
            return MemberSnapshot(set(), False, "collaborators", "GitCode read token is not configured")
        path = f"/repos/{self.repo.owner}/{self.repo.repo}/collaborators"
        try:
            rows = self._paginate(path)
            return MemberSnapshot({self._login(row).lower() for row in rows if self._login(row)}, True, "collaborators")
        except GitCodeError as exc:
            return MemberSnapshot(set(), False, "collaborators", str(exc))

    def post_comment(self, number: int, body: str) -> dict[str, Any]:
        if not self.write_token:
            raise GitCodeError("GITCODE_WRITE_TOKEN is not configured")
        path = f"/repos/{self.repo.owner}/{self.repo.repo}/issues/{number}/comments"
        result = self._request("POST", path, body={"body": body}, write=True)
        if not isinstance(result, dict):
            raise GitCodeError("invalid comment creation response")
        return result
