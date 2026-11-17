"""Ledger: append-only event and issue stores for doc-agent.

Layout (per workspace):
  .doc-work/<repo>/<run-id>/run.json       # isolated run metadata
  .doc-work/<repo>/<run-id>/ledger.jsonl   # phase/gate/checkpoint events
  .doc-work/<repo>/<run-id>/issues.jsonl   # append-only issue lifecycle records

Both JSONL files are append-only. Iteration counts and review history are
derived from the ledger; issue reads project the latest lifecycle record per
rule/location identity without rewriting history.
"""

from __future__ import annotations

import json
import pathlib
import re
import uuid
from datetime import datetime, timezone

LEDGER = "ledger.jsonl"
ISSUES = "issues.jsonl"
RUN_META = "run.json"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ISSUE_STATUSES = {"open", "fixed", "verified", "closed", "disputed"}


class LedgerCorruptionError(ValueError):
    """Raised when an append-only JSONL ledger cannot be trusted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: pathlib.Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerCorruptionError(
                    f"{path}: invalid JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise LedgerCorruptionError(
                    f"{path}: record at line {line_number} must be an object"
                )
            records.append(record)
    return records


def workspace_dir(base: pathlib.Path, repo: str) -> pathlib.Path:
    return base / repo


def validate_run_id(run_id: str) -> str:
    """Validate a run identifier before it is used as a path component."""
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain 1-64 ASCII letters, digits, '.', '_' or '-' "
            "and must start with a letter or digit"
        )
    return run_id


def new_run(base: pathlib.Path, run_id: str | None = None,
            metadata: dict | None = None) -> pathlib.Path:
    """Create an isolated run directory and return its path.

    A run is deliberately a directory, not a mutable flag in the parent
    workspace.  This keeps concurrent reviews of the same repository from
    sharing ledger, issue and report files.
    """
    base = pathlib.Path(base)
    run_id = validate_run_id(run_id) if run_id else uuid.uuid4().hex[:12]
    run = base / run_id
    if run.exists():
        raise FileExistsError(f"run already exists: {run}")
    run.mkdir(parents=True)
    info = {"run_id": run_id, "repo": base.name, "created_at": _now()}
    if metadata:
        info.update(metadata)
    (run / RUN_META).write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run


def run_metadata(workspace: pathlib.Path) -> dict:
    """Read isolated-run metadata, returning an empty mapping if absent."""
    path = pathlib.Path(workspace) / RUN_META
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def append_event(base: pathlib.Path, repo: str, event: dict) -> dict:
    """Append a memory event (see contracts/memory_event.json)."""
    record = dict(event)
    record.setdefault("ts", _now())
    return _append(workspace_dir(base, repo) / LEDGER, record)


def read_events(base: pathlib.Path, repo: str) -> list[dict]:
    return _read(workspace_dir(base, repo) / LEDGER)


def iter_count(base: pathlib.Path, repo: str, state: str) -> int:
    """Count phase_enter events for a state."""
    n = 0
    for ev in read_events(base, repo):
        if ev.get("type") == "phase_enter" and ev.get("phase") == state:
            n += 1
    return n


def append_issue(base: pathlib.Path, repo: str, issue: dict) -> dict:
    """Append an issue record without rewriting history.

    ``read_issues`` collapses records by the stable rule/location identity, so
    a later lifecycle record supersedes an older open record while the raw
    JSONL remains append-only and auditable.  The scorer counts only the
    latest ``open`` projection.
    """
    record = dict(issue)
    record.setdefault("ts", _now())
    path = workspace_dir(base, repo) / ISSUES
    _append(path, record)
    return record


def issue_key(issue: dict) -> tuple:
    """Return the stable identity used for issue lifecycle records."""
    location = issue.get("location") or {}
    line = location.get("line")
    anchor = location.get("anchor")
    return (issue.get("rule_id"), location.get("file"), line, anchor)


def read_issues(base: pathlib.Path, repo: str) -> list[dict]:
    latest: dict[tuple, dict] = {}
    order: list[tuple] = []
    for issue in _read(workspace_dir(base, repo) / ISSUES):
        key = issue_key(issue)
        if key not in latest:
            order.append(key)
        latest[key] = issue
    return [latest[key] for key in order]


def update_issue_status(base: pathlib.Path, repo: str, rule_id: str,
                        file: str, status: str, line: int | None = None,
                        anchor: str | None = None, **updates) -> dict:
    """Append a lifecycle transition for the latest matching issue."""
    if status not in ISSUE_STATUSES:
        raise ValueError(f"unsupported issue status: {status!r}")
    candidates = read_issues(base, repo)
    key = (rule_id, file, line, anchor)
    current = next((item for item in candidates if issue_key(item) == key), None)
    if current is None:
        raise KeyError(f"issue not found: {key!r}")
    record = dict(current)
    record.update(updates)
    record["status"] = status
    record["ts"] = _now()
    return append_issue(base, repo, record)


def open_issues(base: pathlib.Path, repo: str) -> list[dict]:
    return [i for i in read_issues(base, repo) if i.get("status") == "open"]


def current_state(base: pathlib.Path, repo: str, initial: str = "init") -> str:
    state = initial
    for ev in read_events(base, repo):
        if ev.get("type") == "phase_enter" and ev.get("phase"):
            state = ev["phase"]
    return state
