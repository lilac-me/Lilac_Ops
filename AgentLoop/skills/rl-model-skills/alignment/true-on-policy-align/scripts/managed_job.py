#!/usr/bin/env python3
"""Launch, inspect, or stop one recorded POSIX process group without broad matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class JobError(ValueError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_proc_stat(raw: str, pid: int) -> tuple[int, int]:
    closing = raw.rfind(")")
    if closing < 0:
        raise JobError(f"malformed /proc/{pid}/stat")
    fields = raw[closing + 2 :].split()
    try:
        return int(fields[2]), int(fields[19])
    except (IndexError, ValueError) as exc:
        raise JobError(f"malformed process identity for PID {pid}") from exc


def proc_identity(pid: int) -> tuple[int, int]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise JobError(f"cannot inspect /proc/{pid}/stat: {exc}") from exc
    return parse_proc_stat(raw, pid)


def proc_start_ticks(pid: int) -> int:
    return proc_identity(pid)[1]


def process_group_members(pgid: int) -> list[dict[str, int]]:
    members: list[dict[str, int]] = []
    if pgid <= 0:
        return members
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise JobError(f"cannot enumerate /proc: {exc}") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            current_pgid, start_ticks = proc_identity(pid)
        except JobError:
            continue
        if current_pgid == pgid:
            members.append({"pid": pid, "start_ticks": start_ticks})
    return sorted(members, key=lambda item: item["pid"])


def compile_smoke_pattern(value: str | None) -> re.Pattern[str] | None:
    if value is None:
        return None
    try:
        return re.compile(value)
    except re.error as exc:
        raise JobError(f"invalid smoke pattern: {exc}") from exc


def load_record(record_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError(f"cannot read job record {record_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JobError("job record root must be an object")
    return value


def verified_live_record(record: dict[str, Any]) -> tuple[int, int]:
    pid = record.get("leader_pid")
    pgid = record.get("process_group_id")
    start_ticks = record.get("process_start_ticks")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (pid, pgid, start_ticks)):
        raise JobError("job record is missing integer PID/PGID/start ticks")
    if pid != pgid:
        raise JobError("refusing group operation because leader_pid != process_group_id")
    current_pgid, current_ticks = proc_identity(pid)
    if current_ticks != start_ticks:
        raise JobError("PID was reused; recorded process identity no longer matches")
    if current_pgid != pgid or os.getpgid(pid) != pgid:
        raise JobError("live process group does not match the recorded PGID")
    return pid, pgid


def command_contains_secret(command: list[str]) -> bool:
    joined = " ".join(command)
    return bool(
        re.search(
            r"(?i)(?:password|passwd|secret|api[_-]?key|(?:access|auth|bearer|hf)[_-]?token)\s*=|https?://[^\s/:@]+:[^\s/@]+@|\bsshpass\b",
            joined,
        )
    )


def tail_contains(path: Path, pattern: re.Pattern[str], offset: int) -> tuple[bool, int, str | None]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            evidence = None
            while True:
                line = handle.readline()
                if not line:
                    break
                if pattern.search(line):
                    evidence = line.strip()[:500]
            return evidence is not None, handle.tell(), evidence
    except OSError:
        return False, offset, None


def launch(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "posix" or not Path("/proc").is_dir():
        raise JobError("managed_job.py requires a POSIX /proc runtime")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise JobError("a command is required after --")
    if command_contains_secret(command):
        raise JobError("refusing to persist a command that appears to contain credentials")
    smoke = compile_smoke_pattern(args.smoke_pattern)
    if args.liveness_seconds < 0 or args.smoke_timeout <= 0:
        raise JobError("liveness_seconds must be non-negative and smoke_timeout positive")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "job.json"
    if record_path.exists():
        raise JobError(f"job record already exists: {record_path}")
    log_path = (args.log.resolve() if args.log else run_dir / "stdout.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise JobError(f"refusing to append to an existing log: {log_path}")
    working_directory = args.cwd.resolve() if args.cwd else Path.cwd().resolve()
    if not working_directory.is_dir():
        raise JobError(f"working directory does not exist: {working_directory}")
    log_handle = log_path.open("x", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    pid = process.pid
    pgid = os.getpgid(pid)
    record = {
        "schema_version": 1,
        "created_at": now(),
        "status": "starting",
        "leader_pid": pid,
        "process_group_id": pgid,
        "process_start_ticks": proc_start_ticks(pid),
        "launch_group_members": process_group_members(pgid),
        "command": command,
        "command_sha256": "sha256:"
        + hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "working_directory": str(working_directory),
        "log": str(log_path),
        "smoke_pattern": args.smoke_pattern,
        "smoke_evidence": None,
        "exit_code": None,
    }
    atomic_json(record_path, record)

    deadline = time.monotonic() + args.smoke_timeout
    live_deadline = time.monotonic() + args.liveness_seconds
    offset = 0
    smoke_seen = smoke is None
    evidence = None
    while time.monotonic() < deadline:
        code = process.poll()
        if smoke is not None:
            found, offset, found_line = tail_contains(log_path, smoke, offset)
            if found:
                smoke_seen = True
                evidence = found_line
        if code is not None:
            record.update(status="failed", exit_code=code, smoke_evidence=evidence)
            atomic_json(record_path, record)
            raise JobError(f"job exited before launch gate, exit_code={code}")
        if smoke_seen and time.monotonic() >= live_deadline:
            record.update(status="running", smoke_evidence=evidence)
            atomic_json(record_path, record)
            return record
        time.sleep(0.25)
    record.update(status="smoke_timeout", smoke_evidence=evidence)
    atomic_json(record_path, record)
    raise JobError("job stayed live but did not satisfy the smoke/liveness gate")


def status(record_path: Path) -> dict[str, Any]:
    record = load_record(record_path)
    recorded_status = record.get("status")
    pgid = record.get("process_group_id")
    members: list[dict[str, int]] = []
    identity_error = None
    try:
        verified_live_record(record)
        live = True
        runtime_status = "running"
        members = process_group_members(pgid)
    except JobError as exc:
        live = False
        identity_error = str(exc)
        if isinstance(pgid, int) and not isinstance(pgid, bool):
            members = process_group_members(pgid)
        runtime_status = "orphaned_group_unverified" if members else "not_running"
    return {
        **record,
        "recorded_status": recorded_status,
        "status": runtime_status,
        "live_identity_verified": live,
        "process_group_members": members,
        "identity_error": identity_error,
        "observed_exit_code": (
            record.get("exit_code")
            if isinstance(record.get("exit_code"), int)
            else "unknown"
        ),
        "checked_at": now(),
    }


def wait_for_group_empty(pgid: int, deadline: float) -> list[dict[str, int]]:
    members = process_group_members(pgid)
    while members and time.monotonic() < deadline:
        time.sleep(0.25)
        members = process_group_members(pgid)
    return members


def stop(record_path: Path, grace_seconds: float, force_kill: bool) -> dict[str, Any]:
    if grace_seconds < 0:
        raise JobError("grace_seconds must be non-negative")
    record = load_record(record_path)
    pid, pgid = verified_live_record(record)
    initial_members = process_group_members(pgid)
    if not any(member["pid"] == pid for member in initial_members):
        raise JobError("recorded leader is not present in the recorded process group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    remaining = wait_for_group_empty(pgid, time.monotonic() + grace_seconds)
    if not remaining:
        record.update(
            status="stopped",
            stopped_at=now(),
            stop_signal="TERM",
            stopped_group_member_count=len(initial_members),
        )
        atomic_json(record_path, record)
        return record
    if not force_kill:
        raise JobError(
            "recorded group still has members after the grace period; "
            "rerun with --force-kill if approved"
        )
    initial_identities = {
        (member["pid"], member["start_ticks"]) for member in initial_members
    }
    current_identities = {
        (member["pid"], member["start_ticks"]) for member in remaining
    }
    if not current_identities.issubset(initial_identities):
        raise JobError(
            "refusing force kill because process-group membership changed after TERM"
        )
    leader_still_verified = False
    try:
        verified_live_record(record)
        leader_still_verified = True
    except JobError:
        leader_still_verified = False
    if leader_still_verified:
        os.killpg(pgid, signal.SIGKILL)
    else:
        for member in remaining:
            current_pgid, current_ticks = proc_identity(member["pid"])
            if current_pgid != pgid or current_ticks != member["start_ticks"]:
                raise JobError(
                    f"refusing force kill because PID {member['pid']} identity changed"
                )
            os.kill(member["pid"], signal.SIGKILL)
    remaining = wait_for_group_empty(pgid, time.monotonic() + max(5.0, grace_seconds))
    if remaining:
        raise JobError("SIGKILL was sent but recorded process-group members remain")
    record.update(
        status="killed",
        stopped_at=now(),
        stop_signal="KILL",
        stopped_group_member_count=len(initial_members),
    )
    atomic_json(record_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-dir", type=Path, required=True)
    start.add_argument("--log", type=Path)
    start.add_argument("--cwd", type=Path)
    start.add_argument("--smoke-pattern")
    start.add_argument("--liveness-seconds", type=float, default=10.0)
    start.add_argument("--smoke-timeout", type=float, default=120.0)
    start.add_argument("command", nargs=argparse.REMAINDER)
    inspect = subparsers.add_parser("status")
    inspect.add_argument("--record", type=Path, required=True)
    terminate = subparsers.add_parser("stop")
    terminate.add_argument("--record", type=Path, required=True)
    terminate.add_argument("--grace-seconds", type=float, default=30.0)
    terminate.add_argument("--force-kill", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "start":
            result = launch(args)
        elif args.action == "status":
            result = status(args.record.resolve())
        else:
            result = stop(args.record.resolve(), args.grace_seconds, args.force_kill)
    except (JobError, OSError, re.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
