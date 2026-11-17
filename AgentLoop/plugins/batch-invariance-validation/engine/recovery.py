"""Audit evidence-driven repair attempts and enforce bounded recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ISSUE_LIMIT = 5
STAGE_LIMIT = 10
RUN_LIMIT = 20
LEDGER = "recovery_attempts.jsonl"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_key(state: str, reason_code: str, failure_signature: str,
                 config_hash: str) -> str:
    return stable_hash({"state": state, "reason_code": reason_code,
                        "failure_signature": failure_signature})


def read_attempts(workspace: str | Path) -> list[dict[str, Any]]:
    path = Path(workspace) / LEDGER
    if not path.is_file():
        return []
    attempts = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid recovery ledger line {number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"invalid recovery ledger line {number}: object required")
        attempts.append(item)
    return attempts


def validate_attempt(attempt: dict[str, Any], history: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = ("recovery_key", "round", "stage", "state", "reason_code",
                "observations", "root_cause_hypotheses", "selected_hypothesis",
                "planned_actions", "commands", "verification", "reflection",
                "progress_assessment", "repair_result", "before_evidence_sha256",
                "after_evidence_sha256")
    for key in required:
        if key not in attempt or attempt[key] in (None, "", []):
            errors.append(f"recovery attempt requires non-empty {key}")
    key = str(attempt.get("recovery_key", ""))
    issue_history = [item for item in history if item.get("recovery_key") == key]
    stage_history = [item for item in history if item.get("stage") == attempt.get("stage")]
    expected_round = len(issue_history) + 1
    if attempt.get("round") != expected_round:
        errors.append(f"recovery round must be {expected_round}")
    if expected_round > ISSUE_LIMIT:
        errors.append("BLOCKED_RECOVERY_EXHAUSTED: issue limit reached")
    if len(stage_history) >= STAGE_LIMIT:
        errors.append("BLOCKED_RECOVERY_EXHAUSTED: stage limit reached")
    if len(history) >= RUN_LIMIT:
        errors.append("BLOCKED_RECOVERY_EXHAUSTED: run limit reached")
    action_hash = stable_hash({"actions": attempt.get("planned_actions", []),
                               "commands": attempt.get("commands", []),
                               "before": attempt.get("before_evidence_sha256")})
    attempt["action_hash"] = action_hash
    if any(item.get("action_hash") == action_hash for item in issue_history):
        errors.append("mechanical retry rejected: actions, commands and evidence are unchanged")
    if attempt.get("progress_assessment") not in {"PROGRESSED", "NO_PROGRESS"}:
        errors.append("progress_assessment must be PROGRESSED or NO_PROGRESS")
    if attempt.get("repair_result") not in {"PASS", "BLOCKED"}:
        errors.append("repair_result must be PASS or BLOCKED")
    return errors


def record_attempt(workspace: str | Path, attempt: dict[str, Any]) -> dict[str, Any]:
    root = Path(workspace)
    history = read_attempts(root)
    candidate = dict(attempt)
    candidate.setdefault("recorded_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    schema_path = Path(__file__).resolve().parent.parent / "contracts" / "recovery_attempt.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = [error.message for error in Draft202012Validator(schema).iter_errors(candidate)]
    errors = schema_errors + validate_attempt(candidate, history)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    return candidate


def write_exhausted_report(workspace: str | Path, key: str) -> Path:
    """Explain all model repair rounds without hiding commands or evidence."""
    root = Path(workspace)
    attempts = [item for item in read_attempts(root) if item.get("recovery_key") == key]
    if not attempts:
        raise ValueError("cannot report recovery exhaustion without attempts")
    lines = ["# Recovery Exhausted", "",
             f"- 原因码：`BLOCKED_RECOVERY_EXHAUSTED`",
             f"- 问题指纹：`{key}`", f"- 已执行轮数：`{len(attempts)}`", ""]
    for item in attempts:
        lines.extend([
            f"## 第 {item['round']} 轮", "",
            f"- 记录时间：`{item.get('recorded_utc', 'unavailable')}`",
            f"- 根因假设：{item.get('selected_hypothesis')}",
            f"- 修复动作：`{json.dumps(item.get('planned_actions'), ensure_ascii=False)}`",
            f"- 执行命令：`{json.dumps(item.get('commands'), ensure_ascii=False)}`",
            f"- 验证结果：`{item.get('repair_result')}`",
            f"- 进展判断：`{item.get('progress_assessment')}`",
            f"- 反思：{item.get('reflection')}",
            f"- 修复前证据：`{item.get('before_evidence_sha256')}`",
            f"- 修复后证据：`{item.get('after_evidence_sha256')}`", "",
        ])
    lines.extend(["## 用户最小介入", "",
                  "请依据最后一轮反思补充外部权限、硬件资源或超出自动修复边界的决策；恢复时沿用问题指纹和已归档证据。", ""])
    path = root / "recovery_exhausted.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def recovery_status(workspace: str | Path, key: str, stage: str) -> dict[str, Any]:
    history = read_attempts(workspace)
    issue_count = sum(item.get("recovery_key") == key for item in history)
    stage_count = sum(item.get("stage") == stage for item in history)
    exhausted = issue_count >= ISSUE_LIMIT or stage_count >= STAGE_LIMIT or len(history) >= RUN_LIMIT
    return {"issue_attempts": issue_count, "stage_attempts": stage_count,
            "run_attempts": len(history), "max_issue_attempts": ISSUE_LIMIT,
            "max_stage_attempts": STAGE_LIMIT, "max_run_attempts": RUN_LIMIT,
            "exhausted": exhausted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--workspace", required=True, type=Path)
    record.add_argument("--attempt", required=True, type=Path)
    status = sub.add_parser("status")
    status.add_argument("--workspace", required=True, type=Path)
    status.add_argument("--recovery-key", required=True)
    status.add_argument("--stage", required=True)
    args = parser.parse_args()
    if args.command == "record":
        attempt = json.loads(args.attempt.read_text(encoding="utf-8"))
        result = record_attempt(args.workspace, attempt)
    else:
        result = recovery_status(args.workspace, args.recovery_key, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
