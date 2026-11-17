"""Doc-agent state machine CLI.

Commands:
  next     --workspace W --yaml Y [--handoff H | --handoff-file F]
  current  --workspace W --yaml Y
  verify   --workspace W --yaml Y
  score    --workspace W --rules R [--depth L0|L1|L2]
  doc-hash --workspace W [--doc DOC]

State transitions are declared in doc_pipeline.yaml (see yamlite.py for
the supported subset). All transitions are logged to ledger.jsonl
(append-only); verify replays the ledger against the declaration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

# Script-mode bootstrap: the plugin directory name contains a hyphen and
# cannot be a Python package name, so sibling modules are imported directly
# with the engine directory on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gates  # noqa: E402
import ledger  # noqa: E402
import yamlite  # noqa: E402

HANDOFF_RE = re.compile(r"^\s*handoff:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_YAML = PLUGIN_ROOT / "engine" / "doc_pipeline.yaml"
DEFAULT_RULES = PLUGIN_ROOT / "rules" / "rules.yaml"


def extract_handoff(text: str) -> str | None:
    matches = HANDOFF_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def load_pipeline(yaml_path: pathlib.Path) -> dict:
    data = yamlite.load(yaml_path.read_text(encoding="utf-8"))
    states = data.get("states", [])
    index = {}
    for st in states:
        index[st["name"]] = st
    return index


def _resolve_workspace(args, ensure_run: bool = False) -> pathlib.Path:
    """Resolve an optional run id and reject ambiguous shared workspaces.

    Direct temporary workspaces remain supported for library-style callers.
    Under the documented ``.doc-work`` root, however, every engine command
    must target a run created by ``new`` (or pass ``--run-id`` explicitly).
    """
    workspace = pathlib.Path(args.workspace).resolve()
    run_id = getattr(args, "run_id", None)
    if not run_id:
        if any(part.lower() == ".doc-work" for part in workspace.parts) \
                and not (workspace / ledger.RUN_META).is_file():
            raise ValueError(
                "shared .doc-work workspace is not a run; create one with "
                "the `new` command or pass --run-id"
            )
        return workspace
    try:
        ledger.validate_run_id(run_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    workspace = workspace / run_id
    if ensure_run and not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    if ensure_run:
        meta = workspace / ledger.RUN_META
        if not meta.exists():
            meta.write_text(json.dumps({
                "run_id": run_id,
                "repo": pathlib.Path(args.workspace).resolve().name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return workspace


def _workspace_repo(workspace: pathlib.Path) -> str:
    """Return the repository name, not the isolated run id."""
    metadata = ledger.run_metadata(workspace)
    if metadata.get("repo"):
        return str(metadata["repo"])
    if (workspace / ledger.RUN_META).is_file() and workspace.parent.name:
        return workspace.parent.name
    return workspace.name


def _workspace_doc(workspace: pathlib.Path, explicit_doc: str | None = None) -> str:
    """Return the immutable target document recorded for this score run."""
    metadata = ledger.run_metadata(workspace)
    doc = metadata.get("doc")
    if doc is not None and str(doc).strip():
        return str(doc).strip()
    return str(explicit_doc).strip() if explicit_doc and str(explicit_doc).strip() else ""


def _cmd_new(args) -> int:
    base = pathlib.Path(args.workspace).resolve()
    metadata = {"repo": base.name}
    if args.doc:
        metadata["doc"] = args.doc
    try:
        run = ledger.new_run(base, args.run_id, metadata)
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: cannot create run: {exc}", file=sys.stderr)
        return 2
    print(f"WORKSPACE: {run}")
    print(f"RUN_ID: {run.name}")
    return 0


def _cmd_issue_status(args) -> int:
    try:
        workspace = _resolve_workspace(args)
        record = ledger.update_issue_status(
            workspace, "", args.rule_id, args.file, args.status,
            line=args.line, anchor=args.anchor,
        )
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: cannot update issue: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=False))
    return 0


def _when_matches(cond: str, handoff: str | None) -> bool:
    if cond == "always":
        return True
    if cond.startswith("handoff:"):
        token = cond.partition(":")[2]
        return handoff == token
    return False


def _cmd_next(args) -> int:
    try:
        workspace = _resolve_workspace(args, ensure_run=bool(args.run_id))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        states = load_pipeline(pathlib.Path(args.yaml))
        current = ledger.current_state(workspace, "", initial="init")
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read workflow state: {exc}", file=sys.stderr)
        return 2
    handoff_text = args.handoff or ""
    if args.handoff_file:
        handoff_text = pathlib.Path(args.handoff_file).read_text(encoding="utf-8")
    handoff = extract_handoff(handoff_text)
    state = states.get(current)
    if state is None:
        print(f"ERROR: unknown state '{current}'")
        return 2

    def _iter(state_name: str) -> int:
        return ledger.iter_count(workspace, "", state_name)

    for tr in state.get("transitions", []):
        if not _when_matches(tr.get("when", "always"), handoff):
            continue
        gate_specs = tr.get("gates") or []
        ok, detail = gates.evaluate(workspace, gate_specs, _iter)
        target = tr.get("to", current)
        if ok:
            ledger.append_event(workspace, "",
                                {"type": "gate_pass", "phase": current, "payload": {"detail": detail}})
            ledger.append_event(workspace, "", {"type": "phase_enter", "phase": target,
                                                "payload": {"from": current,
                                                            "handoff": handoff}})
            print(f"STATE: {target}")
            print(f"transition: {current} -> {target} (handoff={handoff!r})")
            print(f"gates: {detail}")
            return 0
        fallback = tr.get("fallback")
        ledger.append_event(workspace, "",
                            {"type": "gate_fail", "phase": current, "payload": {"detail": detail}})
        if fallback and fallback != current:
            ledger.append_event(workspace, "", {"type": "phase_enter", "phase": fallback,
                                                "payload": {"from": current, "handoff": handoff,
                                                            "reason": "gate_fail_fallback"}})
            print(f"STATE: {fallback}")
            print(f"gate fail -> fallback {fallback}: {detail}")
            return 0
        # no fallback: try the next transition (e.g. NEED_FIX at cap falls
        # through to the ungated NEED_FIX -> finalize branch)
        continue
    ledger.append_event(workspace, "", {"type": "checkpoint", "phase": current,
                                        "payload": {"note": "no matching transition"}})
    print(f"STATE: {current}")
    print(f"no matching transition for handoff={handoff!r}")
    return 1


def _cmd_current(args) -> int:
    try:
        workspace = _resolve_workspace(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    load_pipeline(pathlib.Path(args.yaml))
    try:
        state = ledger.current_state(workspace, "", initial="init")
    except ledger.LedgerCorruptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"STATE: {state}")
    return 0


def _cmd_verify(args) -> int:
    try:
        workspace = _resolve_workspace(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        states = load_pipeline(pathlib.Path(args.yaml))
        events = ledger.read_events(workspace, "")
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot verify ledger: {exc}", file=sys.stderr)
        return 2
    if not events:
        print("VERIFY: ok (ledger empty)")
        return 0
    problems: list[str] = []
    current = "init"
    pending_gate: str | None = None
    for index, event in enumerate(events, start=1):
        ok, detail = gates.validate_value(event, "memory_event", f"ledger[{index}]")
        if not ok:
            problems.append(detail)
            continue
        event_type = event.get("type")
        phase = event.get("phase")
        if phase and phase not in states:
            problems.append(f"ledger[{index}]: unknown phase {phase!r}")
        if event_type in {"gate_pass", "gate_fail"}:
            if phase != current:
                problems.append(
                    f"ledger[{index}]: gate phase {phase!r} != replay state {current!r}"
                )
            pending_gate = event_type
            continue
        if event_type == "phase_enter":
            payload = event.get("payload") or {}
            from_state = payload.get("from")
            to_state = phase
            handoff = payload.get("handoff")
            fallback = payload.get("reason") == "gate_fail_fallback"
            if from_state != current:
                problems.append(
                    f"ledger[{index}]: payload.from {from_state!r} != replay state {current!r}"
                )
            transitions = states.get(current, {}).get("transitions", [])
            matching = [tr for tr in transitions
                        if _when_matches(tr.get("when", "always"), handoff)]
            declared = any(
                (tr.get("fallback") if fallback else tr.get("to", current)) == to_state
                for tr in matching
            )
            if not declared:
                kind = "fallback" if fallback else "transition"
                problems.append(
                    f"ledger[{index}]: {kind} {current!r} -> {to_state!r} "
                    f"not declared for handoff {handoff!r}"
                )
            expected_gate = "gate_fail" if fallback else "gate_pass"
            if pending_gate != expected_gate:
                problems.append(
                    f"ledger[{index}]: phase entry requires preceding {expected_gate}"
                )
            if to_state in states:
                current = to_state
            pending_gate = None
            continue
        if event_type == "checkpoint" and phase != current:
            problems.append(
                f"ledger[{index}]: checkpoint phase {phase!r} != replay state {current!r}"
            )
    if problems:
        print("VERIFY: problems found")
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    print("VERIFY: ok")
    return 0


def _cmd_doc_hash(args) -> int:
    try:
        workspace = _resolve_workspace(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target_doc = _workspace_doc(workspace, args.doc)
    document = gates.resolve_document(workspace, target_doc)
    if not target_doc or document is None:
        print("ERROR: target document cannot be located", file=sys.stderr)
        return 2
    try:
        digest = gates.file_sha256(document)
    except OSError as exc:
        print(f"ERROR: target document unreadable: {exc}", file=sys.stderr)
        return 2
    print(f"DOC: {document}")
    print(f"SHA256: {digest}")
    print(f"FENCED_LINES: {json.dumps(gates.fenced_code_block_lines(document))}")
    return 0


def _cmd_score(args) -> int:
    try:
        workspace = _resolve_workspace(args, ensure_run=bool(args.run_id))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    out = workspace / "quality_report.json"
    try:
        if out.exists():
            out.unlink()
    except OSError as exc:
        print(f"ERROR: cannot invalidate stale quality report: {exc}", file=sys.stderr)
        return 2

    try:
        rules_data = yamlite.load(pathlib.Path(args.rules).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot load scoring rules: {exc}", file=sys.stderr)
        return 2
    repo_name = _workspace_repo(workspace)
    scoring = rules_data.get("scoring", {})
    weights = scoring.get("weights", {})
    floors = scoring.get("floors", {})
    pass_total = scoring.get("pass_total", 80)
    max_rounds = scoring.get("max_rounds", 3)
    rules = {r["id"]: r for r in rules_data.get("rules", [])}

    depth = args.depth or "L2"
    target_doc = _workspace_doc(workspace, args.doc)
    target_document = gates.resolve_document(workspace, target_doc)
    if not target_doc or target_document is None:
        print(
            "ERROR: score target document is required and must exist; pass --doc "
            "or create the run with `new --doc`",
            file=sys.stderr,
        )
        return 2
    lint_ok, lint_detail = gates.lint_report_state(
        workspace, target_doc, depth
    )
    if not lint_ok:
        print(f"ERROR: lint prerequisite failed: {lint_detail}", file=sys.stderr)
        return 2
    review_ok, review_detail = gates.review_report_state(
        workspace, target_doc, depth
    )
    if not review_ok:
        print(f"ERROR: review prerequisite failed: {review_detail}", file=sys.stderr)
        return 2
    try:
        lint_report = json.loads(
            (workspace / "lint_report.json").read_text(encoding="utf-8-sig")
        )
        ledger_issues = ledger.read_issues(workspace, "")
        review_round = max(1, ledger.iter_count(workspace, "", "review"))
        current_phase = ledger.current_state(workspace, "", initial="init")
    except (OSError, json.JSONDecodeError, ledger.LedgerCorruptionError) as exc:
        print(f"ERROR: cannot read review inputs: {exc}", file=sys.stderr)
        return 2

    # A current lint hit overrides an older lifecycle record with the same
    # identity, so a rediscovered deterministic problem cannot stay "fixed".
    issue_projection = {ledger.issue_key(issue): issue for issue in ledger_issues}
    for issue in lint_report.get("issues", []):
        issue_projection[ledger.issue_key(issue)] = issue
    all_issues = list(issue_projection.values())
    for index, issue in enumerate(all_issues):
        ok, detail = gates.validate_value(issue, "issue", f"issues[{index}]")
        if not ok:
            print(f"ERROR: invalid issue ledger: {detail}", file=sys.stderr)
            return 2
        rule = rules.get(issue.get("rule_id"))
        if rule is None:
            print(
                f"ERROR: issues[{index}] references unknown rule_id "
                f"{issue.get('rule_id')!r}",
                file=sys.stderr,
            )
            return 2
        if issue.get("dimension") != rule.get("dimension"):
            print(
                f"ERROR: issues[{index}] dimension does not match rule "
                f"{issue.get('rule_id')}",
                file=sys.stderr,
            )
            return 2

    # Only the latest open projection contributes to the current score.
    # fixed/verified/closed/disputed records remain auditable in issues.jsonl
    # but must not keep a repaired document below the quality threshold.
    issues = [i for i in all_issues if i.get("status") == "open"]
    # cross-dimension dedup: group by object (file, line, anchor); when several
    # dimensions claim one object, keep only primary-rule issues; within one
    # dimension all rules count.
    groups: dict = {}
    for issue in issues:
        loc = issue.get("location", {})
        key = (loc.get("file"), loc.get("line"), loc.get("anchor"))
        groups.setdefault(key, []).append(issue)
    deduped = []
    for group in groups.values():
        dims = {i.get("dimension") for i in group}
        if len(dims) <= 1:
            deduped.extend(group)
            continue
        primaries = [i for i in group if rules.get(i.get("rule_id", ""), {}).get("primary")]
        if primaries:
            deduped.extend(primaries)
        else:
            first_dim = sorted(dims)[0]
            deduped.extend(i for i in group if i.get("dimension") == first_dim)

    dims = ["correctness", "completeness", "clarity", "accessibility"]
    dim_scores = {}
    for dim in dims:
        penalty = 0
        for issue in deduped:
            if issue.get("dimension") != dim:
                continue
            rule = rules.get(issue.get("rule_id", ""), {})
            penalty += int(rule.get("weight", 0))
        dim_scores[dim] = max(0, 100 - penalty)

    total = round(sum(dim_scores[d] * float(weights.get(d, 0)) for d in dims) / 100.0, 1)
    open_critical = any(i.get("severity") == "critical" and i.get("status") == "open"
                        for i in deduped)
    dim_fail = [d for d in dims if dim_scores[d] < int(floors.get(d, 0))]
    if depth == "L0":
        code_state, code_detail = gates.code_verify_state(workspace, depth)
    else:
        code_state, code_detail = gates.code_verify_state(
            workspace, depth, expected_doc=target_doc
        )
    if total >= pass_total and not dim_fail and not open_critical and code_state == "pass":
        verdict = "pass"
    elif code_state == "manual" or review_round >= int(max_rounds):
        verdict = "manual_intervention"
    else:
        verdict = "fail"

    manual_reasons = ["open critical issues remain" if open_critical else "",
                      f"dimensions below floor: {dim_fail}" if dim_fail else "",
                      f"round {review_round} >= max {max_rounds}" if review_round >= int(max_rounds) else "",
                      code_detail if code_state != "pass" else ""]
    # manual code-verification blocks surface as pending-human-check items
    code_verify = workspace / "code_verify.json"
    if code_verify.is_file():
        try:
            cv = json.loads(code_verify.read_text(encoding="utf-8-sig"))
            for block in cv.get("blocks", []):
                if block.get("status") == "manual":
                    loc = block.get("location", {})
                    where = f"{loc.get('file', '')}:{loc.get('line', '?')}"
                    manual_reasons.append(f"code block {where} 待人工实测验证（{block.get('tier', '?')}）")
        except json.JSONDecodeError:
            pass
    manual_reasons = [r for r in manual_reasons if r]

    try:
        final_doc_hash = gates.file_sha256(target_document)
    except OSError as exc:
        print(f"ERROR: target document became unreadable: {exc}", file=sys.stderr)
        return 2
    if final_doc_hash != lint_report.get("doc_sha256"):
        print("ERROR: target document changed during scoring; rerun review", file=sys.stderr)
        return 2
    try:
        inputs_hash = gates.review_inputs_sha256(workspace, depth)
    except OSError as exc:
        print(f"ERROR: review inputs became unavailable: {exc}", file=sys.stderr)
        return 2
    report = {
        "repo": repo_name,
        "doc": target_doc,
        "doc_sha256": final_doc_hash,
        "inputs_sha256": inputs_hash,
        "round": review_round,
        "depth": depth,
        "scores": {**dim_scores, "total": total},
        "issues": deduped,
        "verdict": verdict,
        "manual_reasons": manual_reasons,
    }
    report_ok, report_detail = gates.validate_value(
        report, "quality_report", "quality_report.json"
    )
    if not report_ok:
        print(f"ERROR: generated report violates contract: {report_detail}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        ledger.append_event(workspace, "", {"type": "checkpoint", "phase": current_phase,
                                              "payload": {"verdict": verdict, "total": total}})
    except OSError as exc:
        try:
            out.unlink()
        except OSError:
            pass
        print(f"ERROR: cannot append score checkpoint: {exc}", file=sys.stderr)
        return 2
    print(f"SCORE: {json_dims(dim_scores)} total={total} verdict={verdict} round={review_round}")
    print(f"written: {out}")
    return 0


def json_dims(scores: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in scores.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doc-agent.state_machine",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_next = sub.add_parser("next")
    p_next.add_argument("--workspace", required=True)
    p_next.add_argument("--yaml", default=str(DEFAULT_YAML))
    p_next.add_argument("--handoff", default=None)
    p_next.add_argument("--handoff-file", default=None)
    p_next.add_argument("--run-id", default=None,
                        help="run id below --workspace; use `new` to create isolated runs")

    p_cur = sub.add_parser("current")
    p_cur.add_argument("--workspace", required=True)
    p_cur.add_argument("--yaml", default=str(DEFAULT_YAML))
    p_cur.add_argument("--run-id", default=None)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--workspace", required=True)
    p_ver.add_argument("--yaml", default=str(DEFAULT_YAML))
    p_ver.add_argument("--run-id", default=None)

    p_score = sub.add_parser("score")
    p_score.add_argument("--workspace", required=True)
    p_score.add_argument("--rules", default=str(DEFAULT_RULES))
    p_score.add_argument("--depth", choices=["L0", "L1", "L2"], default=None,
                         help="review depth recorded in quality_report.json "
                              "(default L2; L0 exempts the code_verify gate)")
    p_score.add_argument("--doc", default=None)
    p_score.add_argument("--run-id", default=None)

    p_hash = sub.add_parser("doc-hash", help="print the target document SHA-256")
    p_hash.add_argument("--workspace", required=True)
    p_hash.add_argument("--doc", default=None)
    p_hash.add_argument("--run-id", default=None)

    p_new = sub.add_parser("new", help="create an isolated documentation run")
    p_new.add_argument("--workspace", required=True,
                       help="run base, normally .doc-work/<repo>")
    p_new.add_argument("--run-id", default=None,
                       help="optional stable id; generated when omitted")
    p_new.add_argument("--doc", default=None,
                       help="document name recorded in run.json")

    p_issue = sub.add_parser("issue-status", help="append an issue lifecycle transition")
    p_issue.add_argument("--workspace", required=True)
    p_issue.add_argument("--run-id", default=None)
    p_issue.add_argument("--rule-id", required=True)
    p_issue.add_argument("--file", required=True)
    p_issue.add_argument("--line", type=int, default=None)
    p_issue.add_argument("--anchor", default=None)
    p_issue.add_argument("--status", choices=sorted(ledger.ISSUE_STATUSES), required=True)

    args = parser.parse_args(argv)
    if args.cmd == "new":
        return _cmd_new(args)
    if args.cmd == "issue-status":
        return _cmd_issue_status(args)
    if args.cmd == "next":
        return _cmd_next(args)
    if args.cmd == "current":
        return _cmd_current(args)
    if args.cmd == "verify":
        return _cmd_verify(args)
    if args.cmd == "score":
        return _cmd_score(args)
    if args.cmd == "doc-hash":
        return _cmd_doc_hash(args)
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
