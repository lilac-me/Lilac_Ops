#!/usr/bin/env python3
"""Runtime black-box regression checks for Doc-Agent fail-closed invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "doc-agent"
ENGINE = PLUGIN / "engine"
CLI = ENGINE / "state_machine.py"
RULES = PLUGIN / "rules" / "rules.yaml"
PIPELINE = ENGINE / "doc_pipeline.yaml"
LINTER = PLUGIN / "linters" / "doc_lint.py"
LINTERS = PLUGIN / "linters"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(LINTERS))
import gates  # noqa: E402
import doc_lint  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, args)],
        text=True, encoding="utf-8", errors="replace",
        capture_output=True,
        check=False,
    )


def _prepare_review(workspace: pathlib.Path, depth: str, with_code: bool = False):
    document = workspace / "draft_guide.md"
    text = "# Guide\n\nNo code.\n"
    if with_code:
        text = (
            "# Guide\n\n```bash\necho one\n```\n\n"
            "```python\nprint(2)\n```\n"
        )
    document.write_text(text, encoding="utf-8")
    digest = _hash(document)
    _write_json(workspace / "lint_report.json", {
        "doc": str(document),
        "doc_sha256": digest,
        "depth": depth,
        "status": "completed",
        "issues": [],
        "errors": [],
    })
    reviewers = sorted(gates.REVIEWERS_BY_DEPTH[depth])
    _write_json(workspace / "review_report.json", {
        "doc": str(document),
        "doc_sha256": digest,
        "depth": depth,
        "status": "completed",
        "required_reviewers": reviewers,
        "completed_reviewers": reviewers,
        "errors": [],
    })
    return document, digest


def check_score_prerequisites() -> None:
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        (workspace / "draft_guide.md").write_text("# Guide\n", encoding="utf-8")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 2, "score accepted a missing lint report")
        _check(not (workspace / "quality_report.json").exists(),
               "failed score left a quality report behind")

    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        document = workspace / "draft_guide.md"
        document.write_text("# Guide\n", encoding="utf-8")
        _write_json(workspace / "lint_report.json", {
            "doc": str(document), "doc_sha256": _hash(document),
            "depth": "L0", "status": "completed", "issues": [], "errors": [],
        })
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 2, "score accepted a missing review report")

    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        document, _ = _prepare_review(workspace, "L0")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 0, "valid L0 score failed")
        _check(gates.score_pass(workspace)[0], "valid L0 report did not pass")
        (workspace / "issues.jsonl").write_text("{}\n", encoding="utf-8")
        _check(not gates.score_pass(workspace)[0],
               "quality report ignored changed review inputs")
        (workspace / "issues.jsonl").unlink()
        document.write_text("# changed\n", encoding="utf-8")
        _check(not gates.score_pass(workspace)[0], "stale quality report passed")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 2, "stale lint/review reports were accepted")


def check_code_verification() -> None:
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        _, digest = _prepare_review(workspace, "L2", with_code=True)
        blocks = [
            {
                "location": {"file": "draft_guide.md", "line": 3},
                "tier": "T1", "status": "pass",
                "evidence": "shell output verified",
            },
            {
                "location": {"file": "draft_guide.md", "line": 7},
                "tier": "T2", "status": "pass",
                "evidence": "minimal Python execution verified",
            },
        ]
        _write_json(workspace / "code_verify.json", {
            "doc": "guide", "doc_sha256": digest, "blocks": blocks,
        })
        hash_result = _run_cli(
            "doc-hash", "--workspace", workspace, "--doc", "guide",
        )
        _check(hash_result.returncode == 0 and "FENCED_LINES: [3, 7]" in hash_result.stdout,
               "doc-hash did not expose exact fenced block locations")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L2", "--doc", "guide",
        )
        _check(result.returncode == 0, "valid L2 score failed")
        report = json.loads((workspace / "quality_report.json").read_text(encoding="utf-8"))
        _check(report["verdict"] == "pass", "valid L2 review did not pass")

        duplicate = dict(blocks[0])
        _write_json(workspace / "code_verify.json", {
            "doc": "guide", "doc_sha256": digest,
            "blocks": [duplicate, duplicate],
        })
        state, _ = gates.code_verify_state(workspace, "L2", "guide")
        _check(state == "fail", "duplicate code-block locations passed")


def check_corruption_and_schema() -> None:
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        _prepare_review(workspace, "L0")
        (workspace / "issues.jsonl").write_text('{"status":"open"\n', encoding="utf-8")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 2, "corrupt issue ledger was ignored")
        _check(not (workspace / "quality_report.json").exists(),
               "corrupt input left a quality report behind")

    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        _prepare_review(workspace, "L0")
        bad_issue = {
            "id": "bad", "rule_id": "R-XXX-999", "dimension": "clarity",
            "severity": "major", "location": {"file": "guide.md"},
            "source": "reviewer:test", "status": "open",
        }
        (workspace / "issues.jsonl").write_text(
            json.dumps(bad_issue) + "\n", encoding="utf-8"
        )
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 2, "unknown issue rule_id was accepted")

    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        _write_json(workspace / "quality_report.json", {"verdict": "pass"})
        _check(not gates.score_pass(workspace)[0], "malformed quality report passed")


def check_ledger_replay() -> None:
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        valid = [
            {"ts": "now", "type": "gate_pass", "phase": "init", "payload": {}},
            {"ts": "now", "type": "phase_enter", "phase": "context",
             "payload": {"from": "init", "handoff": "CONTEXT_READY"}},
        ]
        (workspace / "ledger.jsonl").write_text(
            "\n".join(json.dumps(item) for item in valid) + "\n", encoding="utf-8"
        )
        _check(_run_cli("verify", "--workspace", workspace,
                        "--yaml", PIPELINE).returncode == 0,
               "valid ledger replay failed")
        invalid = valid + [
            {"ts": "now", "type": "gate_pass", "phase": "context", "payload": {}},
            {"ts": "now", "type": "phase_enter", "phase": "done",
             "payload": {"from": "context", "handoff": "DONE"}},
        ]
        (workspace / "ledger.jsonl").write_text(
            "\n".join(json.dumps(item) for item in invalid) + "\n", encoding="utf-8"
        )
        _check(_run_cli("verify", "--workspace", workspace,
                        "--yaml", PIPELINE).returncode == 1,
               "undeclared transition passed replay")
        (workspace / "ledger.jsonl").write_text("{bad\n", encoding="utf-8")
        _check(_run_cli("verify", "--workspace", workspace,
                        "--yaml", PIPELINE).returncode == 2,
               "corrupt ledger passed replay")


def check_end_to_end_workflow() -> None:
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        _prepare_review(workspace, "L0")
        transitions = [
            ("PLAN_READY", "plan"),
            ("REVIEW_READY", "review"),
        ]
        for handoff, expected in transitions:
            result = _run_cli(
                "next", "--workspace", workspace, "--yaml", PIPELINE,
                "--handoff", f"handoff: {handoff}",
            )
            _check(result.returncode == 0 and f"STATE: {expected}" in result.stdout,
                   f"workflow did not enter {expected}")
        result = _run_cli(
            "score", "--workspace", workspace, "--rules", RULES,
            "--depth", "L0", "--doc", "guide",
        )
        _check(result.returncode == 0, "end-to-end score failed")
        for handoff, expected in [("PASSED", "finalize"), ("DONE", "done")]:
            result = _run_cli(
                "next", "--workspace", workspace, "--yaml", PIPELINE,
                "--handoff", f"handoff: {handoff}",
            )
            _check(result.returncode == 0 and f"STATE: {expected}" in result.stdout,
                   f"workflow did not enter {expected}")
        _check(_run_cli("verify", "--workspace", workspace,
                        "--yaml", PIPELINE).returncode == 0,
               "engine-generated ledger failed replay")


def check_lint_failure() -> None:
    with tempfile.TemporaryDirectory() as temp:
        repo = pathlib.Path(temp)
        document = repo / "guide.md"
        report = repo / "lint_report.json"
        document.write_text("# Guide\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable, str(LINTER), "lint", "--repo", str(repo),
                "--doc", str(document), "--depth", "L0", "--no-network",
                "--out", str(report),
            ],
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        _check(result.returncode in {0, 1} and data["status"] == "completed",
               "normal lint run did not complete")
        _check(gates.schema_valid(repo, "lint_report.json")[0],
               "normal lint report violates its contract")
        anchors = [
            item.get("location", {}).get("anchor")
            for item in data["issues"] if item.get("source") == "lint:manifest"
        ]
        _check(len(anchors) == len(set(anchors)),
               "manifest lint issues collapse to duplicate lifecycle identities")
        _write_json(repo / "review_report.json", {
            "doc": str(document), "doc_sha256": data["doc_sha256"],
            "depth": "L0", "status": "completed",
            "required_reviewers": [], "completed_reviewers": [], "errors": [],
        })
        score = _run_cli(
            "score", "--workspace", repo, "--rules", RULES,
            "--depth", "L0", "--doc", document,
        )
        quality = json.loads((repo / "quality_report.json").read_text(encoding="utf-8"))
        manifest_count = sum(
            item.get("source") == "lint:manifest" for item in quality["issues"]
        )
        _check(score.returncode == 0 and quality["verdict"] == "fail",
               "score did not directly consume completed lint results")
        _check(manifest_count == len(anchors),
               "score collapsed distinct manifest lint issues")

        report.unlink()
        result = subprocess.run(
            [
                sys.executable, str(LINTER), "lint", "--repo", str(repo),
                "--doc", str(document), "--manifest", str(repo / "missing.yaml"),
                "--depth", "L0", "--no-network", "--out", str(report),
            ],
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        _check(result.returncode == 2 and data["status"] == "failed",
               "linter dependency failure did not fail closed")
        _check(gates.schema_valid(repo, "lint_report.json")[0],
               "failed lint report violates its diagnostic contract")

        english = repo / "docs" / "en" / "guide.md"
        english.parent.mkdir(parents=True)
        english.write_text("# English Guide\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable, str(LINTER), "lint", "--repo", str(repo),
                "--doc", str(english), "--depth", "L0", "--no-network",
                "--out", str(report),
            ],
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        _check(
            result.returncode == 2 and data["status"] == "failed"
            and data["issues"] == []
            and any("outside doc-agent scope" in error for error in data["errors"]),
            "docs/en target was not excluded before lint execution",
        )
        _check(gates.schema_valid(repo, "lint_report.json")[0],
               "docs/en exclusion report violates its contract")

        document.write_text("# Guide\n", encoding="utf-8")
        original_run_linter = doc_lint._run_linter
        changed = False

        def mutate_during_lint(name, issues, args, doc_path, target_repo):
            nonlocal changed
            error = original_run_linter(name, issues, args, doc_path, target_repo)
            if not changed:
                doc_path.write_text("# Guide\n\nTODO: changed during lint\n", encoding="utf-8")
                changed = True
            return error

        lint_args = argparse.Namespace(
            repo=str(repo), doc=str(document), manifest=None,
            capabilities=None, include=None, exclude=None,
            self_developed=False, rules=str(RULES), figures=None,
            depth="L0", no_network=True, out=str(report),
        )
        doc_lint._run_linter = mutate_during_lint
        try:
            returncode = doc_lint._cmd_lint(lint_args)
        finally:
            doc_lint._run_linter = original_run_linter
        data = json.loads(report.read_text(encoding="utf-8"))
        _check(
            returncode == 2 and data["status"] == "failed"
            and any("changed during lint" in error for error in data["errors"]),
            "document mutation during lint did not fail closed",
        )
        _check(gates.schema_valid(repo, "lint_report.json")[0],
               "document-mutation report violates its contract")


def main() -> int:
    checks = [
        check_score_prerequisites,
        check_code_verification,
        check_corruption_and_schema,
        check_ledger_replay,
        check_end_to_end_workflow,
        check_lint_failure,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"RESULT: PASS ({len(checks)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
