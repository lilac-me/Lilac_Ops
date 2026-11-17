from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from engine.agent_registry import load_agent_registry
from engine.gates import (
    load_json,
    validate_artifacts,
    validate_schema,
    write_confirmed_run_config,
)
from engine.orchestrator import next_agent_for_state
from engine.workflow_runner import next_action
from engine.state_machine import (
    _entry_hash,
    current_state,
    load_state_machine,
    read_log,
    select_transition,
    transition,
    validate_definition,
    verify_log,
    sha256_file,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PLUGIN_ROOT.parents[1]
WORKFLOW = PLUGIN_ROOT / "workflows" / "batch-invariance-validation.yaml"
RUN_SCHEMA = PLUGIN_ROOT / "contracts" / "run_config.schema.json"
STAGE_SCHEMA = PLUGIN_ROOT / "contracts" / "stage_result.schema.json"


@pytest.fixture(scope="module")
def sm():
    return load_state_machine(WORKFLOW)


def _candidate() -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "run-001",
        "model": {"path": "/models/qwen", "architecture": "Qwen"},
        "chip": "A3",
        "device_topology": {
            "visible_devices": list(range(8)),
            "device_count": 8,
            "parallelism": {"tp": 8},
        },
        "tp_plan": {
            "phase1": 8,
            "phase2_long_output": 8,
            "phase2_long_input": 8,
            "phase2_matrix": [1, 2, 4],
            "phase3": 8,
        },
        "sampling": {"phase2_temperature": 0.6, "phase3_temperature": 0, "seed": 1234},
        "service": {
            "generation_config": "vllm",
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        "scheduling": {
            "max_model_len": 22000,
            "max_num_batched_tokens": 65536,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.9,
            "async_scheduling": False,
        },
        "mutation_policy": "validate_only",
        "tp_change_history": [],
    }


def _write_config(workspace: Path) -> Path:
    path = workspace / "run_config.json"
    write_confirmed_run_config(_candidate(), path, RUN_SCHEMA, "确认默认 TP 计划", "default_confirmed")
    return path


def _result(sm: dict, state: str, handoff: str, *, blocked_reason: str | None = None,
            resume_state: str | None = None) -> dict:
    if state == "environment_admission":
        stage, summary = "environment", {"tp": 8, "chip": "A3", "visible_devices": list(range(8))}
    elif state.startswith("phase1_"):
        stage, summary = "phase1", {"tp": 8, "temperature": 0.6}
    elif state == "phase2_long_output":
        stage, summary = "phase2", {"tp": 8, "batch_size": 64, "input_tokens": 2048, "output_tokens": 20480}
    elif state == "phase2_long_input":
        stage, summary = "phase2", {"tp": 8, "batch_size": 64, "input_tokens": 20480, "output_tokens": 2048}
    elif state in {"phase2_matrix", "phase2_failure_diagnosis"}:
        stage, summary = "phase2", {
            "tp": [1, 2, 4], "batch_sizes": [16, 32], "rounds": 3,
            "tp_results": [
                {"tp": 1, "status": "PASS", "reason_code": None},
                {"tp": 2, "status": "PASS", "reason_code": None},
                {"tp": 4, "status": "PASS", "reason_code": None},
            ],
            "scenario_results": [
                {"tp": tp, "batch_size": bs, "input_tokens": inp, "output_tokens": out, "rounds": 3, "status": "PASS"}
                for tp in (1, 2, 4)
                for bs, inp, out in ((16, 100, 512), (32, 100, 512), (16, 1024, 2048), (32, 1024, 2048))
            ],
        }
    elif state == "phase3_math500":
        stage, summary = "phase3", {
            "tp": 8, "rounds": 10, "samples_per_round": 500, "temperature": 0,
            "enable_prefix_caching": False, "enable_chunked_prefill": False,
            "async_scheduling": False,
            "accuracy_pass": True,
            "round_results": [
                {"round": index, "successful_samples": 500, "score": 0.9, "raw_text_sha256": "a" * 64}
                for index in range(1, 11)
            ],
        }
    else:
        raise AssertionError(state)
    catalog = sm["handoff_catalog"][handoff.removeprefix("handoff: ")]
    outcome = "BLOCKED" if blocked_reason else catalog["outcome"]
    return {
        "schema_version": "1.0", "run_id": "run-001", "stage": stage, "state": state,
        "outcome": outcome, "reason_code": blocked_reason or catalog.get("reason_code"),
        "resume_state": resume_state, "attempt": 1, "config_summary": summary,
        "artifacts": [], "handoff": handoff,
    }


def _advance_worker(workspace: Path, sm: dict, handoff: str, **kwargs) -> None:
    state = current_state(workspace, sm)
    result = _result(sm, state, handoff, **kwargs)
    required = {
        "environment_admission": ["environment_snapshot"], "phase1_profiling": ["profiling_manifest"],
        "phase1_triage": ["operator_analysis"], "phase1_critical_path_review": ["critical_path_evidence"],
        "phase1_operator_10k": ["operator_10k_result"], "phase1_replacement": ["replacement_evidence"],
        "phase1_reprofile": ["reprofile_result"], "phase2_long_output": ["functional_result"],
        "phase2_long_input": ["functional_result"], "phase2_matrix": ["matrix_result"],
        "phase2_failure_diagnosis": ["diagnosis_evidence"],
        "phase3_math500": ["math500_manifest", "math500_round_results", "math500_raw_text_hashes"],
    }[state]
    for artifact_type in required:
        relative = f"evidence/{state}-{artifact_type}.json"
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": state, "type": artifact_type}), encoding="utf-8")
        result["artifacts"].append({"path": relative, "type": artifact_type, "sha256": sha256_file(path)})
    transition(workspace, sm, handoff, context=result)


def test_real_workflow_is_reachable_and_fail_closed(sm):
    assert validate_definition(sm) == []
    expected = {
        "tp_plan_gate", "blocked_tp_plan_unconfirmed", "environment_admission",
        "blocked_environment", "phase1_profiling", "phase1_triage",
        "phase1_critical_path_review", "phase1_operator_10k", "phase1_replacement",
        "phase1_reprofile", "blocked_phase1", "phase2_long_output",
        "phase2_long_input", "phase2_matrix", "phase2_failure_diagnosis",
        "blocked_phase2", "phase3_math500", "blocked_phase3", "recovery_controller",
        "recovery_exhausted", "finalize", "done", "failed",
    }
    assert {state["id"] for state in sm["states"]} == expected


def test_every_declared_handoff_edge_is_selectable(sm):
    for state in sm["states"]:
        for index, edge in enumerate(state.get("exit_transitions", [])):
            token = edge.get("condition", {}).get("handoff_match")
            if not token:
                continue
            matched, selected = select_transition(sm, state["id"], token)
            assert matched == index
            assert selected["goto"] == edge["goto"]


def test_every_fallback_is_selectable(sm):
    for state in sm["states"]:
        transitions = state.get("exit_transitions", [])
        if not transitions:
            continue
        index, selected = select_transition(sm, state["id"], "handoff: UNKNOWN")
        assert index == len(transitions) - 1
        assert selected["condition"] == {"always": True}


def test_success_path_is_logged_and_replayable(tmp_path, sm):
    _write_config(tmp_path)
    handoffs = [
        "handoff: TP_PLAN_CONFIRMED", "handoff: ENVIRONMENT_READY",
        "handoff: PROFILING_READY", "handoff: ALLOWLIST_PASS",
        "handoff: LONG_OUTPUT_PASS", "handoff: LONG_INPUT_PASS",
        "handoff: MATRIX_PASS", "handoff: MATH500_PASS",
        "handoff: FINALIZE_SUCCESS",
    ]
    transition(tmp_path, sm, handoffs[0])
    for handoff in handoffs[1:-1]:
        _advance_worker(tmp_path, sm, handoff)
    transition(tmp_path, sm, handoffs[-1])
    assert current_state(tmp_path, sm) == "done"
    assert verify_log(tmp_path, sm) == []
    assert len(read_log(tmp_path)) == len(handoffs)
    assert (tmp_path / "PROGRESS.md").is_file()


def test_replay_rejects_tampered_evidence(tmp_path, sm):
    config = _write_config(tmp_path)
    evidence = tmp_path / "environment" / "snapshot.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    result = _result(sm, "environment_admission", "handoff: ENVIRONMENT_READY")
    result["artifacts"] = [{"path": "environment/snapshot.json", "type": "environment_snapshot", "sha256": sha256_file(evidence)}]
    transition(tmp_path, sm, "handoff: ENVIRONMENT_READY", context=result)
    assert verify_log(tmp_path, sm) == []
    evidence.write_text('{"tampered": true}\n', encoding="utf-8")
    assert any("artifact hash mismatch" in item for item in verify_log(tmp_path, sm))


@pytest.mark.parametrize(
    "to_block,resume,expected",
    [
        ("handoff: TP_PLAN_UNCONFIRMED", "handoff: TP_PLAN_RESUME", "tp_plan_gate"),
    ],
)
def test_tp_pause_is_resumable(tmp_path, sm, to_block, resume, expected):
    transition(tmp_path, sm, to_block)
    transition(tmp_path, sm, resume)
    assert current_state(tmp_path, sm) == expected


def test_environment_and_phase_pauses_are_resumable(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_BLOCKED",
                    blocked_reason="BLOCKED_RUNTIME_PACKAGE_MISSING", resume_state="environment_admission")
    transition(tmp_path, sm, "handoff: ENVIRONMENT_RESUME")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_READY")
    _advance_worker(tmp_path, sm, "handoff: PHASE1_BLOCKED",
                    blocked_reason="BLOCKED_MISSING_OP_STATISTIC", resume_state="phase1_profiling")
    transition(tmp_path, sm, "handoff: PHASE1_RESUME")
    _advance_worker(tmp_path, sm, "handoff: PROFILING_READY")
    _advance_worker(tmp_path, sm, "handoff: ALLOWLIST_PASS")
    _advance_worker(tmp_path, sm, "handoff: PHASE2_BLOCKED",
                    blocked_reason="BLOCKED_SCHEDULER_EVIDENCE", resume_state="phase2_long_output")
    transition(tmp_path, sm, "handoff: PHASE2_RESUME_LONG_OUTPUT")
    assert current_state(tmp_path, sm) == "phase2_long_output"
    assert verify_log(tmp_path, sm) == []


def test_blocked_phase_must_resume_to_recorded_state(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_READY")
    _advance_worker(tmp_path, sm, "handoff: PROFILING_READY")
    _advance_worker(tmp_path, sm, "handoff: ALLOWLIST_PASS")
    _advance_worker(tmp_path, sm, "handoff: PHASE2_BLOCKED",
                    blocked_reason="BLOCKED_SCHEDULER_EVIDENCE", resume_state="phase2_long_output")
    with pytest.raises(ValueError, match="must resume to phase2_long_output"):
        transition(tmp_path, sm, "handoff: PHASE2_RESUME_MATRIX")


def test_blocked_state_automatically_dispatches_model_recovery(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_BLOCKED",
                    blocked_reason="BLOCKED_OPERATOR_INSTALLATION",
                    resume_state="environment_admission")
    action = next_action(tmp_path, sm, RUN_SCHEMA)
    assert action["action"] == "recover"
    assert action["resume_state"] == "environment_admission"
    assert current_state(tmp_path, sm) == "recovery_controller"
    assert action["goal"]
    assert "apply_minimal_repair" in action["allowed_actions"]
    assert "lower_acceptance_threshold" in action["forbidden_actions"]
    assert verify_log(tmp_path, sm) == []


def test_unconfirmed_tp_never_writes_run_config(tmp_path):
    destination = tmp_path / "run_config.json"
    with pytest.raises(ValueError, match="BLOCKED_TP_PLAN_UNCONFIRMED"):
        write_confirmed_run_config(_candidate(), destination, RUN_SCHEMA, "")
    assert not destination.exists()


def test_transition_cannot_forge_tp_confirmation(tmp_path, sm):
    with pytest.raises(ValueError, match="BLOCKED_TP_PLAN_UNCONFIRMED"):
        transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    assert read_log(tmp_path) == []


def test_from_state_override_cannot_skip_pipeline(tmp_path, sm):
    _write_config(tmp_path)
    forged = _result(sm, "phase3_math500", "handoff: MATH500_PASS")
    with pytest.raises(ValueError, match="does not match current state"):
        transition(tmp_path, sm, "handoff: MATH500_PASS", from_state="phase3_math500", context=forged)


def test_worker_cannot_pass_without_required_evidence(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    result = _result(sm, "environment_admission", "handoff: ENVIRONMENT_READY")
    with pytest.raises(ValueError, match="artifacts|artifact types"):
        transition(tmp_path, sm, "handoff: ENVIRONMENT_READY", context=result)


def test_verify_log_replays_stage_contract_after_consistent_rehash(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_READY")
    entries = read_log(tmp_path)
    entries[1]["stage_result"]["config_summary"].pop("visible_devices")
    entries[1]["stage_result_hash"] = hashlib.sha256(json.dumps(
        entries[1]["stage_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    entries[1]["entry_hash"] = _entry_hash(entries[1])
    (tmp_path / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in entries) + "\n",
        encoding="utf-8",
    )
    assert any("stage result replay failed" in item for item in verify_log(tmp_path, sm))


def test_matrix_pass_requires_all_scenarios_for_passing_tp(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    _advance_worker(tmp_path, sm, "handoff: ENVIRONMENT_READY")
    _advance_worker(tmp_path, sm, "handoff: PROFILING_READY")
    _advance_worker(tmp_path, sm, "handoff: ALLOWLIST_PASS")
    _advance_worker(tmp_path, sm, "handoff: LONG_OUTPUT_PASS")
    _advance_worker(tmp_path, sm, "handoff: LONG_INPUT_PASS")
    result = _result(sm, "phase2_matrix", "handoff: MATRIX_PASS")
    result["config_summary"]["scenario_results"][0]["status"] = "SKIPPED_TP_RESOURCE_UNAVAILABLE"
    evidence = tmp_path / "evidence" / "matrix.json"
    evidence.parent.mkdir(exist_ok=True)
    evidence.write_text("{}", encoding="utf-8")
    result["artifacts"] = [{"path": "evidence/matrix.json", "type": "matrix_result", "sha256": sha256_file(evidence)}]
    with pytest.raises(ValueError, match="every scenario for a passing TP"):
        transition(tmp_path, sm, "handoff: MATRIX_PASS", context=result)


def test_worker_transition_rejects_wrong_stage_handoff(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    forged = _result(sm, "environment_admission", "handoff: MATH500_PASS")
    with pytest.raises(ValueError, match="stage_result|explicitly declared"):
        transition(tmp_path, sm, "handoff: MATH500_PASS", context=forged)


def test_confirmed_config_cannot_be_replaced_without_reconfirmation(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    changed = _candidate()
    changed["device_topology"]["parallelism"]["tp"] = 4
    changed["tp_plan"]["phase1"] = 4
    write_confirmed_run_config(changed, tmp_path / "run_config.json", RUN_SCHEMA, "改为 TP4")
    result = _result(sm, "environment_admission", "handoff: ENVIRONMENT_READY")
    result["config_summary"]["tp"] = 4
    evidence = tmp_path / "environment" / "snapshot.json"
    evidence.parent.mkdir()
    evidence.write_text("{}", encoding="utf-8")
    result["artifacts"] = [{"path": "environment/snapshot.json", "type": "environment_snapshot", "sha256": sha256_file(evidence)}]
    with pytest.raises(ValueError, match="changed without a new TP confirmation"):
        transition(tmp_path, sm, "handoff: ENVIRONMENT_READY", context=result)


def test_math500_failure_cannot_be_flipped_to_success(tmp_path, sm):
    _write_config(tmp_path)
    for handoff in [
        "handoff: TP_PLAN_CONFIRMED", "handoff: ENVIRONMENT_READY",
        "handoff: PROFILING_READY", "handoff: ALLOWLIST_PASS",
        "handoff: LONG_OUTPUT_PASS", "handoff: LONG_INPUT_PASS", "handoff: MATRIX_PASS",
    ]:
        if current_state(tmp_path, sm) == "tp_plan_gate":
            transition(tmp_path, sm, handoff)
        else:
            _advance_worker(tmp_path, sm, handoff)
    _advance_worker(tmp_path, sm, "handoff: MATH500_FAIL")
    with pytest.raises(ValueError, match="FINALIZE_FAILURE"):
        transition(tmp_path, sm, "handoff: FINALIZE_SUCCESS")
    transition(tmp_path, sm, "handoff: FINALIZE_FAILURE")
    assert current_state(tmp_path, sm) == "failed"
    assert verify_log(tmp_path, sm) == []


def test_log_hash_chain_detects_history_rewrite(tmp_path, sm):
    _write_config(tmp_path)
    transition(tmp_path, sm, "handoff: TP_PLAN_CONFIRMED")
    path = tmp_path / "state_transitions.jsonl"
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["rationale"] = "rewritten"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    assert any("entry hash mismatch" in item for item in verify_log(tmp_path, sm))


def test_explicit_confirmation_atomically_writes_valid_config(tmp_path):
    destination = tmp_path / "run_config.json"
    write_confirmed_run_config(_candidate(), destination, RUN_SCHEMA, "确认默认 TP 计划", "default_confirmed")
    document = load_json(destination)
    assert document["tp_confirmation"]["user_response"] == "确认默认 TP 计划"
    assert validate_schema(document, RUN_SCHEMA) == []


def test_stage_dispatch_is_impossible_before_tp_confirmation(tmp_path, sm):
    with pytest.raises(ValueError, match="BLOCKED_TP_PLAN_UNCONFIRMED"):
        next_agent_for_state(sm, "environment_admission", run_config=None,
                             run_config_schema=RUN_SCHEMA)
    assert next_agent_for_state(sm, "blocked_environment", run_config=None,
                                run_config_schema=RUN_SCHEMA) is None


def test_confirmed_config_allows_only_the_declared_stage_agent(tmp_path, sm):
    config = tmp_path / "run_config.json"
    write_confirmed_run_config(_candidate(), config, RUN_SCHEMA, "确认")
    assert next_agent_for_state(sm, "environment_admission", run_config=config,
                                run_config_schema=RUN_SCHEMA) == (
                                    "batch-invariance-validation:environment-admission"
                                )


def test_stage_result_schema_and_evidence_hash(tmp_path):
    evidence = tmp_path / "environment" / "snapshot.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    result = {
        "schema_version": "1.0", "run_id": "run-001", "stage": "environment",
        "state": "environment_admission", "outcome": "PASS", "reason_code": None,
        "resume_state": None, "attempt": 1,
        "config_summary": {"tp": 8, "chip": "A3", "visible_devices": list(range(8))},
        "artifacts": [{"path": "environment/snapshot.json", "type": "environment_snapshot", "sha256": digest}],
        "handoff": "handoff: ENVIRONMENT_READY",
    }
    assert validate_schema(result, STAGE_SCHEMA) == []
    assert validate_artifacts(result, tmp_path) == []
    result["artifacts"][0]["sha256"] = "0" * 64
    assert "hash mismatch" in validate_artifacts(result, tmp_path)[0]


def test_stage_contract_rejects_wrong_scenario_parameters(sm):
    long_output = _result(sm, "phase2_long_output", "handoff: LONG_OUTPUT_PASS")
    long_output["config_summary"]["batch_size"] = 32
    assert any("64" in error for error in validate_schema(long_output, STAGE_SCHEMA))
    phase3 = _result(sm, "phase3_math500", "handoff: MATH500_PASS")
    phase3["config_summary"]["async_scheduling"] = True
    assert not any("async_scheduling" in error for error in validate_schema(phase3, STAGE_SCHEMA))


def test_all_documented_reason_codes_are_registered():
    schema = load_json(STAGE_SCHEMA)
    registered = set(schema["properties"]["reason_code"]["enum"])
    source_root = REPO_ROOT / "skills" / "inference-model-skills" / "batch-invariance-validation"
    documented: set[str] = set()
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        documented.update(re.findall(
            r"\b(?:BLOCKED|FAIL|INVALID|SKIPPED)_[A-Z0-9]+(?:_[A-Z0-9]+)*\b", text
        ))
    assert documented <= registered


def test_agent_registry_uses_qualified_ids():
    registry = load_agent_registry(PLUGIN_ROOT)
    assert len(registry) == 5
    assert all(key.startswith("batch-invariance-validation:") for key in registry)
    worker = registry["batch-invariance-validation:phase1-operator-analysis"]
    assert worker["schema_path"] == STAGE_SCHEMA


def test_workflow_agents_and_agent_handoffs_are_registered(sm):
    registry = load_agent_registry(PLUGIN_ROOT)
    declared_agents = {state.get("agent") for state in sm["states"] if state.get("agent")}
    assert declared_agents <= set(registry)
    catalog = {f"handoff: {name}" for name in sm["handoff_catalog"]}
    for prompt in (PLUGIN_ROOT / "agents").rglob("AGENT.md"):
        tokens = set(re.findall(r"handoff: [A-Z][A-Z0-9_]+", prompt.read_text(encoding="utf-8")))
        assert tokens <= catalog, f"{prompt}: {tokens - catalog}"


def test_every_catalog_handoff_is_used_by_the_real_graph(sm):
    used = {
        edge["condition"]["handoff_match"].removeprefix("handoff: ")
        for state in sm["states"]
        for edge in state.get("exit_transitions", [])
        if "handoff_match" in edge.get("condition", {})
    }
    assert set(sm["handoff_catalog"]) == used


def test_contracts_are_valid_draft_2020_schemas():
    Draft202012Validator.check_schema(load_json(RUN_SCHEMA))
    Draft202012Validator.check_schema(load_json(STAGE_SCHEMA))
    Draft202012Validator.check_schema(load_json(PLUGIN_ROOT / "contracts" / "recovery_attempt.schema.json"))


def test_split_skills_follow_repository_schema_and_have_no_duplicate_names():
    root = REPO_ROOT / "skills" / "inference-model-skills" / "batch-invariance-validation"
    skill_files = sorted(root.glob("*/SKILL.md"))
    assert len(skill_files) == 5
    names: set[str] = set()
    for skill_file in skill_files:
        text = skill_file.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        assert frontmatter["name"] == skill_file.parent.name
        assert frontmatter["name"] not in names
        names.add(frontmatter["name"])
        assert frontmatter["description"].startswith("This skill should be used when")
        assert frontmatter["domain"] == "inference-model"
        assert frontmatter["function"] == "batch-invariance-validation"
        assert text.split("---", 2)[2].strip()


def test_installers_and_marketplace_register_the_new_plugin():
    for path in (REPO_ROOT / "install.sh", REPO_ROOT / "install.bat",
                 REPO_ROOT / ".claude-plugin" / "marketplace.json"):
        assert "batch-invariance-validation" in path.read_text(encoding="utf-8")
    shell = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    batch = (REPO_ROOT / "install.bat").read_text(encoding="utf-8")
    for shared in ("references", "scripts"):
        assert f'ln -s "$PWD/skills/inference-model-skills/batch-invariance-validation/{shared}"' not in shell
    assert "call :copy_batch_shared" not in batch
    assert ':copy_batch_shared\n' not in batch


def test_cli_runs_from_repository_root_with_utf8(tmp_path):
    process = subprocess.run(
        [sys.executable, str(PLUGIN_ROOT / "engine" / "state_machine.py"),
         "validate-definition", "--yaml", str(WORKFLOW)],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="strict",
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["valid"] is True
