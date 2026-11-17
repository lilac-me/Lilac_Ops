from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TEMPLATES = SKILL_ROOT / "references" / "templates"
sys.path.insert(0, str(SCRIPTS))

import audit_e2e_log as log_audit  # noqa: E402
import build_public_package as public_package  # noqa: E402
import fix_casebook  # noqa: E402
import generate_alignment_report as report  # noqa: E402
import prepare_remote_script as transport  # noqa: E402
import record_state_event as state_event  # noqa: E402
import validate_difference_packet as packet_validator  # noqa: E402
import validate_production_patch as patch_validator  # noqa: E402
import validate_state_card as state_validator  # noqa: E402
import managed_job  # noqa: E402


class DifferencePacketV2Tests(unittest.TestCase):
    def test_v1_tensor_packet_remains_valid(self) -> None:
        packet = json.loads((TEMPLATES / "difference_packet.json").read_text(encoding="utf-8"))
        packet["schema_version"] = 1
        packet.pop("boundary_kind")
        result = packet_validator.validate(packet)
        self.assertEqual(result["boundary_kind"], "tensor")

    def test_v2_tensor_and_state_packets_are_valid(self) -> None:
        tensor_packet = json.loads(
            (TEMPLATES / "difference_packet.json").read_text(encoding="utf-8")
        )
        state_packet = json.loads(
            (TEMPLATES / "state_difference_packet.json").read_text(encoding="utf-8")
        )
        self.assertEqual(packet_validator.validate(tensor_packet)["boundary_kind"], "tensor")
        self.assertEqual(
            packet_validator.validate(state_packet)["boundary_kind"],
            "state_transition",
        )
        self.assertEqual(
            state_packet["proof"]["target_before_state_digest"],
            state_packet["proof"]["target_after_state_digest"],
            "a skipped reload must remain a valid state-difference proof",
        )
        state_packet["proof"]["target_after_state_digest"] = state_packet["proof"]["expected_state_digest"]
        with self.assertRaises(packet_validator.PacketError):
            packet_validator.validate(state_packet)


class StateCardTests(unittest.TestCase):
    def test_builtin_state_card_is_resumable_and_valid(self) -> None:
        card = json.loads((TEMPLATES / "state_card.json").read_text(encoding="utf-8"))
        result = state_validator.validate(card)
        self.assertTrue(result["valid"])
        self.assertIn("requested_model_final", result["required_gates"])
        self.assertFalse(result["diagnostic_runtime_active"])

    def test_reopened_registry_requires_reason(self) -> None:
        card = json.loads((TEMPLATES / "state_card.json").read_text(encoding="utf-8"))
        card["state"]["registry_status"] = "reopened"
        card["state"]["reopen_reason"] = None
        with self.assertRaises(state_validator.StateCardError):
            state_validator.validate(card)

    def test_timed_state_event_updates_ledger(self) -> None:
        card = json.loads((TEMPLATES / "state_card.json").read_text(encoding="utf-8"))
        started = state_event.start_event(
            card,
            event_id="module-2",
            event="minimal_evidence_case",
            module=2,
            run_id="V001",
            evidence=[],
            next_gate="collect exact tensors",
        )
        finished = state_event.finish_event(
            card,
            event_id="module-2",
            status="pass",
            evidence=["discovery/epoch-001/manifest.json"],
            next_gate="discover first divergence",
        )
        state_validator.validate(card)
        self.assertEqual(started["event_id"], "module-2")
        self.assertEqual(finished["status"], "pass")
        self.assertGreaterEqual(finished["duration_seconds"], 0)

    def test_completed_truth_requires_evidence_and_confirmed_clean_state(self) -> None:
        card = json.loads((TEMPLATES / "state_card.json").read_text(encoding="utf-8"))
        card["truth"]["tensor_prefill"]["status"] = "exact"
        with self.assertRaises(state_validator.StateCardError):
            state_validator.validate(card)
        card["truth"]["tensor_prefill"]["evidence"] = ["discovery/exact.json"]
        card["state"]["task_status"] = "complete"
        with self.assertRaises(state_validator.StateCardError):
            state_validator.validate(card)

    def test_fully_evidenced_state_can_close(self) -> None:
        card = json.loads((TEMPLATES / "state_card.json").read_text(encoding="utf-8"))
        card["authorization"]["confirmed"] = True
        card["objective"].update(
            statement="Align the requested model end to end",
            requested_model="Qwen3.5-35B",
            requested_scope="requested training and rollout runtime",
        )
        card["environment"].update(
            execution_target="approved remote runtime",
            working_directory="approved worktree",
        )
        card["model"].update(
            source_model="approved model source",
            immutable_diagnostic_model="session/minimal-model",
        )
        card["artifacts"]["session_root"] = "alignment-session"
        for gate, (axis, status) in state_validator.GATE_TRUTH_EXPECTED.items():
            card["truth"][axis] = {
                "status": status,
                "evidence": [f"evidence/{gate}.json"],
            }
        card["runtime"]["clean_process_started"] = True
        card["state"]["task_status"] = "complete"
        result = state_validator.validate(card)
        self.assertTrue(result["completion_ready"])


class LongHorizonAuditTests(unittest.TestCase):
    def write_log(self, path: Path, *, steps: int, nonzero_step: int | None = None) -> None:
        lines = []
        for step in range(1, steps + 1):
            value = 0.125 if step == nonzero_step else 0.0
            lines.append(
                f"step:{step} training/rollout_probs_diff_mean: {value} "
                "training/rollout_probs_diff_valid: 1\n"
            )
        path.write_text("".join(lines), encoding="utf-8", newline="\n")

    def test_1000_exact_steps_cross_historical_failure_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            self.write_log(log, steps=1000)
            result = log_audit.audit(
                log,
                zero_metrics=["training/rollout_probs_diff_mean"],
                valid_metrics=["training/rollout_probs_diff_valid"],
                expected_steps=1000,
                expected_first_step=1,
                historical_failure_step=777,
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["observed_steps"], 1000)
        self.assertTrue(result["crossed_historical_failure_step"])
        self.assertEqual(
            result["metrics"]["training/rollout_probs_diff_mean"]["nonzero_count"],
            0,
        )

    def test_first_nonzero_emits_safe_trigger_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            trigger = root / "first_nonzero.json"
            self.write_log(log, steps=1000, nonzero_step=777)
            result = log_audit.audit(
                log,
                zero_metrics=["training/rollout_probs_diff_mean"],
                valid_metrics=["training/rollout_probs_diff_valid"],
                expected_steps=1000,
                expected_first_step=1,
                trigger_path=trigger,
            )
            trigger_value = json.loads(trigger.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["first_nonzero"]["step"], 777)
        self.assertEqual(trigger_value["failure"]["safe_excerpt"], "training/rollout_probs_diff_mean=0.125")
        self.assertTrue(trigger_value["capture_contract"]["prearmed_required"])
        self.assertFalse(
            trigger_value["capture_contract"]["already_lost_tensor_recovery_claimed"]
        )

    def test_incomplete_run_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            self.write_log(log, steps=999)
            result = log_audit.audit(
                log,
                zero_metrics=["training/rollout_probs_diff_mean"],
                valid_metrics=[],
                expected_steps=1000,
                expected_first_step=1,
            )
        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "fail")

    def test_discontinuous_steps_fail_without_explicit_first_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            steps = [*range(1, 501), *range(1001, 1501)]
            log.write_text(
                "".join(
                    f"step:{step} training/rollout_probs_diff_mean: 0\n"
                    for step in steps
                ),
                encoding="utf-8",
            )
            result = log_audit.audit(
                log,
                zero_metrics=["training/rollout_probs_diff_mean"],
                valid_metrics=[],
                expected_steps=1000,
                historical_failure_step=777,
            )
        self.assertEqual(result["status"], "fail")
        self.assertIn(501, result["violations"][-1]["missing_steps"])

    def test_live_heartbeat_timeout_is_an_error_not_eof(self) -> None:
        with mock.patch.object(log_audit, "process_alive", return_value=True):
            with self.assertRaises(log_audit.AuditError):
                log_audit.read_following(
                    io.StringIO(""),
                    follow_pid=123,
                    idle_timeout=0.001,
                    poll_seconds=0.001,
                )


class ProductionPatchAuditTests(unittest.TestCase):
    def test_clean_profile_and_patch_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "production.patch"
            patch.write_text(
                "diff --git a/model.py b/model.py\n+++ b/model.py\n+ALIGNMENT_EPSILON = 0.0\n",
                encoding="utf-8",
            )
            result = patch_validator.audit(
                patch,
                env={"TRUE_ON_POLICY_ALIGNMENT_PROFILE": "qwen35-v1"},
                profile={
                    "truth_role": "production_alignment_profile",
                    "active": True,
                    "features": ["aligned_lm_head"],
                    "diagnostic_capture_enabled": False,
                    "diagnostic_replay_enabled": False,
                    "activation_evidence": "fresh worker startup log",
                },
                require_clean_runtime=True,
            )
        self.assertEqual(result["status"], "pass")

    def test_diagnostic_capture_and_private_path_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "bad.patch"
            patch.write_text(
                "diff --git a/model.py b/model.py\n+++ b/model.py\n"
                "+capture = os.getenv('TRUE_ON_POLICY_PREFILL_TENSOR_CAPTURE_DIR')\n"
                "+torch.save(hidden, '/opt/private/hidden.pt')\n",
                encoding="utf-8",
            )
            result = patch_validator.audit(patch)
        self.assertEqual(result["status"], "fail")
        self.assertGreaterEqual(result["added_line_violations"], 3)

    def test_unquoted_secret_and_custom_fingerprint_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "bad.patch"
            patch.write_text(
                "diff --git a/model.py b/model.py\n+++ b/model.py\n"
                "+password=unquoted-secret\n"
                "+_write_model_weight_sync_fingerprints(worker)\n"
                "+MODEL=/opt/private/model\n",
                encoding="utf-8",
            )
            result = patch_validator.audit(patch)
        rules = {item["rule"] for item in result["violations"]}
        self.assertIn("credential_assignment", rules)
        self.assertIn("diagnostic_fingerprint", rules)
        self.assertIn("private_posix_path", rules)


class RemoteTransportTests(unittest.TestCase):
    def test_normalizes_crlf_and_builds_container_argv(self) -> None:
        result = transport.prepare(
            b"#!/bin/bash\r\necho ok\r\n",
            container="train_infer_ai",
            workdir="/workspace/verl",
            ssh_target="root@example.test",
        )
        self.assertEqual(result["line_endings"], "LF")
        self.assertTrue(result["workdir_fail_closed"])
        self.assertEqual(result["ssh_argv"][:3], ["ssh", "--", "root@example.test"])
        self.assertIn("docker exec -i train_infer_ai", result["remote_command"])
        encoded = result["remote_command"].split("printf %s ", 1)[1].split(
            " | base64 -d", 1
        )[0]
        decoded = base64.b64decode(encoded).decode("utf-8")
        self.assertTrue(decoded.startswith("cd -- /workspace/verl || exit 97\n"))

    def test_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(transport.TransportError):
            transport.prepare(b"password=do-not-store\n")


class ManagedJobTests(unittest.TestCase):
    def test_proc_stat_parser_handles_spaces_and_parentheses_in_comm(self) -> None:
        trailing = ["S", "1", "4321"] + ["0"] * 16 + ["987654"]
        pgid, start_ticks = managed_job.parse_proc_stat(
            "123 (worker name (x)) " + " ".join(trailing),
            123,
        )
        self.assertEqual(pgid, 4321)
        self.assertEqual(start_ticks, 987654)

    def test_invalid_smoke_pattern_fails_before_launch(self) -> None:
        with self.assertRaises(managed_job.JobError):
            managed_job.compile_smoke_pattern("[")


class PublicPackageTests(unittest.TestCase):
    def test_allowlisted_build_and_hash_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "report.md").write_text("# 脱敏报告\n", encoding="utf-8")
            allowlist = root / "allowlist.json"
            allowlist.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [
                            {"source": "report.md", "destination": "reports/report.md"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "public"
            built = public_package.build(
                source,
                allowlist,
                output,
                max_file_bytes=1024,
            )
            verified = public_package.verify(output, output / "sha256_manifest.json")
            (output / "unexpected.txt").write_text("extra", encoding="utf-8")
            contaminated = public_package.verify(output, output / "sha256_manifest.json")
        self.assertEqual(built["status"], "pass")
        self.assertEqual(verified["status"], "pass")
        self.assertEqual(contaminated["status"], "fail")
        self.assertEqual(contaminated["unexpected_files"], ["unexpected.txt"])

    def test_private_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "report.md").write_text("server 192.168.1.2\n", encoding="utf-8")
            allowlist = root / "allowlist.json"
            allowlist.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [{"source": "report.md"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(public_package.PackageError):
                public_package.build(
                    source,
                    allowlist,
                    root / "public",
                    max_file_bytes=1024,
                )

    def test_empty_manifest_cannot_hide_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "secret.pt").write_bytes(b"raw")
            manifest = root / "sha256_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "truth_role": "sanitized_public_package_manifest",
                        "file_count": 0,
                        "files": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(public_package.PackageError):
                public_package.verify(root, manifest)

    def test_unquoted_credential_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "report.md").write_text(
                "password=do-not-publish\n", encoding="utf-8"
            )
            allowlist = root / "allowlist.json"
            allowlist.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [{"source": "report.md"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(public_package.PackageError):
                public_package.build(
                    source,
                    allowlist,
                    root / "public",
                    max_file_bytes=1024,
                )


class CasebookWritePolicyTests(unittest.TestCase):
    def test_add_requires_explicit_maintainer_confirmation(self) -> None:
        parser = fix_casebook.build_parser()
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            parser.parse_args(
                [
                    "add",
                    "--case",
                    "verified_fix_case.json",
                    "--casebook-root",
                    "references/fix-cases",
                ]
            )

    def test_programmatic_add_refuses_unconfirmed_write(self) -> None:
        args = type(
            "Args",
            (),
            {
                "confirm_maintainer_update": False,
                "case": Path("missing-case.json"),
                "casebook_root": Path("references/fix-cases"),
            },
        )()
        with self.assertRaises(fix_casebook.CaseError):
            fix_casebook.command_add(args)


class ExtendedReportTests(unittest.TestCase):
    def passing_matrix_entry(self, gate: str = "long_horizon_e2e") -> dict[str, object]:
        return {
            "run_id": "V001",
            "label": "验证运行",
            "gate": gate,
            "model_scope": "声明范围",
            "response_length": 1,
            "expected_steps": 1000,
            "observed_steps": 1000,
            "complete": True,
            "metric_name": "training/rollout_probs_diff_mean",
            "max_abs_diff": 0,
            "nonzero_count": 0,
            "nonfinite_count": 0,
            "crossed_historical_failure_step": True,
            "status": "pass",
            "evidence": "runs/V001/audit.json",
        }

    def test_template_and_validation_matrix_render_in_all_views(self) -> None:
        source = json.loads(
            (TEMPLATES / "alignment_report_source.zh-CN.json").read_text(encoding="utf-8")
        )
        source["validation_matrix"]["entries"] = [self.passing_matrix_entry()]
        source["results"].update(e2e_metric_status="zero", e2e_metric_value=0)
        source["verification_axes"]["long_horizon_e2e"] = {
            "status": "pass",
            "evidence": "runs/V001/audit.json",
        }
        source["acceptance_contract"]["historical_failure_step"] = 777
        source["production_alignment_profile"] = {
            "truth_role": "production_alignment_profile",
            "active": True,
            "features": ["aligned_lm_head"],
            "diagnostic_replay_enabled": False,
            "diagnostic_capture_enabled": False,
            "activation_evidence": "runs/clean/profile.json",
        }
        source["clean_run_gate"] = {
            "truth_role": "production_activation_only",
            "fresh_process": True,
            "fresh_weights_loaded": True,
            "diagnostic_replay_disabled": True,
            "diagnostic_capture_disabled": True,
            "activation_proved": True,
            "evidence": "runs/clean/audit.json",
        }
        report.validate(source)
        txt = report.render_txt(source)
        md = report.render_md(source)
        rendered_html = report.render_html(source)
        for rendered in (txt, md, rendered_html):
            self.assertIn("runs/V001/audit.json", rendered)
            self.assertIn("777", rendered)
            self.assertIn("runs/clean/profile.json", rendered)
            self.assertIn("runs/clean/audit.json", rendered)
        self.assertIn("【多维真值状态】", txt)
        self.assertIn("## 分层验证矩阵", md)
        self.assertIn("<h2>目标与验收契约</h2>", rendered_html)

    def test_acceptance_pass_fails_closed_when_required_gate_is_missing(self) -> None:
        source = json.loads(
            (TEMPLATES / "alignment_report_source.zh-CN.json").read_text(encoding="utf-8")
        )
        source["acceptance_contract"]["required_gates"] = ["long_horizon_e2e"]
        source["acceptance_contract"]["status"] = "pass"
        with self.assertRaises(report.ReportError):
            report.validate(source)
        source["validation_matrix"]["entries"] = [self.passing_matrix_entry()]
        source["results"].update(e2e_metric_status="zero", e2e_metric_value=0)
        source["verification_axes"]["long_horizon_e2e"] = {
            "status": "pass",
            "evidence": "runs/V001/audit.json",
        }
        report.validate(source)

    def test_zero_step_or_wrong_final_scope_cannot_close_acceptance(self) -> None:
        source = json.loads(
            (TEMPLATES / "alignment_report_source.zh-CN.json").read_text(encoding="utf-8")
        )
        source["acceptance_contract"].update(
            status="pass",
            requested_model_scope="Qwen3.5-35B requested runtime",
            required_gates=["requested_model_final"],
            long_horizon_required=False,
        )
        source["results"].update(e2e_metric_status="zero", e2e_metric_value=0)
        source["verification_axes"]["requested_model_final"] = {
            "status": "pass",
            "evidence": "runs/final/audit.json",
        }
        entry = self.passing_matrix_entry("requested_model_final")
        entry["model_scope"] = "one-layer diagnostic model"
        entry["expected_steps"] = 0
        entry["observed_steps"] = 0
        source["validation_matrix"]["entries"] = [entry]
        with self.assertRaises(report.ReportError):
            report.validate(source)
        entry["expected_steps"] = 1000
        entry["observed_steps"] = 1000
        with self.assertRaises(report.ReportError):
            report.validate(source)
        entry["model_scope"] = "Qwen3.5-35B requested runtime"
        report.validate(source)

    def test_axis_and_headline_contradiction_fails(self) -> None:
        source = json.loads(
            (TEMPLATES / "alignment_report_source.zh-CN.json").read_text(encoding="utf-8")
        )
        source["verification_axes"]["tensor_prefill"] = {
            "status": "exact",
            "evidence": "discovery/exact.json",
        }
        source["results"]["native_prefill_status"] = "not_exact"
        with self.assertRaises(report.ReportError):
            report.validate(source)

    def test_resolved_difference_requires_and_renders_before_after(self) -> None:
        source = json.loads(
            (TEMPLATES / "alignment_report_source.zh-CN.json").read_text(encoding="utf-8")
        )
        exact_gate = {
            "same_shape": True,
            "same_dtype": True,
            "no_nonfinite": True,
            "torch_equal": True,
            "downstream_checked": True,
            "real_path_activation_proved": True,
            "max_abs_diff": 0,
        }
        source["differences"] = [
            {
                "diff_id": "D001",
                "semantic_boundary": "model.layer0.output",
                "observation": "native",
                "root_cause_class": "Op path",
                "root_cause_confidence": "CONFIRMED",
                "status": "resolved",
                "max_abs_diff": 0.25,
                "discovery_max_abs_diff": 0.25,
                "verification_max_abs_diff": 0,
                "upstream_replay_ids": [],
                "summary": "发现并修复算子路径差异。",
                "proof": "精确 Tensor 证据。",
                "repair_result": "已修复。",
                "replay_summary": "未使用 Replay。",
                "verification": "真实路径精确。",
                "verification_gate": exact_gate,
            }
        ]
        report.validate(source)
        self.assertIn("0.25 → 0", report.render_md(source))
        source["differences"][0]["verification_max_abs_diff"] = 0.1
        with self.assertRaises(report.ReportError):
            report.validate(source)


if __name__ == "__main__":
    unittest.main()
