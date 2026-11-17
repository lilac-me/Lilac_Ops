from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_alignment_report as report  # noqa: E402
import triage_module_artifacts as triage  # noqa: E402
import validate_semantic_mapping as mapping_validator  # noqa: E402


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(key: str) -> dict[str, object]:
    return {"key": key, "selector": "$", "call_index": 0, "transforms": []}


def mapping_entry(
    mapping_id: str,
    order: int,
    relation: str = "1:1",
) -> dict[str, object]:
    train_count, infer_count = {
        "1:1": (1, 1),
        "1:N": (1, 2),
        "N:1": (2, 1),
    }[relation]
    return {
        "mapping_id": mapping_id,
        "semantic_node": f"model.{mapping_id}.output",
        "relation": relation,
        "execution_order": order,
        "execution_order_evidence": {
            "kind": "runtime_sequence",
            "detail": f"captured runtime order {order}",
        },
        "tensor_roles": ["input", "parameter", "output"],
        "train": [endpoint(f"train.{mapping_id}.{index}") for index in range(train_count)],
        "infer": [endpoint(f"infer.{mapping_id}.{index}") for index in range(infer_count)],
        "semantic_basis": f"source-backed semantic mapping for {mapping_id}",
        "confidence": "high",
    }


def exact_record(
    mapping_id: str,
    role: str,
    *,
    equal: bool,
    train_dtype: str = "torch.float32",
    infer_dtype: str = "torch.float32",
    train_shape: list[int] | None = None,
    infer_shape: list[int] | None = None,
) -> dict[str, object]:
    train_shape = train_shape or [2, 2]
    infer_shape = infer_shape or [2, 2]
    same_shape = train_shape == infer_shape
    same_dtype = train_dtype == infer_dtype
    can_be_equal = equal and same_shape and same_dtype
    return {
        "mapping_id": mapping_id,
        "role": role,
        "train": {
            "present": True,
            "dtype": train_dtype,
            "shape": train_shape,
            "statistics": {"max": 1.0, "min": -1.0, "mean": 0.0, "norm": 2.0},
        },
        "infer": {
            "present": True,
            "dtype": infer_dtype,
            "shape": infer_shape,
            "statistics": {"max": 1.0, "min": -1.0, "mean": 0.0, "norm": 2.0},
        },
        "comparison": {
            "evidence_kind": "exact_tensor",
            "same_shape": same_shape,
            "same_dtype": same_dtype,
            "no_nonfinite": True,
            "torch_equal": can_be_equal,
            "max_abs_diff": 0.0 if equal else 0.25,
            "basis": "raw Tensor comparison",
        },
    }


def statistics_record(
    mapping_id: str, role: str, relation: str
) -> dict[str, object]:
    return {
        "mapping_id": mapping_id,
        "role": role,
        "train": {
            "present": True,
            "dtype": "torch.float32",
            "shape": [2, 2],
            "statistics": {"max": 1.0, "min": -1.0, "mean": 0.0, "norm": 2.0},
        },
        "infer": {
            "present": True,
            "dtype": "torch.float32",
            "shape": [2, 2],
            "statistics": {"max": 1.0, "min": -1.0, "mean": 0.0, "norm": 2.0},
        },
        "comparison": {
            "evidence_kind": "statistics",
            "summary_relation": relation,
            "basis": "summary fields only",
        },
    }


class SemanticMappingTests(unittest.TestCase):
    def test_accepts_supported_cardinalities(self) -> None:
        value = {
            "schema_version": 1,
            "phase": "prefill",
            "execution_order_basis": "runtime call sequence",
            "source_hashes": {},
            "mappings": [
                mapping_entry("M001", 10, "1:1"),
                mapping_entry("M002", 20, "1:N"),
                mapping_entry("M003", 30, "N:1"),
            ],
        }
        result = mapping_validator.validate(value)
        self.assertEqual(result["relation_counts"], {"1:1": 1, "1:N": 1, "N:1": 1})

    def test_rejects_json_insertion_order(self) -> None:
        value = {
            "schema_version": 1,
            "phase": "prefill",
            "execution_order_basis": "json_insertion_order",
            "source_hashes": {},
            "mappings": [mapping_entry("M001", 10)],
        }
        with self.assertRaises(mapping_validator.MappingError):
            mapping_validator.validate(value)

    def test_allows_endpoint_reuse_across_semantic_mappings(self) -> None:
        first = mapping_entry("M001", 10)
        second = mapping_entry("M002", 20)
        second["train"] = first["train"]
        second["infer"] = first["infer"]
        value = {
            "schema_version": 1,
            "phase": "prefill",
            "execution_order_basis": "runtime call sequence",
            "source_hashes": {},
            "mappings": [first, second],
        }
        self.assertTrue(mapping_validator.validate(value)["valid"])


class StructuralTriageTests(unittest.TestCase):
    def build_case(self, root: Path) -> tuple[dict[str, object], dict[str, object], Path, Path]:
        mapping_value = {
            "schema_version": 1,
            "phase": "prefill",
            "execution_order_basis": "runtime call sequence",
            "source_hashes": {},
            "mappings": [
                mapping_entry("M001", 10, "1:1"),
                {
                    **mapping_entry("M002", 20, "1:N"),
                    "structural_relation": {
                        "kind": "fusion_split",
                        "basis": "source shows a fused inference implementation",
                    },
                },
                mapping_entry("M003", 30, "N:1"),
                mapping_entry("M004", 40, "1:1"),
                mapping_entry("M005", 50, "1:1"),
                mapping_entry("M006", 60, "1:1"),
            ],
        }
        mapping_path = root / "semantic_mapping.json"
        mapping_path.write_text(json.dumps(mapping_value), encoding="utf-8")
        observations_value = {
            "schema_version": 1,
            "phase": "prefill",
            "mapping_file_sha256": digest(mapping_path),
            "input_hashes": {
                "train_artifacts": "sha256:" + "1" * 64,
                "infer_artifacts": "sha256:" + "2" * 64,
            },
            "records": [
                exact_record("M001", "input", equal=True),
                exact_record("M001", "parameter", equal=False),
                exact_record("M001", "output", equal=True),
                statistics_record("M002", "input", "same"),
                {
                    "mapping_id": "M002",
                    "role": "parameter",
                    "train": {"present": False, "missing_reason": "not captured"},
                    "infer": {"present": False, "missing_reason": "not captured"},
                    "comparison": {
                        "evidence_kind": "metadata",
                        "basis": "parameter capture unavailable",
                    },
                },
                statistics_record("M002", "output", "different"),
                exact_record("M003", "input", equal=True),
                exact_record("M003", "parameter", equal=True),
                exact_record("M003", "output", equal=False),
                statistics_record("M004", "output", "same"),
                exact_record(
                    "M005",
                    "output",
                    equal=True,
                    train_dtype="torch.float32",
                    infer_dtype="torch.bfloat16",
                ),
                exact_record(
                    "M006",
                    "output",
                    equal=True,
                    train_shape=[2, 2],
                    infer_shape=[4],
                ),
            ],
            "unmatched_nodes": [
                {
                    "side": "infer",
                    "key": "infer.extra_activation",
                    "execution_order": 15,
                    "semantic_hint": "activation",
                    "source_evidence": "runtime call index",
                }
            ],
            "structural_hints": [
                {
                    "hint_id": "S001",
                    "mapping_id": "M002",
                    "kind": "fusion_split",
                    "basis": "source evidence",
                }
            ],
        }
        observations_path = root / "triage_observations.json"
        observations_path.write_text(json.dumps(observations_value), encoding="utf-8")
        return mapping_value, observations_value, mapping_path, observations_path

    def test_ranking_does_not_replace_coverage_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_value, observations_value, mapping_path, observations_path = (
                self.build_case(root)
            )
            mapping_validator.validate(mapping_value)
            normalized = triage.validate_observations(
                observations_value,
                {entry["mapping_id"] for entry in mapping_value["mappings"]},
                digest(mapping_path),
            )
            result = triage.build_triage(
                mapping_value, normalized, mapping_path, observations_path
            )

        self.assertEqual(
            [item["mapping_id"] for item in result["coverage_order"]],
            ["M001", "M002", "M003", "M004", "M005", "M006"],
        )
        self.assertEqual(result["candidates"][0]["mapping_id"], "M003")
        self.assertTrue(result["candidate_ranking_must_not_define_first_divergence"])
        self.assertFalse(result["may_create_difference_packet"])
        self.assertTrue(
            result["candidates"][0]["previous_exact_boundary_is_not_exact_prefix"]
        )

        m002 = next(
            item for item in result["candidates"] if item["mapping_id"] == "M002"
        )
        self.assertNotEqual(m002["classification"], "parameter_or_checkpoint_issue")
        self.assertEqual(m002["evidence_strength"], "summary_only")

        m004_gate = next(
            item
            for item in result["exact_verification_required"]
            if item["mapping_id"] == "M004"
        )
        self.assertEqual(m004_gate["observed_state"], "statistics_same")
        self.assertTrue(
            any(item["mapping_id"] == "M001" for item in result["excluded_candidates"])
        )
        self.assertTrue(
            any(
                item["mapping_id"] == "M002"
                for item in result["structural_false_positive_candidates"]
            )
        )
        m005 = next(
            item for item in result["candidates"] if item["mapping_id"] == "M005"
        )
        self.assertEqual(m005["observed_output_state"], "dtype_mismatch")
        m006 = next(
            item for item in result["candidates"] if item["mapping_id"] == "M006"
        )
        self.assertEqual(m006["observed_output_state"], "shape_mismatch")

    def test_cli_writes_current_round_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, mapping_path, observations_path = self.build_case(root)
            output = root / "structural_triage.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "triage_module_artifacts.py"),
                    "--mapping",
                    str(mapping_path),
                    "--observations",
                    str(observations_path),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["truth_role"], "triage_only")
            self.assertEqual(value["source"]["semantic_mapping_sha256"], digest(mapping_path))

    def test_rejects_stale_mapping_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_value, observations_value, mapping_path, _ = self.build_case(root)
            observations_value["mapping_file_sha256"] = "sha256:" + "0" * 64
            with self.assertRaises(triage.TriageError):
                triage.validate_observations(
                    observations_value,
                    {entry["mapping_id"] for entry in mapping_value["mappings"]},
                    digest(mapping_path),
                )

    def test_rejects_stale_current_case_hash(self) -> None:
        observed = {"train_artifacts": "sha256:" + "1" * 64}
        expected = triage.parse_expected_hashes(
            ["train_artifacts=sha256:" + "2" * 64]
        )
        with self.assertRaises(triage.TriageError):
            triage.verify_expected_hashes(observed, expected)

    def test_unmatched_node_needs_an_exact_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_value = {
                "schema_version": 1,
                "phase": "prefill",
                "execution_order_basis": "runtime call sequence",
                "source_hashes": {},
                "mappings": [mapping_entry("M001", 10)],
            }
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(mapping_value), encoding="utf-8")
            observations_value = {
                "schema_version": 1,
                "phase": "prefill",
                "mapping_file_sha256": digest(mapping_path),
                "input_hashes": {"artifacts": "sha256:" + "1" * 64},
                "records": [statistics_record("M001", "output", "different")],
                "unmatched_nodes": [
                    {
                        "side": "infer",
                        "key": "infer.extra",
                        "execution_order": 5,
                        "semantic_hint": "unknown",
                        "source_evidence": "runtime call index",
                    }
                ],
                "structural_hints": [],
            }
            observations_path = root / "observations.json"
            observations_path.write_text(json.dumps(observations_value), encoding="utf-8")
            normalized = triage.validate_observations(
                observations_value, {"M001"}, digest(mapping_path)
            )
            result = triage.build_triage(
                mapping_value, normalized, mapping_path, observations_path
            )
        self.assertNotEqual(
            result["candidates"][0]["classification"],
            "missing_or_extra_op_between_exact_boundary",
        )


class ReportCompatibilityTests(unittest.TestCase):
    def base_source(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "language": "zh-CN",
            "title": "合成测试报告",
            "scope": {
                "phase": "prefill",
                "model": "合成模型",
                "coverage": "已声明的测试边界",
                "training_backend": "custom",
                "inference_backend": "custom",
                "hardware_backend": "cpu",
                "evidence_provider": "已保存工件",
                "topology": {"tp": 1, "pp": 1, "ep": 1},
            },
            "results": {
                "native_prefill_status": "not_run",
                "conditional_prefill_status": "not_run",
                "unresolved_replay_ids": [],
                "e2e_metric_status": "not_run",
                "e2e_metric_value": None,
            },
            "differences": [],
        }

    def test_old_report_without_triage_remains_valid(self) -> None:
        source = self.base_source()
        report.validate(source)
        self.assertIn("【结论】", report.render_txt(source))
        self.assertNotIn("【结构预分析", report.render_txt(source))
        self.assertIn("## 结论概览", report.render_md(source))
        self.assertIn("<html lang=\"zh-CN\">", report.render_html(source))

    def test_report_requires_chinese_source_language(self) -> None:
        source = self.base_source()
        source["language"] = "en"
        with self.assertRaises(report.ReportError):
            report.validate(source)

    def test_metric_progression_is_validated_and_rendered(self) -> None:
        source = self.base_source()
        source["results"].update(
            {
                "e2e_metric_name": "training/rollout_probs_diff_mean",
                "e2e_metric_status": "nonzero",
                "e2e_metric_value": 0.25,
            }
        )
        source["metric_progression"] = {
            "metric_name": "training/rollout_probs_diff_mean",
            "truth_role": "e2e_impact_only",
            "headline_stage_id": "P001",
            "evidence": "用户提供的运行数值",
            "note": "该指标不能替代 Prefill Tensor 证明。",
            "entries": [
                {
                    "stage_id": "P001",
                    "label": "全量 rollout step 1",
                    "run_scope": "full_rollout",
                    "step": 1,
                    "value": 0.25,
                    "related_difference_ids": [],
                }
            ],
        }
        report.validate(source)
        self.assertIn("【E2E 指标演进｜", report.render_txt(source))
        self.assertIn("## E2E 指标演进：", report.render_md(source))
        self.assertIn("<h2>E2E 指标演进：", report.render_html(source))
        self.assertEqual(report.metric_value_text(0.0000143852), "0.0000143852")

        source["results"]["e2e_metric_value"] = 0.5
        with self.assertRaises(report.ReportError):
            report.validate(source)

    def test_cli_initializes_source_from_builtin_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "report_source.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "generate_alignment_report.py"),
                    "--init-source",
                    str(source_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(source["language"], "zh-CN")
            self.assertEqual(source["scope"]["phase"], "prefill")
            self.assertEqual(source["scope"]["topology"]["tp"], 1)
            report.validate(source)

    def test_triage_section_is_non_authoritative_and_rendered(self) -> None:
        source = self.base_source()
        source["triage"] = {
            "truth_role": "triage_only",
            "status": "partial",
            "may_create_difference_packet": False,
            "candidate_ranking_must_not_define_first_divergence": True,
            "exact_comparison_must_follow_semantic_execution_order": True,
            "candidates": [
                {
                    "candidate_id": "C001",
                    "mapping_id": "M002",
                    "priority_rank": 1,
                    "semantic_node": "model.layer0.output",
                    "observed_output_state": "statistics_difference",
                    "classification": "needs_exact_input_evidence",
                    "formal_verification_required": True,
                    "previous_exact_boundary": None,
                }
            ],
            "structural_false_positive_candidates": [],
            "excluded_candidates": [],
            "next_checks": ["run exact comparison in coverage order"],
        }
        report.validate(source)
        self.assertIn("结构预分析（非正式结论）", report.render_md(source))
        self.assertIn("【结构预分析（非正式结论）】", report.render_txt(source))
        self.assertIn("结构预分析", report.render_html(source))

        source["triage"]["may_create_difference_packet"] = True
        with self.assertRaises(report.ReportError):
            report.validate(source)

    def test_report_cli_writes_all_views_with_triage(self) -> None:
        source = self.base_source()
        source["triage"] = {
            "truth_role": "triage_only",
            "status": "complete",
            "may_create_difference_packet": False,
            "candidate_ranking_must_not_define_first_divergence": True,
            "exact_comparison_must_follow_semantic_execution_order": True,
            "candidates": [],
            "structural_false_positive_candidates": [],
            "excluded_candidates": [],
            "next_checks": ["continue exact comparison in semantic execution order"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "report_source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            output = root / "reports"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "generate_alignment_report.py"),
                    "--input",
                    str(source_path),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "alignment_report.html",
                    "alignment_report.json",
                    "alignment_report.md",
                    "alignment_summary.txt",
                },
            )
            self.assertIn(
                "【结论】",
                (output / "alignment_summary.txt").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "## 差异总览",
                (output / "alignment_report.md").read_text(encoding="utf-8"),
            )
            html_text = (output / "alignment_report.html").read_text(encoding="utf-8")
            self.assertIn("<html lang=\"zh-CN\">", html_text)
            self.assertIn("<h2>差异总览</h2>", html_text)

    def test_terminal_report_wraps_by_display_width(self) -> None:
        source = self.base_source()
        source["scope"]["coverage"] = "中文范围" * 80
        rendered = report.render_txt(source)
        self.assertLessEqual(
            max(report.display_width(line) for line in rendered.splitlines()),
            96,
        )

    def test_markdown_and_html_keep_long_narrative_out_of_summary_table(self) -> None:
        source = self.base_source()
        source["differences"] = [
            {
                "diff_id": "D001",
                "semantic_boundary": "model.layer0.output",
                "observation": "native",
                "root_cause_class": "Op path",
                "root_cause_confidence": "SUPPORTED",
                "status": "pending",
                "max_abs_diff": 0.25,
                "upstream_replay_ids": [],
                "summary": "这是用于验证长段落不会挤进总览表格的摘要。" * 10,
                "proof": "精确 Tensor 证据。",
                "repair_result": "待验证候选。",
                "replay_summary": "未使用 Replay。",
                "verification": "待真实路径验证。",
            }
        ]
        report.validate(source)
        markdown = report.render_md(source)
        html_text = report.render_html(source)
        self.assertIn(
            "| ID | 语义边界 | 观测条件 | 根因 | 状态 | 最大绝对差 |",
            markdown,
        )
        self.assertNotIn("| ID | 语义边界 | 摘要 |", markdown)
        self.assertIn("### 一句话摘要", markdown)
        self.assertEqual(html_text.count("<details class=\"card difference-card\" open"), 1)
        self.assertIn("@media(max-width:620px)", html_text)
        self.assertNotIn("Difference summary", html_text)


if __name__ == "__main__":
    unittest.main()
