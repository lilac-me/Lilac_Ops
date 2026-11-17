from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "scorer"))

import score  # noqa: E402

MIN_ENV = {
    "mode": "static",
    "hardware": {"chip": "910B", "cpu_arch": "aarch64", "os": "Linux"},
    "software": {"python": "3.10", "veomni": "0.1", "pytorch": "2.1",
                  "torchnpu": "6.0", "cann": "8.0", "transformers": "4.46"},
    "compatibility": [{"component": "cann", "status": "OK", "note": ""}],
    "issues": [],
    "suggestions": ["s1", "s2", "s3"],
}

MIN_ACCEPTANCE = {
    "model": "toy-model",
    "stages": [
        {"stage": "environment", "status": "PASSED", "artifacts": ["environment_report.json"]},
        {"stage": "delivery", "status": "PASSED", "artifacts": ["delivery_coverage_report.md", "migration_process_record.md"]},
    ],
    "regression": {"passed": True, "scope": "unit test"},
    "limitations": [],
    "reproduce_commands": ["python -m unittest"],
    "conclusion": "PASS",
}


def write_env(base: Path, payload) -> None:
    (base / "environment_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class StageTableTests(unittest.TestCase):
    def test_nine_stages_cover_pipeline(self) -> None:
        names = [n for n, _ in score.STAGES]
        self.assertEqual(len(names), 9)
        self.assertEqual(names[0], "environment")
        self.assertEqual(names[-1], "delivery")

    def test_stage_index_covers_all(self) -> None:
        for name, artifacts in score.STAGES:
            self.assertIn(name, score.STAGE_INDEX)
            self.assertTrue(artifacts)


class CheckStageTests(unittest.TestCase):
    def test_all_missing_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = score.check_stage(Path(tmp), score.STAGE_INDEX["environment"])
            self.assertEqual(result["status"], "PENDING")
            self.assertEqual(result["artifacts"][0]["status"], "MISSING")

    def test_too_small_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_env(base, {"mode": "static"})  # 过小且不合规
            result = score.check_stage(base, score.STAGE_INDEX["environment"])
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["artifacts"][0]["status"], "TOO_SMALL")

    def test_schema_invalid_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            payload = dict(MIN_ENV)
            payload["mode"] = "npu"  # 合法大小但缺 issues 之外还破坏 required
            del payload["compatibility"]
            write_env(base, payload)
            result = score.check_stage(base, score.STAGE_INDEX["environment"])
            self.assertEqual(result["status"], "FAILED")
            entry = result["artifacts"][0]
            self.assertEqual(entry["status"], "SCHEMA_INVALID")
            self.assertTrue(entry["errors"])

    def test_valid_artifact_is_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_env(base, MIN_ENV)
            result = score.check_stage(base, score.STAGE_INDEX["environment"])
            self.assertEqual(result["status"], "PASSED")
            self.assertEqual(result["artifacts"][0]["status"], "OK")

    def test_delivery_requires_acceptance_contract_and_two_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "acceptance_report.json").write_text(
                json.dumps(MIN_ACCEPTANCE, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (base / "delivery_coverage_report.md").write_text("x" * 800, encoding="utf-8")
            (base / "migration_process_record.md").write_text("x" * 600, encoding="utf-8")
            result = score.check_stage(base, score.STAGE_INDEX["delivery"])
            self.assertEqual(result["status"], "PASSED")


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML 未安装")
class LoadScenarioStagesTests(unittest.TestCase):
    def test_omni_declares_five_stages(self) -> None:
        names = score.load_scenario_stages(BENCH / "scenarios" / "omni-qwen3-omni.yaml")
        self.assertEqual(names, ["environment", "model-analysis", "adaptation",
                                 "functional-validation", "precision-alignment", "delivery"])

    def test_dense_declares_full_pipeline(self) -> None:
        names = score.load_scenario_stages(BENCH / "scenarios" / "dense-qwen3.yaml")
        self.assertEqual(names, [n for n, _ in score.STAGES])

    def test_exception_cases_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score.load_scenario_stages(BENCH / "scenarios" / "exception-cases.yaml")

    def test_unknown_stage_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text("stages:\n  - environment: {}\n  - not-a-stage: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                score.load_scenario_stages(p)


if __name__ == "__main__":
    unittest.main()
