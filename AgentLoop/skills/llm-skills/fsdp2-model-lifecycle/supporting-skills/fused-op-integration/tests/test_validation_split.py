import importlib.util
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = SKILL_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE = load_module("compare_fusion_runs", "scripts/compare_fusion_runs.py")
MARKDOWN = load_module("update_markdown_report", "scripts/update_markdown_report.py")


class PerformanceComparisonTest(unittest.TestCase):
    def test_log_parser_does_not_require_loss(self):
        log = (
            "iteration 1/2 | elapsed time per iteration (ms): 100.0 | "
            "max_memory_allocated(GB): 10.0 | max_memory_reserved(GB): 12.0\n"
            "iteration 2/2 | elapsed time per iteration (ms): 90.0 | "
            "max_memory_allocated(GB): 10.5 | max_memory_reserved(GB): 12.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.log"
            path.write_text(log, encoding="utf-8")
            total, records = COMPARE.parse_log(path)

        self.assertEqual(total, 2)
        self.assertNotIn("lm_loss", records[1])

    def test_comparison_contains_only_performance_metrics(self):
        baseline = {
            1: {
                "elapsed_time_ms": 100.0,
                "max_memory_allocated_gb": 10.0,
                "max_memory_reserved_gb": 12.0,
            },
            2: {
                "elapsed_time_ms": 100.0,
                "max_memory_allocated_gb": 10.0,
                "max_memory_reserved_gb": 12.0,
            },
        }
        candidate = {
            1: {
                "elapsed_time_ms": 90.0,
                "max_memory_allocated_gb": 10.0,
                "max_memory_reserved_gb": 12.0,
            },
            2: {
                "elapsed_time_ms": 90.0,
                "max_memory_allocated_gb": 10.0,
                "max_memory_reserved_gb": 12.0,
            },
        }

        comparison = COMPARE.compare(2, baseline, 2, candidate, 0)

        self.assertNotIn("loss", comparison)
        self.assertAlmostEqual(comparison["delta"]["speedup_percent"], 10.0)


class MarkdownReportTest(unittest.TestCase):
    def test_uses_external_accuracy_report(self):
        performance_report = {
            "status": "success",
            "comparison": {
                "baseline_metrics": {"mean_step_time_ms": 100.0},
                "candidate_metrics": {"mean_step_time_ms": 90.0},
                "delta": {"speedup_percent": 10.0},
            },
        }
        accuracy_report = {
            "workflow": "mindspeed-llm-loss-accuracy-validation",
            "status": "success",
            "threshold": {"max_relative_error_percent": 1.0},
            "comparison": {
                "mean_relative_absolute_error_percent": 0.2,
                "max_relative_absolute_error_percent": 0.65,
            },
        }

        step_time, performance, performance_status = (
            MARKDOWN.build_performance_metrics(performance_report)
        )
        accuracy, accuracy_status = MARKDOWN.build_accuracy_metrics(accuracy_report)
        correctness_report = {
            "workflow": "mindspeed-llm-fsdp2-fusion-correctness",
            "status": "success",
            "gates": {
                "requested_active": "passed",
                "structure_signature": "passed",
                "semantic_probes": "passed",
                "logits_precision": "passed",
            },
        }
        correctness, correctness_status = MARKDOWN.build_correctness_metrics(
            correctness_report
        )

        self.assertEqual(step_time, "100.00 → 90.00 ms")
        self.assertEqual(performance, "+10.00%（加速）")
        self.assertEqual(performance_status, "success")
        self.assertIn("MeanRelAbs=0.2%", accuracy)
        self.assertIn("MaxRelAbs=0.65%", accuracy)
        self.assertEqual(accuracy_status, "success")
        self.assertIn("requested_active=passed", correctness)
        self.assertIn("logits_precision=passed", correctness)
        self.assertEqual(correctness_status, "success")

    def test_correctness_status_must_match_gates(self):
        report = {
            "workflow": "mindspeed-llm-fsdp2-fusion-correctness",
            "status": "success",
            "gates": {
                "requested_active": "passed",
                "structure_signature": "failed",
                "semantic_probes": "passed",
                "logits_precision": "passed",
            },
        }

        with self.assertRaises(MARKDOWN.ReportInputError):
            MARKDOWN.build_correctness_metrics(report)


if __name__ == "__main__":
    unittest.main()
