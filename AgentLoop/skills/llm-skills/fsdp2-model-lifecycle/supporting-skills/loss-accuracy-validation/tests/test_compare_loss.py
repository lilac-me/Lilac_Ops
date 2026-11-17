import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_loss.py"
SPEC = importlib.util.spec_from_file_location("compare_loss", SCRIPT_PATH)
COMPARE_LOSS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE_LOSS)


class CompareSeriesTest(unittest.TestCase):
    def test_mean_relative_absolute_error(self):
        result = COMPARE_LOSS.compare_series(
            2,
            {1: 10.0, 2: 20.0},
            2,
            {1: 10.1, 2: 19.6},
        )

        self.assertAlmostEqual(result["mean_relative_absolute_error"], 0.015)
        self.assertAlmostEqual(result["mean_relative_absolute_error_percent"], 1.5)
        self.assertAlmostEqual(result["max_relative_absolute_error"], 0.02)
        self.assertAlmostEqual(result["max_relative_absolute_error_percent"], 2.0)
        self.assertAlmostEqual(
            result["per_iteration"][0]["relative_absolute_error_percent"],
            1.0,
        )

    def test_compares_common_prefix_when_candidate_has_more_steps(self):
        result = COMPARE_LOSS.compare_series(
            2,
            {1: 10.0, 2: 20.0},
            3,
            {1: 10.1, 2: 19.6, 3: 30.0},
        )

        self.assertEqual(result["compared_points"], 2)
        self.assertEqual(result["compared_iteration_start"], 1)
        self.assertEqual(result["compared_iteration_end"], 2)
        self.assertEqual(result["baseline_total_training_steps"], 2)
        self.assertEqual(result["candidate_total_training_steps"], 3)
        self.assertEqual(
            [item["iteration"] for item in result["per_iteration"]], [1, 2]
        )

    def test_compares_common_prefix_when_baseline_has_more_steps(self):
        result = COMPARE_LOSS.compare_series(
            3,
            {1: 10.0, 2: 20.0, 3: 30.0},
            2,
            {1: 10.1, 2: 19.6},
        )

        self.assertEqual(result["compared_points"], 2)
        self.assertEqual(result["compared_iteration_end"], 2)

    def test_missing_iteration_in_common_prefix_blocks_comparison(self):
        with self.assertRaisesRegex(COMPARE_LOSS.GateBlocked, "iteration 不完整"):
            COMPARE_LOSS.compare_series(
                2,
                {1: 10.0, 2: 20.0},
                3,
                {1: 10.1, 3: 30.0},
            )

    def test_zero_baseline_blocks_comparison(self):
        with self.assertRaisesRegex(COMPARE_LOSS.GateBlocked, "baseline loss 为 0"):
            COMPARE_LOSS.compare_series(1, {1: 0.0}, 1, {1: 0.1})


class BaselineResolutionTest(unittest.TestCase):
    def test_uses_remote_validation_0001_by_default(self):
        output_root = Path("/tmp/output")

        run_id, run_dir = COMPARE_LOSS.resolve_baseline_run(None, output_root)

        self.assertEqual(run_id, "remote-validation-0001")
        self.assertEqual(run_dir, output_root / "remote-validation-0001")

    def test_resolves_run_name_under_output_root(self):
        output_root = Path("/tmp/output")

        run_id, run_dir = COMPARE_LOSS.resolve_baseline_run(
            "remote-validation-0010", output_root
        )

        self.assertEqual(run_id, "remote-validation-0010")
        self.assertEqual(run_dir, output_root / "remote-validation-0010")

    def test_accepts_explicit_baseline_directory(self):
        baseline_dir = Path("/tmp/custom/remote-validation-0010")

        run_id, run_dir = COMPARE_LOSS.resolve_baseline_run(
            str(baseline_dir), Path("/tmp/output")
        )

        self.assertEqual(run_id, "remote-validation-0010")
        self.assertEqual(run_dir, baseline_dir.resolve())


class ArgumentDefaultsTest(unittest.TestCase):
    def test_uses_one_percent_loss_gate_by_default(self):
        args = COMPARE_LOSS.parse_args(
            ["--candidate-run", "remote-validation-0002"]
        )

        self.assertEqual(args.max_relative_error_percent, 1.0)

    def test_explicit_loss_gate_overrides_default(self):
        args = COMPARE_LOSS.parse_args(
            [
                "--candidate-run",
                "remote-validation-0002",
                "--max-relative-error-percent",
                "0.5",
            ]
        )

        self.assertEqual(args.max_relative_error_percent, 0.5)


if __name__ == "__main__":
    unittest.main()
