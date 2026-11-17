from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("precision_run_alignment", SCRIPTS_DIR / "run_alignment.py")
assert SPEC is not None and SPEC.loader is not None
run_alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_alignment)


class OrchestrationTest(unittest.TestCase):
    def test_layer_counts_are_deduplicated_and_capped(self) -> None:
        self.assertEqual(run_alignment.parse_layer_counts("1,2,4,full,8", 4), [1, 2, 4])

    def test_reference_and_candidate_pythonpaths_are_isolated(self) -> None:
        args = argparse.Namespace(
            transformers_source=Path("/opt/transformers/src"),
            repo_root=Path("/srv/MindSpeed-LLM"),
            candidate_pythonpath=["/opt/candidate-only"],
        )
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/inherited"}, clear=False):
            reference = run_alignment.command_environment(args)
            candidate = run_alignment.command_environment(args, candidate=True)
        self.assertNotIn("/opt/candidate-only", reference["PYTHONPATH"])
        self.assertIn("/opt/candidate-only", candidate["PYTHONPATH"])
        self.assertEqual(candidate["MINDSPEED_LLM_REPO_ROOT"], "/srv/MindSpeed-LLM")

    def test_command_result_contains_log_and_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "step.log"
            result = run_alignment.run_command(
                "probe",
                [sys.executable, "-c", "print('x' * 5000)"],
                log,
                os.environ.copy(),
            )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(result["output_tail"]), 4000)
        self.assertTrue(str(result["log"]).endswith("step.log"))

    def test_early_failure_writes_machine_readable_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
            )
            args = argparse.Namespace(
                model_dir=model_dir,
                repo_root=root / "repo",
                output_dir=root / "output",
                candidate_config=root / "candidate.yaml",
                candidate_world_size=2,
                candidate_master_port=29500,
                candidate_config_override=[],
                checkpoint_config_override=[],
                candidate_pythonpath=[],
                reference_python=sys.executable,
                candidate_python=sys.executable,
                layer_counts="1,2",
                dtype="bf16",
                device="npu:0",
                device_map="none",
                device_map_json=None,
                threshold=0.98,
                accelerator_cleanup_delay=0,
                max_length=4096,
                exact_sequence_length=4096,
                input_format="auto",
                prompt=None,
                model_label="test",
                execution_label="eager",
                modeling_source="modeling.py",
                repo_commit="test-dirty",
                transformers_source=None,
            )
            with mock.patch.object(run_alignment, "parse_args", return_value=args):
                with self.assertRaises(FileNotFoundError):
                    run_alignment.main()

            status = json.loads(
                (args.output_dir / run_alignment.STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "FAILED")
            self.assertEqual(status["current_stage"], "checkpoint_index")
            self.assertEqual(status["error"]["type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
