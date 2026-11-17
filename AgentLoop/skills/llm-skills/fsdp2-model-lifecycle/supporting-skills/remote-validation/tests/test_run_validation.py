from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_validation.py"
SPEC = importlib.util.spec_from_file_location("remote_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
remote_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote_validation)


def args_for(config_path: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "config": str(config_path),
        "mode": "plan",
        "local_repo": None,
        "task": None,
        "host": None,
        "user": None,
        "port": None,
        "remote_repo": None,
        "script": None,
        "conda_env": None,
        "cann_env_scripts": None,
        "output_dir": None,
        "run_id": None,
        "timeout_seconds": None,
        "skip_sync": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RemoteValidationTest(unittest.TestCase):
    def make_config(self, root: Path, *, task_type: str = "precision-alignment") -> tuple[Path, dict]:
        repo = root / "repo"
        launcher = repo / "examples" / "launch.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        runtime = root / "runtime"
        runtime.mkdir()
        (runtime / "run_alignment.py").write_text("print('ok')\n", encoding="utf-8")
        config = {
            "schema_version": 1,
            "connection": {"host": "npu.example", "user": "tester", "port": 22},
            "repository": {
                "local_path": str(repo),
                "remote_path": "/srv/validation/MindSpeed-LLM",
            },
            "execution": {
                "task_type": task_type,
                "script": "examples/launch.sh",
                "script_args": ["--smoke"],
                "environment": {
                    "OUTPUT_DIR": "{remote_repo}/.validation-artifacts/{run_id}",
                    "PRECISION_SKILL_DIR": "{remote_repo}/.validation-runtime/precision-alignment",
                },
                "required_remote_paths": [
                    {
                        "path": "{remote_repo}/.validation-runtime/precision-alignment/run_alignment.py",
                        "kind": "file",
                    }
                ],
                "conda_env": "candidate",
                "cann_env_scripts": ["/opt/cann/set_env.sh"],
                "timeout_seconds": 60,
            },
            "sync": {
                "delete": True,
                "runtime_bundles": [
                    {
                        "local_path": str(runtime),
                        "remote_subdir": ".validation-runtime/precision-alignment",
                    }
                ],
            },
            "artifacts": {
                "remote_path": "{remote_repo}/.validation-artifacts/{run_id}",
                "include": ["*.json", "*.log"],
                "exclude": ["*.pt"],
                "pull_on_failure": True,
            },
            "output": {"local_path": str(root / "output")},
        }
        config_path = root / "remote.local.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, config

    def test_precision_config_builds_single_remote_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _ = self.make_config(Path(temp_dir))
            config = remote_validation.resolve_config(args_for(config_path))

            plan = remote_validation.build_plan(config, args_for(config_path), "remote-validation-0007")
            execution = " ".join(plan["commands"]["execution"])
            preflight = " ".join(plan["commands"]["preflight"])

            self.assertEqual(config["execution"]["task_type"], "precision-alignment")
            self.assertEqual(len(plan["commands"]["runtime_syncs"]), 1)
            self.assertNotIn("training", plan["commands"])
            self.assertIn("bash -e -o pipefail examples/launch.sh --smoke", execution)
            self.assertIn("remote-validation-0007", execution)
            self.assertIn("run_alignment.py", preflight)
            self.assertIn("REPO_REVISION", execution)

    def test_repository_sync_protects_runtime_artifacts_and_git(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _ = self.make_config(Path(temp_dir))
            config = remote_validation.resolve_config(args_for(config_path))

            _, command = remote_validation.build_sync(config)
            rendered = " ".join(command)

            self.assertIn("--delete", command)
            self.assertIn(".git", command)
            self.assertIn(".validation-runtime/", command)
            self.assertIn(".validation-artifacts/", command)
            self.assertIn("/srv/validation/MindSpeed-LLM/", rendered)

    def test_runtime_bundle_can_sync_only_declared_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, raw = self.make_config(root)
            raw["sync"]["runtime_bundles"][0]["include"] = [
                "alignment_utils.py",
                "run_alignment.py",
            ]
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = remote_validation.resolve_config(args_for(config_path))

            command = remote_validation.build_runtime_syncs(config)[0]["sync"]

            self.assertIn("alignment_utils.py", command)
            self.assertIn("run_alignment.py", command)
            self.assertIn("--delete-excluded", command)
            self.assertEqual(command[-3:-2], ["*"])

    def test_artifact_excludes_win_over_includes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _ = self.make_config(root)
            config = remote_validation.resolve_config(args_for(config_path))

            command = remote_validation.build_artifact_sync(
                config, "remote-validation-0003", root / "pulled"
            )
            assert command is not None

            self.assertLess(command.index("*.pt"), command.index("*.json"))
            self.assertIn(
                "tester@npu.example:/srv/validation/MindSpeed-LLM/.validation-artifacts/remote-validation-0003/",
                command,
            )

    def test_rejects_reserved_environment_and_overlapping_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, raw = self.make_config(root)
            raw["execution"]["environment"]["REPO_REVISION"] = "fake"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(remote_validation.ConfigError, "保留变量"):
                remote_validation.resolve_config(args_for(config_path))

            raw["execution"]["environment"].pop("REPO_REVISION")
            raw["sync"]["runtime_bundles"].append(
                {
                    "local_path": raw["sync"]["runtime_bundles"][0]["local_path"],
                    "remote_subdir": ".validation-runtime/precision-alignment/internal",
                }
            )
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(remote_validation.ConfigError, "重复或嵌套"):
                remote_validation.resolve_config(args_for(config_path))

    def test_training_report_keeps_physical_training_log_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = remote_validation.initial_report(
                "run", "remote-validation-0001", Path(temp_dir), "training"
            )
            self.assertTrue(report["artifacts"]["execution_log"].endswith("training.log"))
            self.assertEqual(
                report["artifacts"]["training_log"], report["artifacts"]["execution_log"]
            )

    def test_legacy_training_config_defaults_and_command_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, raw = self.make_config(root)
            raw["execution"].pop("task_type")
            raw["execution"].pop("environment")
            raw["execution"].pop("required_remote_paths")
            raw["sync"].pop("runtime_bundles")
            raw.pop("artifacts")
            config_path.write_text(json.dumps(raw), encoding="utf-8")

            config = remote_validation.resolve_config(args_for(config_path))
            plan = remote_validation.build_plan(config, args_for(config_path))

            self.assertEqual(config["execution"]["task_type"], "training")
            self.assertEqual(plan["commands"]["training"], plan["commands"]["execution"])
            self.assertEqual(plan["commands"]["runtime_syncs"], [])

    def test_run_syncs_bundles_then_executes_launcher_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _ = self.make_config(root)
            config = remote_validation.resolve_config(args_for(config_path))
            cli_args = args_for(config_path, mode="run")

            def success(command: list[str], timeout: int) -> dict:
                stdout = ""
                if "npu-smi info" in " ".join(command):
                    stdout = (
                        "| NPU Chip | Process id | Process name |\n"
                        "| No running processes found in NPU 0 |\n"
                    )
                return {
                    "command": command,
                    "exit_code": 0,
                    "stdout": stdout,
                    "stderr": "",
                    "elapsed_seconds": 0.0,
                    "timed_out": False,
                }

            execution_result = {
                "command": ["ssh", "launcher"],
                "exit_code": 0,
                "elapsed_seconds": 0.0,
                "timed_out": False,
                "log_path": str(root / "output" / "execution.log"),
                "output_tail": "",
            }
            with (
                mock.patch.object(remote_validation, "parse_args", return_value=cli_args),
                mock.patch.object(remote_validation, "resolve_config", return_value=config),
                mock.patch.object(remote_validation.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(remote_validation, "command_result", side_effect=success) as calls,
                mock.patch.object(
                    remote_validation, "stream_training", return_value=execution_result
                ) as execution,
                mock.patch("builtins.print"),
            ):
                exit_code = remote_validation.main([])

            self.assertEqual(exit_code, 0)
            execution.assert_called_once()
            issued = [" ".join(call.args[0]) for call in calls.call_args_list]
            main_sync = next(index for index, item in enumerate(issued) if item.startswith("rsync"))
            runtime_sync = next(
                index
                for index, item in enumerate(issued)
                if item.startswith("rsync") and ".validation-runtime/precision-alignment" in item
            )
            preflight = next(index for index, item in enumerate(issued) if "task_type=" in item)
            self.assertLess(main_sync, runtime_sync)
            self.assertLess(runtime_sync, preflight)


if __name__ == "__main__":
    unittest.main()


class SilenceWatchdogTest(unittest.TestCase):
    """stream_training 的静默看门狗：无输出超时判定挂死（exit 125），正常输出不受影响。"""

    def test_silent_process_is_killed_with_125(self) -> None:
        import time as _time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "out.log"
            started = _time.monotonic()
            result = remote_validation.stream_training(
                ["bash", "-c", "sleep 60"], log, timeout=3600, silence_timeout=2
            )
            self.assertEqual(result["exit_code"], 125)
            self.assertTrue(result["stalled"])
            self.assertLess(_time.monotonic() - started, 30)

    def test_chatty_process_runs_to_completion(self) -> None:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "out.log"
            result = remote_validation.stream_training(
                ["bash", "-c", "for i in 1 2 3; do echo line$i; sleep 1; done"],
                log, timeout=3600, silence_timeout=5,
            )
            self.assertEqual(result["exit_code"], 0)
            self.assertFalse(result["stalled"])
            self.assertIn("line3", result["output_tail"])


class ArtifactDirCheckTest(unittest.TestCase):
    """precision-alignment 启动前的远端产物目录冲突快速失败检查。"""

    def _config(self, tmp: str, task_type: str = "precision-alignment") -> dict:
        script = Path(tmp) / "examples" / "x.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\n")
        config_path = Path(tmp) / "cfg.json"
        config_path.write_text(json.dumps({
            "connection": {"host": "h", "user": "u"},
            "repository": {"local_path": tmp, "remote_path": "/remote/repo"},
            "execution": {"task_type": task_type, "script": "examples/x.sh",
                          "conda_env": "e", "cann_env_scripts": ["/opt/cann/set_env.sh"]},
            "artifacts": {"remote_path": "{remote_repo}/.validation-artifacts/{run_id}"},
            "output": {"local_path": tmp},
        }))
        return remote_validation.resolve_config(
            args_for(config_path)
        )

    def test_builds_check_for_precision_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp)
            cmd = remote_validation.build_artifact_dir_check(config, "remote-validation-0007")
            self.assertIsNotNone(cmd)
            self.assertIn("remote-validation-0007", cmd[-1])

    def test_skipped_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, task_type="training")
            self.assertIsNone(remote_validation.build_artifact_dir_check(config, "remote-validation-0001"))

    def test_skipped_without_remote_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp)
            config["artifacts"] = {}
            self.assertIsNone(remote_validation.build_artifact_dir_check(config, "remote-validation-0001"))
