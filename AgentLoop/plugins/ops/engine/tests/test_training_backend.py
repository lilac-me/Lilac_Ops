"""CPU-only tests: fake profiler and fake training, not NPU evidence."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_backend.base import normalize, phase_env
from training_backend import TrainingHook


class FakeProfiler:
    def __init__(self, directory, skip, warmup, active):
        self.args = (skip, warmup, active)
        self.directory = directory
        self.steps = 0
    def __enter__(self):
        return self
    def step(self):
        self.steps += 1
    def __exit__(self, *exc):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "kernel_details.csv").write_text("fake")
        return False


class BackendTests(unittest.TestCase):
    def config(self, backend="fsdp"):
        return {"training": {"backend": backend, "model_root": "/tmp/model",
                             "launch_command": ["bash", "train.sh"]}}

    def test_defaults_and_megatron_native(self):
        cfg = normalize(self.config())
        self.assertEqual(cfg["profile_subdir"], "rank_0")
        self.assertEqual(cfg["profile_command"], ["bash", "train.sh"])
        self.assertEqual(cfg["benchmark_command"], cfg["profile_command"])
        m = normalize(self.config("megatron"))
        self.assertIn("training_benchmark.py", m["benchmark_command"][1])
        self.assertEqual(phase_env(cfg, "tune"), {})
        self.assertEqual(normalize({"profile_command": ["x"]}), {"profile_command": ["x"]})

    def test_bad_window_and_rank(self):
        for p in ({"active": 11}, {"ranks": [0, 1]}, {"repeat": 2}):
            cfg = self.config()
            cfg["profiling"] = p
            with self.assertRaises(ValueError):
                normalize(cfg)

    def env(self, root, phase):
        return {**phase_env(normalize(self.config()), phase),
                "AGENTLOOP_PHASE": phase, "AGENTLOOP_RUN_ID": "r",
                "AGENTLOOP_WORKLOAD_ID": "w",
                "AGENTLOOP_PROFILE_ENABLED": str(int(phase == "profile")),
                "AGENTLOOP_PROFILE_DIR": str(root / "profile"),
                "AGENTLOOP_METRICS_PATH": str(root / "metrics.json")}

    def test_profile_once_selected_rank_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            e = self.env(root, "profile")
            created = []
            def factory(*args):
                obj = FakeProfiler(*args)
                created.append(obj)
                return obj
            for rank in (1, 0):
                with TrainingHook(rank=rank, env=e, profiler_factory=factory) as hook:
                    self.assertEqual([hook.step() for _ in range(20)], [False]*19+[True])
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].args, (10, 1, 3))
            self.assertEqual(created[0].steps, 20)
            done = json.loads((root / "profile/rank_0/agentloop_profile_done.json").read_text())
            self.assertEqual(done["steps"], 20)
            self.assertFalse((root / "metrics.json").exists())

    def test_benchmark_window_no_profiler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for phase in ("baseline", "regression"):
                e = self.env(root, phase)
                with TrainingHook(rank=0, env=e) as hook:
                    for i in range(20):
                        self.assertEqual(hook.needs_metrics, i >= 10)
                        stop = hook.step(loss=1.0, step_time_ms=100.0, tokens=4096)
                    self.assertTrue(stop)
                data = json.loads((root / "metrics.json").read_text())
                self.assertEqual([s["step"] for s in data["samples"]], list(range(11, 21)))
                self.assertFalse(data["profiling_enabled"])

    def test_early_exit_fails_and_unmanaged_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                with TrainingHook(rank=0, env=self.env(Path(temp), "baseline")) as hook:
                    hook.step()
        with TrainingHook(env={}) as hook:
            self.assertFalse(hook.step())
            self.assertFalse(hook.needs_metrics)


if __name__ == "__main__":
    unittest.main()
