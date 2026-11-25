"""CPU fake CLI only; not actual CANNBot or NPU acceptance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatch_operator_tasks import dispatch


class DispatchTests(unittest.TestCase):
    def test_two_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            model = run / "model"
            model.mkdir()
            commands = {}
            for i, plugin in enumerate(("triton-op-generator", "tilelang2ascendc-ops-generator")):
                task = run / "requests" / f"task_{i:03d}"
                (run / "state.json").write_text(json.dumps({"status":"READY", "selected_operator":{"kind":"generation","task":task.name}}))
                task.mkdir(parents=True)
                (task / "request.json").write_text(json.dumps({
                    "route": {"plugin": plugin, "action": "optimize_existing"},
                    "output_dir": str(task), "model_root": str(model)}))
                (task / "prompt.txt").write_text("reviewed")
                commands[plugin] = [sys.executable, "-c",
                    "import os,sys; assert os.environ['AGENTLOOP_CANNBOT_PLUGIN']==sys.argv[1]; print(sys.argv[2])",
                    "{plugin}", "{prompt}"]
                result, attempt = dispatch(run, task.name, {"commands": commands}, "requests")
                self.assertEqual(result["status"], "CLI_COMPLETED_UNVERIFIED")
                self.assertEqual(result["acceptance"], "NOT_VERIFIED")
                self.assertIn("reviewed", (attempt / "command.log").read_text())

    def test_no_unknown_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dispatch(tmp, "../escape", {}, "requests")
