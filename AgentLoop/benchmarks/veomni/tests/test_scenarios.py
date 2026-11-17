from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_scenarios.py"
SPEC = importlib.util.spec_from_file_location("veomni_scenarios", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@unittest.skipIf(runner.yaml is None, "PyYAML 未安装")
class ScenarioValidationTests(unittest.TestCase):
    def test_all_declared_scenarios_are_valid(self) -> None:
        paths = sorted(runner.SCENARIO_DIR.glob("*.yaml"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertEqual(runner.validate_scenario(path), [], path.name)

    def test_omni_preview_only_requires_declared_stages(self) -> None:
        payload = runner.load_yaml(runner.SCENARIO_DIR / "omni-qwen3-omni.yaml")
        stages, errors = runner.stage_names(payload["stages"])
        self.assertEqual(errors, [])
        self.assertEqual(stages, [
            "environment", "model-analysis", "adaptation", "functional-validation", "precision-alignment"
        ])


if __name__ == "__main__":
    unittest.main()
