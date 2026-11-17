"""Resource ownership and materialized-install smoke tests (no NPU required)."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
SKILLS = ROOT / "skills/inference-model-skills/batch-invariance-validation"


def test_resources_are_owned_by_skills_and_links_resolve():
    assert not list((SKILLS / "scripts").rglob("*.py"))
    assert not list((SKILLS / "references").rglob("*.md"))
    for skill in SKILLS.glob("*/SKILL.md"):
        assert (skill.parent / "references").is_dir()
    for document in SKILLS.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (document.parent / target.split("#", 1)[0]).exists(), (document, target)


@pytest.mark.parametrize("relative", [
    "batch-invariance-environment/scripts/check_batch_invariant_env.py",
    "batch-invariance-operator-analysis/scripts/analyze_op_statistic.py",
    "batch-invariance-functional-test/scripts/analyze_scheduler_log.py",
    "batch-invariance-dataset-acceptance/scripts/validate_math500_acceptance.py",
])
def test_materialized_skill_scripts_resolve_dependencies(tmp_path, relative):
    installed = tmp_path / "plugin" / "skills"
    for skill in SKILLS.glob("*/SKILL.md"):
        shutil.copytree(skill.parent, installed / skill.parent.name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    result = subprocess.run(
        [sys.executable, str(installed / relative), "--help"], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="strict",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stderr
    if "operator-analysis" in relative:
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
             "import analyze_op_statistic as m; assert m.DEFAULT_ALLOWLIST.is_file(); "
             "assert m.soc_version_to_chip_family('Ascend910B') == 'A2'",
             str((installed / relative).parent)], cwd=tmp_path,
            capture_output=True, text=True, encoding="utf-8", errors="strict",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        assert probe.returncode == 0, probe.stderr


def test_phase2_reuses_environment_skill():
    runner = (SKILLS / "batch-invariance-functional-test/scripts/run_phase2_acceptance.sh").read_text(encoding="utf-8")
    assert '${TEST_ROOT}/../batch-invariance-environment' in runner
    assert '${ENVIRONMENT_ROOT}/scripts/check_batch_invariant_env.py' in runner
