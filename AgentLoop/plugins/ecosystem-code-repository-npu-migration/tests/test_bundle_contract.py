"""Bundle integrity contracts for the dual-platform ecosystem-code-repository-npu-migration plugin."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(root: Path, expect: int) -> dict:
    completed = subprocess.run([sys.executable, str(root / "scripts" / "validate_bundle.py"), "--plugin-root", str(root), "--json"], text=True, capture_output=True, check=False)
    assert completed.returncode == expect, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def test_bundle_accepts_current_checkout() -> None:
    assert run(ROOT, 0)["reason_code"] == "BUNDLE_INTEGRITY_OK"


def test_bundle_rejects_tamper(tmp_path: Path) -> None:
    copied = tmp_path / "plugin"
    shutil.copytree(ROOT, copied)
    helper = copied / "scripts" / "plugin_bundle.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    assert run(copied, 2)["reason_code"] == "BUNDLE_INTEGRITY_MISMATCH"


def test_bundle_rejects_manifest_version_mismatch(tmp_path: Path) -> None:
    copied = tmp_path / "plugin"
    shutil.copytree(ROOT, copied)
    manifest = copied / ".codex-plugin" / "plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "9.9.9"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert run(copied, 2)["reason_code"] == "BUNDLE_INTEGRITY_MISMATCH"


def test_plugin_ships_dual_platform_manifests() -> None:
    assert not (ROOT / "runtime").exists()
    assert not list(ROOT.rglob("*.js"))
    codex_manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    claude_manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert codex_manifest["version"] == "0.16.0"
    assert claude_manifest["version"] == codex_manifest["version"]
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import plugin_bundle

    assert sorted(claude_manifest["skills"]) == sorted(
        f"./skills/{name}" for name in plugin_bundle.EXPECTED_SKILLS
    )
    repo_root = ROOT.parents[1]
    market = json.loads((repo_root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    matches = [item for item in market["plugins"] if item.get("name") == "ecosystem-code-repository-npu-migration"]
    assert len(matches) == 1
    assert matches[0]["version"] == codex_manifest["version"]
    lock = json.loads((ROOT / "bundle.lock.json").read_text(encoding="utf-8"))
    assert lock["schema_version"] == "ecosystem-code-repository-npu-migration-bundle-lock-v3"
    assert len(lock["skills"]) == 8


def test_skills_view_links_to_canonical_domain_dirs() -> None:
    import sys

    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import plugin_bundle

    repo_root = ROOT.parents[1]
    for name in plugin_bundle.EXPECTED_SKILLS:
        assert (ROOT / "skills" / name / "SKILL.md").is_file(), name
        canonical = (
            repo_root
            / "skills"
            / plugin_bundle.DOMAIN_SKILLS_DIRNAME
            / plugin_bundle.SKILL_FUNCTION[name]
            / name
        )
        assert canonical.is_dir(), canonical
        assert (ROOT / "skills" / name).resolve() == canonical.resolve(), name
