"""Validate a standalone marketplace source, without access to the root skills."""
import json
import importlib.util
from unittest.mock import patch
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packager = load_module("package_plugin", PLUGIN / "scripts/package_plugin.py")
installer = load_module("install_unit", PLUGIN.parents[1] / "tools/install-unit.py")


class PackagingTests(unittest.TestCase):
    def test_link_and_failed_install_restore_canonical_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "skills/common/common/gitcode-issue-triage"
            shutil.copytree(packager.SOURCE, source)
            plugin = root / "plugins/mindspeed-scan-issue-agent"
            plugin.mkdir(parents=True)
            unit = {"id": plugin.name, "install": {
                "skills_root": "skills/common/common/gitcode-issue-triage"}}
            with patch.object(installer, "ROOT", root), \
                 patch.object(packager, "PLUGIN", plugin), \
                 patch.object(packager, "SOURCE", source):
                self.assertEqual(installer.link_plugin_view(unit), 0)
                self.assertEqual(installer.link_plugin_view(unit), 0)
                view = plugin / "skills/gitcode-issue-triage"
                self.assertFalse(installer._is_real_dir(view))
                with self.assertRaisesRegex(RuntimeError, "install failed"):
                    with packager.materialized_install():
                        self.assertTrue(installer._is_real_dir(view))
                        self.assertEqual((view / "SKILL.md").read_bytes(),
                                         (source / "SKILL.md").read_bytes())
                        raise RuntimeError("install failed")
                self.assertFalse(installer._is_real_dir(view))
                self.assertEqual(view.resolve(), source.resolve())
                self.assertTrue((source / "SKILL.md").is_file())
                # Remove only the link before TemporaryDirectory cleans its tree.
                installer._remove_link(view)

    def test_platform_install_uses_real_files_and_restores_on_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "plugins/mindspeed-scan-issue-agent"
            shutil.copytree(PLUGIN, plugin, ignore=lambda directory, names:
                            {"skills", "__pycache__"} & set(names))
            source = root / "skills/common/common/gitcode-issue-triage"
            shutil.copytree(packager.SOURCE, source)
            unit = {"id": plugin.name, "install": {
                "skills_root": "skills/common/common/gitcode-issue-triage",
                "install_hint": {"codex": "python tools/install-unit.py codex mindspeed-scan-issue-agent --run"}}}
            view = plugin / "skills/gitcode-issue-triage"
            with patch.object(installer, "ROOT", root):
                self.assertEqual(installer.link_plugin_view(unit), 0)
                def install(command, **kwargs):
                    self.assertEqual(command, "codex plugin add mindspeed-scan-issue-agent@MindSpeed-Agent")
                    self.assertTrue(installer._is_real_dir(view))
                    return 7
                # Keep the real link operation outside the subprocess mock.
                with patch.object(installer, "link_plugin_view", return_value=0), \
                     patch.object(installer.shutil, "which", return_value="codex"), \
                     patch.object(installer.subprocess, "call", side_effect=install):
                    self.assertEqual(installer.platform_install(unit, "codex", True), 7)
                self.assertFalse(installer._is_real_dir(view))
                installer._remove_link(view)

    def test_existing_output_and_missing_source_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                packager.build_package(Path(tmp))
            with patch.object(packager, "SOURCE", Path(tmp) / "missing"):
                with self.assertRaises(FileNotFoundError):
                    packager.build_package(Path(tmp) / "package")

    def test_isolated_package_exposes_skill_and_runs_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "installed-plugin"
            packager.build_package(package)
            bundled = package / "skills/gitcode-issue-triage"
            self.assertTrue(installer._is_real_dir(bundled))
            for original in packager.SOURCE.rglob("*"):
                if original.is_file() and "__pycache__" not in original.parts and original.suffix != ".pyc":
                    self.assertEqual(original.read_bytes(),
                                     (bundled / original.relative_to(packager.SOURCE)).read_bytes())
            entry = package / "skills/gitcode-issue-triage/SKILL.md"
            self.assertTrue(entry.is_file())
            text = entry.read_text(encoding="utf-8")
            for reference in ("modes.md", "multi-repository.md", "publishing-safety.md"):
                self.assertIn(f"references/{reference}", text)
                self.assertTrue((entry.parent / "references" / reference).is_file())
            for platform in ("claude", "codex", "dsh"):
                manifest = json.loads((package / f".{platform}-plugin/plugin.json")
                                      .read_text(encoding="utf-8"))
                paths = manifest["skills"]
                if isinstance(paths, str):
                    paths = [paths]
                discovered = set()
                for relative in paths:
                    directory = package / relative
                    self.assertTrue(directory.is_dir(), relative)
                    discovered.update(directory.rglob("SKILL.md"))
                self.assertIn(entry, discovered)
            # Both direct plugin entry and skill-local bridge must work after copying
            # only the marketplace source into an unrelated installation directory.
            for script in (package / "scripts/issue_agent.py",
                           entry.parent / "scripts/issue_agent.py"):
                result = subprocess.run([sys.executable, str(script), "scan", "--help"],
                                        cwd=tmp, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--allow-post", result.stdout)


if __name__ == "__main__":
    unittest.main()
