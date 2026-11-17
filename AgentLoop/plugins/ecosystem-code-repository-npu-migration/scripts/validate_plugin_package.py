#!/usr/bin/env python3
"""Validate the Claude/Codex manifests, marketplaces, skills and bundle lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plugin_bundle import EXPECTED_SKILLS, BundleIntegrityError, load_bundle_context, read_json


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, default=PLUGIN_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.plugin_root.resolve()
    repo = args.repo_root.resolve()
    errors: list[str] = []
    try:
        context = load_bundle_context(root)
    except (BundleIntegrityError, OSError, ValueError, json.JSONDecodeError) as exc:
        context = None
        errors.append(str(exc))
    manifest: dict | None = None
    try:
        manifest = read_json(root / ".codex-plugin" / "plugin.json")
        if manifest.get("name") != "ecosystem-code-repository-npu-migration":
            errors.append("Codex manifest must use name=ecosystem-code-repository-npu-migration")
    except (BundleIntegrityError, OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    claude_manifest: dict | None = None
    try:
        claude_manifest = read_json(root / ".claude-plugin" / "plugin.json")
        claude_market = read_json(repo / ".claude-plugin" / "marketplace.json")
        claude_matches = [
            item
            for item in claude_market.get("plugins", [])
            if isinstance(item, dict) and item.get("name") == "ecosystem-code-repository-npu-migration"
        ]
        if len(claude_matches) != 1:
            errors.append("shared marketplace must contain ecosystem-code-repository-npu-migration exactly once")
        elif claude_manifest.get("version") != claude_matches[0].get("version"):
            errors.append("Claude manifest and marketplace version mismatch")
        if claude_manifest.get("name") != "ecosystem-code-repository-npu-migration":
            errors.append("Claude manifest must use name=ecosystem-code-repository-npu-migration")
        if manifest is not None and claude_manifest.get("version") != manifest.get("version"):
            errors.append("Claude and Codex manifest version mismatch")
        expected_claude_skills = sorted(f"./skills/{name}" for name in EXPECTED_SKILLS)
        if sorted(claude_manifest.get("skills", [])) != expected_claude_skills:
            errors.append("Claude manifest skills must list every bundled skill exactly once")
    except (BundleIntegrityError, OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    skill_dirs = sorted(path.name for path in (root / "skills").iterdir() if path.is_dir())
    if tuple(skill_dirs) != EXPECTED_SKILLS:
        errors.append(f"expected exactly {len(EXPECTED_SKILLS)} bundled skills: {skill_dirs}")
    for name in skill_dirs:
        skill_file = root / "skills" / name / "SKILL.md"
        if not skill_file.is_file() or f"name: {name}" not in skill_file.read_text(encoding="utf-8").split("---", 2)[1]:
            errors.append(f"skill frontmatter name mismatch: {name}")

    forbidden_paths = [root / "runtime", root / "skills" / "custom-workflow"]
    for path in forbidden_paths:
        if path.exists():
            errors.append(f"plugin contains forbidden path: {path.relative_to(root)}")
    js_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.js"))
    if js_files:
        errors.append(f"plugin contains JavaScript files: {js_files}")

    payload = {
        "status": "PASS" if not errors else "BLOCKED",
        "reason_code": "PLUGIN_PACKAGE_VALID" if not errors else "PLUGIN_PACKAGE_INVALID",
        "plugin_version": context.version if context else "",
        "bundle_digest": context.bundle_digest if context else "",
        "skill_count": len(skill_dirs),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
