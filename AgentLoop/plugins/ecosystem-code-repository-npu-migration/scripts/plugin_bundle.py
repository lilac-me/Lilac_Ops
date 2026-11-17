#!/usr/bin/env python3
"""ecosystem-code-repository-npu-migration bundle identity and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BUNDLE_LOCK_SCHEMA_VERSION = "ecosystem-code-repository-npu-migration-bundle-lock-v3"
PLUGIN_NAME = "ecosystem-code-repository-npu-migration"
EXPECTED_SKILLS = (
    "docs-support",
    "env-setup",
    "feature-model",
    "goal",
    "local-assets",
    "repair",
    "report-pr",
    "vllm-ascend-env",
)
# Canonical skill sources live at <repo>/skills/<domain-skills>/<function>/<skill>/.
# The plugin keeps a flat skills/<skill> view through relative symlinks so the
# Codex manifest, EXPECTED_SKILLS and the bundle lock keep their shape.  This
# mapping is the single Python-side source of truth; install.sh, install.bat
# and plugin README mirror it for shell-side link creation.
PLUGIN_DOMAIN = "ecosystem-code-repository-npu-migration"
DOMAIN_SKILLS_DIRNAME = "ecosystem-code-repository-npu-migration"
SKILL_FUNCTION = {
    "docs-support": "common",
    "env-setup": "deployment",
    "feature-model": "model-analysis",
    "goal": "model-adaptation",
    "local-assets": "common",
    "repair": "model-adaptation",
    "report-pr": "common",
    "vllm-ascend-env": "deployment",
}
EXCLUDED_PARTS = {".git", ".npu-agent", ".pytest_cache", ".ruff_cache", "__pycache__"}
EXCLUDED_NAMES = {"bundle.lock.json"}


class BundleIntegrityError(RuntimeError):
    """Raised when a plugin checkout does not match its Codex bundle lock."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"expected JSON object: {path}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def find_plugin_root(start: str | Path) -> Path:
    candidate = Path(start).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for root in (candidate, *candidate.parents):
        if (root / ".codex-plugin" / "plugin.json").is_file():
            return root
    raise BundleIntegrityError(f"cannot find {PLUGIN_NAME} Codex plugin root from {candidate}")


def canonical_skill_dir(repo_root: Path, name: str) -> Path:
    if name not in SKILL_FUNCTION:
        raise BundleIntegrityError(f"unknown ecosystem-code-repository-npu-migration skill: {name}")
    return repo_root / "skills" / DOMAIN_SKILLS_DIRNAME / SKILL_FUNCTION[name] / name


def ensure_skill_links(plugin_root: Path) -> list[Path]:
    """Idempotently create the plugin's flat skills/<name> symlink view.

    Correct links are kept, wrong-target links are rebuilt, and existing real
    directories are left untouched (Windows junctions from install.bat,
    staging copies made around ``codex plugin add``, and test fixtures copied
    with ``shutil.copytree`` all land in that category).  The canonical source
    is only required when a link actually has to be created.
    """
    repo_root = plugin_root.parents[1]
    skills_dir = plugin_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name in EXPECTED_SKILLS:
        link = skills_dir / name
        canonical = canonical_skill_dir(repo_root, name)
        if link.is_symlink():
            if link.resolve() == canonical.resolve():
                continue
            link.unlink()
        elif link.exists():
            continue
        if not canonical.is_dir():
            raise BundleIntegrityError(
                f"canonical skill source missing: {canonical}; run install.sh / install.bat"
            )
        try:
            link.symlink_to(os.path.relpath(canonical, skills_dir))
        except OSError as exc:
            raise BundleIntegrityError(
                f"cannot create skill link {link} -> {canonical}: {exc}; run install.bat"
            ) from exc
        created.append(link)
    return created


def _is_excluded(path: Path, relative: Path) -> bool:
    return (
        not path.is_file()
        or path.name in EXCLUDED_NAMES
        or path.suffix == ".pyc"
        or any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def iter_bundle_files(plugin_root: Path) -> Iterable[Path]:
    # Path.rglob does not descend into symlinked directories, so the skills/
    # entries are traversed through their canonical targets separately and
    # re-exposed under the plugin-relative skills/<name>/ view paths.
    for path in sorted(plugin_root.rglob("*")):
        relative = path.relative_to(plugin_root)
        if relative.parts and relative.parts[0] == "skills":
            continue
        if _is_excluded(path, relative):
            continue
        yield path
    skills_dir = plugin_root / "skills"
    if not skills_dir.is_dir():
        return
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        canonical = entry.resolve()
        for path in sorted(canonical.rglob("*")):
            relative_to_skill = path.relative_to(canonical)
            if _is_excluded(path, relative_to_skill):
                continue
            yield entry / relative_to_skill


def file_digests(plugin_root: Path) -> dict[str, str]:
    return {
        path.relative_to(plugin_root).as_posix(): sha256_bytes(path.read_bytes())
        for path in iter_bundle_files(plugin_root)
    }


def build_bundle_payload(
    plugin_root: str | Path,
    *,
    source_commit: str = "",
    source_dirty: bool = False,
) -> dict[str, Any]:
    root = find_plugin_root(plugin_root)
    ensure_skill_links(root)
    manifest = read_json(root / ".codex-plugin" / "plugin.json")
    skills = sorted(path.name for path in (root / "skills").iterdir() if path.is_dir())
    files = file_digests(root)
    semantic = {
        "schema_version": BUNDLE_LOCK_SCHEMA_VERSION,
        "plugin": {"name": manifest.get("name"), "version": manifest.get("version")},
        "skills": skills,
        "files": files,
    }
    return {
        **semantic,
        "bundle_digest": sha256_json(semantic),
        "source": {"commit": source_commit, "dirty": bool(source_dirty)},
    }


@dataclass(frozen=True)
class BundleContext:
    plugin_root: Path
    version: str
    bundle_digest: str

    def skill(self, name: str) -> Path:
        if name not in EXPECTED_SKILLS:
            raise BundleIntegrityError(f"unknown ecosystem-code-repository-npu-migration skill: {name}")
        path = (self.plugin_root / "skills" / name).resolve()
        if not (path / "SKILL.md").is_file():
            raise BundleIntegrityError(f"missing skill: {name}")
        return path


def load_bundle_context(start: str | Path, *, verify_files: bool = True) -> BundleContext:
    root = find_plugin_root(start)
    ensure_skill_links(root)
    manifest = read_json(root / ".codex-plugin" / "plugin.json")
    lock = read_json(root / "bundle.lock.json")
    errors: list[str] = []
    if manifest.get("name") != PLUGIN_NAME:
        errors.append("Codex manifest name mismatch")
    if lock.get("schema_version") != BUNDLE_LOCK_SCHEMA_VERSION:
        errors.append("unsupported or missing bundle lock schema")
    identity = {"name": PLUGIN_NAME, "version": manifest.get("version")}
    if lock.get("plugin") != identity:
        errors.append("bundle lock plugin identity mismatch")
    actual_skills = tuple(sorted(path.name for path in (root / "skills").iterdir() if path.is_dir()))
    if actual_skills != EXPECTED_SKILLS or tuple(lock.get("skills") or ()) != EXPECTED_SKILLS:
        errors.append(f"expected exactly {len(EXPECTED_SKILLS)} Codex skills")
    if verify_files:
        current = build_bundle_payload(
            root,
            source_commit=str((lock.get("source") or {}).get("commit") or ""),
            source_dirty=bool((lock.get("source") or {}).get("dirty")),
        )
        for key in ("files", "bundle_digest"):
            if current.get(key) != lock.get(key):
                errors.append(f"bundle {key} mismatch")
    if errors:
        raise BundleIntegrityError("; ".join(errors))
    return BundleContext(root, str(manifest.get("version") or ""), str(lock.get("bundle_digest") or ""))
