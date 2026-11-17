#!/usr/bin/env python3
"""Validate the mechanical information architecture of bundled skills.

This is intentionally a small structural lint, not an NLP quality score.  It
checks routing metadata, progressive-disclosure pointers, and the few sibling
boundaries that must remain unambiguous.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plugin_bundle import (  # noqa: E402
    DOMAIN_SKILLS_DIRNAME,
    PLUGIN_DOMAIN,
    SKILL_FUNCTION,
    ensure_skill_links,
)


SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
RICH_REQUIRED_FIELDS = ("version", "domain", "function", "tags", "author")
DESCRIPTION_PREFIX = "This skill should be used when"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---", 4)
    if marker < 0:
        return {}, text
    values: dict[str, Any] = {}
    current_key: str | None = None
    for line in text[4:marker].splitlines():
        item = re.match(r"^\s+-\s+(.*)$", line)
        if item and current_key is not None:
            entry = item.group(1).strip()
            if len(entry) >= 2 and entry[0] == entry[-1] and entry[0] in {'"', "'"}:
                entry = entry[1:-1]
            existing = values.get(current_key)
            if isinstance(existing, list):
                existing.append(entry)
            else:
                values[current_key] = [entry]
            continue
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not match:
            current_key = None
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
        # List items may only follow a bare `key:` line.
        current_key = key if not value else None
    return values, text[marker + 4 :]


def _clean_resource(value: str) -> str:
    value = value.strip().strip("`<>")
    value = value.split("#", 1)[0]
    value = value.split("?", 1)[0]
    value = value.rstrip(".,;:)")
    if value.startswith("<plugin root>/"):
        return ""
    return value


def referenced_paths(body: str) -> set[str]:
    values: set[str] = set()
    # Inline code and Markdown link destinations cover the intended resource
    # declarations while avoiding ordinary prose such as "references are...".
    candidates = re.findall(r"`([^`]+)`", body)
    candidates.extend(re.findall(r"\[[^\]]+\]\(([^)]+)\)", body))
    for candidate in candidates:
        plugin_skill = re.search(
            r"<plugin-root>/((?:skills/[A-Za-z0-9_.-]+/)?(?:references|scripts|templates|schemas)/[A-Za-z0-9_.*/?=-]+)",
            candidate,
        )
        if plugin_skill:
            values.add("@plugin/" + _clean_resource(plugin_skill.group(1)))
            continue
        for match in re.finditer(
            r"(?:(?:\.\./)+)?(?:references|scripts|templates|schemas)/"
            r"[A-Za-z0-9_.*/?=-]+",
            candidate,
        ):
            value = _clean_resource(match.group(0))
            if "<plugin root>" in candidate and value.startswith("scripts/"):
                value = "@plugin/" + value
            if value:
                values.add(value)
    return values


def _containment_anchors(plugin_root: Path) -> list[Path]:
    """Resolve-relative containment anchors for the symlinked skill layout.

    Skill-internal paths resolve to their canonical targets under the repo
    skills/ domain directory, while @plugin/ paths stay under the plugin root;
    both trees are legitimate resource locations.
    """
    anchors = [plugin_root.resolve()]
    canonical_root = plugin_root.parents[1] / "skills" / DOMAIN_SKILLS_DIRNAME
    if canonical_root.is_dir():
        anchors.append(canonical_root.resolve())
    return anchors


def resource_exists(skill_dir: Path, value: str) -> bool:
    plugin_root = skill_dir.parents[1]
    if value.startswith("@plugin/"):
        path = (plugin_root / value[len("@plugin/") :]).resolve()
    else:
        path = (skill_dir / value).resolve()
    anchors = _containment_anchors(plugin_root)
    if not any(_is_relative_to(path, anchor) for anchor in anchors):
        return False
    if any(char in value for char in "*?["):
        return any(path.parent.glob(path.name)) if path.parent.exists() else False
    return path.exists()


def _is_relative_to(path: Path, anchor: Path) -> bool:
    try:
        path.relative_to(anchor)
    except ValueError:
        return False
    return True


def markdown_resource_paths(body: str) -> set[str]:
    values: set[str] = set()
    for candidate in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        # _clean_resource strips "<>" which would mangle the <plugin-root>
        # anchor, so normalize it before the generic cleanup.
        if candidate.strip().startswith("<plugin-root>/"):
            anchored = _clean_resource(candidate.strip()[len("<plugin-root>/") :])
            if anchored:
                values.add("@plugin/" + anchored)
            continue
        value = _clean_resource(candidate)
        if not value or value.startswith(("http://", "https://", "#")):
            continue
        values.add(value)
    return values


def referenced_file_exists(plugin_root: Path, source: Path, value: str) -> bool:
    if value.startswith("@plugin/"):
        path = plugin_root / value[len("@plugin/") :]
    else:
        path = source.parent / value
    path = path.resolve()
    if not any(_is_relative_to(path, anchor) for anchor in _containment_anchors(plugin_root)):
        return False
    if any(char in value for char in "*?["):
        return path.parent.exists() and any(path.parent.glob(path.name))
    return path.exists()


def is_compatibility_alias(frontmatter: dict[str, str], body: str) -> bool:
    """Recognize an alias from its own metadata, without naming a product role."""
    description = frontmatter.get("description", "")
    return bool(
        re.search(r"\balias\b", description, re.I)
        or re.search(r"^#{1,3}\s+compatibility\s+alias\s*$", body, re.I | re.M)
    )


def validate_skill(skill_dir: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    name = skill_dir.name
    skill_path = skill_dir / "SKILL.md"
    errors: list[str] = []
    warnings: list[str] = []
    if not skill_path.is_file():
        return [f"{name}: missing SKILL.md"], warnings, {}
    text = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    declared_name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")
    if declared_name != name:
        errors.append(f"{name}: frontmatter name must be {name!r}")
    if not description:
        errors.append(f"{name}: frontmatter description is required")
    if len(description) > 600:
        warnings.append(f"{name}: description is {len(description)} characters")
    if len(text.splitlines()) > 400:
        warnings.append(f"{name}: SKILL.md has {len(text.splitlines())} lines")

    missing = [field for field in RICH_REQUIRED_FIELDS if not frontmatter.get(field)]
    if missing:
        errors.append(f"{name}: frontmatter missing required fields: {', '.join(missing)}")
    domain = frontmatter.get("domain")
    if domain and domain != PLUGIN_DOMAIN:
        errors.append(f"{name}: frontmatter domain must be {PLUGIN_DOMAIN!r}")
    function = frontmatter.get("function")
    expected_function = SKILL_FUNCTION.get(name)
    if function and expected_function and function != expected_function:
        errors.append(f"{name}: frontmatter function must be {expected_function!r}")
    version = frontmatter.get("version")
    if version and not SEMVER.fullmatch(str(version)):
        errors.append(f"{name}: frontmatter version must be semver")
    tags = frontmatter.get("tags")
    if tags and not isinstance(tags, list):
        errors.append(f"{name}: frontmatter tags must be a YAML list")
    elif isinstance(tags, list):
        entries = [tag for tag in tags if isinstance(tag, str) and tag.strip()]
        if not 1 <= len(entries) <= 5 or len(entries) != len(tags):
            errors.append(f"{name}: frontmatter tags must contain 1-5 non-empty entries")

    lower = body.lower()
    alias = is_compatibility_alias(frontmatter, body)
    if not alias and not description.startswith(DESCRIPTION_PREFIX):
        errors.append(f"{name}: description must start with {DESCRIPTION_PREFIX!r}")
    if not alias:
        required = {
            "role": (r"^#{1,3}\s+(?:role|purpose|overview|responsibilities?|角色|目的|概述|职责)\s*#*\s*$", "role"),
            "routing": (r"^#{1,3}\s+(?:use when|when to use|usage|routing|使用场景|路由)\s*#*\s*$", "routing"),
            "boundary": (r"^#{1,3}\s+(?:do not use when|boundaries?|边界)\s*#*\s*$", "boundary"),
            "completion": (r"^#{1,3}\s+(?:completion|output contract|pass conditions?|完成条件)\s*#*\s*$", "completion"),
            "resources": (r"^#{1,3}\s+(?:resources?|资源)\s*#*\s*$", "resources"),
        }
        for label, (pattern, display) in required.items():
            if not re.search(pattern, lower, flags=re.MULTILINE):
                errors.append(f"{name}: missing {display} semantics")
    elif not re.search(
        r"^#{1,3}\s+(?:completion|output contract|pass conditions?|完成条件)\s*#*\s*$",
        lower,
        flags=re.MULTILINE,
    ):
        errors.append(f"{name}: compatibility aliases must define completion semantics")
    if alias and not re.search(r"\balias\b", description, re.I):
        errors.append(f"{name}: compatibility alias description must identify the alias")
    for value in sorted(referenced_paths(body)):
        if not resource_exists(skill_dir, value):
            errors.append(f"{name}: referenced resource does not exist: {value}")
    plugin_root = skill_dir.parents[1]
    for source in sorted(skill_dir.rglob("*.md")):
        if source == skill_path:
            continue
        for value in sorted(markdown_resource_paths(source.read_text(encoding="utf-8"))):
            if not referenced_file_exists(plugin_root, source, value):
                relative_source = source.relative_to(skill_dir)
                errors.append(f"{name}: {relative_source} referenced resource does not exist: {value}")
        nested_text = source.read_text(encoding="utf-8")
        for value in sorted(referenced_paths(nested_text)):
            if value.startswith("@plugin/"):
                path = plugin_root / value[len("@plugin/") :]
            elif value.startswith("../"):
                path = source.parent / value
            else:
                path = skill_dir / value
            path = path.resolve()
            if not any(
                _is_relative_to(path, anchor) for anchor in _containment_anchors(plugin_root)
            ):
                errors.append(f"{name}: {source.relative_to(skill_dir)} resource escapes plugin: {value}")
                continue
            if any(char in value for char in "*?["):
                exists = path.parent.exists() and any(path.parent.glob(path.name))
            else:
                exists = path.exists()
            if not exists:
                errors.append(
                    f"{name}: {source.relative_to(skill_dir)} referenced code/resource path does not exist: {value}"
                )

    detail = {
        "name": name,
        "domain": domain or "",
        "function": function or "",
        "version": str(version or ""),
        "description_length": len(description),
        "line_count": len(text.splitlines()),
        "alias": alias,
        "resource_count": len(referenced_paths(body)),
        "errors": errors,
        "warnings": warnings,
    }
    return errors, warnings, detail


def validate(plugin_root: Path) -> dict[str, Any]:
    ensure_skill_links(plugin_root)
    skills_root = plugin_root / "skills"
    errors: list[str] = []
    warnings: list[str] = []
    details: list[dict[str, Any]] = []
    if not skills_root.is_dir():
        errors.append(f"missing skills directory: {skills_root}")
    else:
        for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
            skill_errors, skill_warnings, detail = validate_skill(skill_dir)
            errors.extend(skill_errors)
            warnings.extend(skill_warnings)
            details.append(detail)
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "reason_code": "SKILL_DESIGN_VALID" if not errors else "SKILL_DESIGN_INVALID",
        "skill_count": len(details),
        "errors": errors,
        "warnings": warnings,
        "skills": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, default=PLUGIN_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        frontmatter, body = parse_frontmatter("---\nname: demo\ndescription: demo\n---\n# Role\n")
        assert frontmatter["name"] == "demo" and "# Role" in body
        assert "references/x.md" in referenced_paths("`references/x.md`")
        listed, _ = parse_frontmatter("---\nname: demo\ntags:\n  - ascend\n  - npu\n---\n# Role\n")
        assert listed["tags"] == ["ascend", "npu"]
        assert "@plugin/references/policy.md" in referenced_paths(
            "`<plugin-root>/references/policy.md`"
        )
        assert "@plugin/references/policy.md" in markdown_resource_paths(
            "[policy](<plugin-root>/references/policy.md)"
        )
        print("SELF_TEST_PASS")
        return 0
    result = validate(args.plugin_root.resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['reason_code']}")
        for item in result["errors"] + result["warnings"]:
            print(item, file=sys.stderr if item in result["errors"] else sys.stdout)
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
