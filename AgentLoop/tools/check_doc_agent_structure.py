#!/usr/bin/env python3
"""Doc-Agent structural consistency checker for MindSpeed-Agent.

Validates the invariants that keep the documentation-agent system self-consistent:

  1. skills/doc-skills: the 5 thin skills have valid frontmatter and non-empty bodies;
  2. plugins/doc-agent agents: valid frontmatter, tools non-empty, and every
     referenced skill is registered in the plugin manifest;
  3. Claude Code/Codex plugin manifests: versions match and skill paths stay
     inside the plugin root; generated skill links or staged copies match the
     canonical skill sources when present;
  4. contracts: every JSON Schema file parses and has the core structural keys;
  5. rules.yaml: rule ids unique and complete, deterministic rules have a linter;
  6. templates/repo_manifest.yaml: entry ids unique and complete, `when`
     conditions legal, required/optional partitions present;
  7. engine and linters: every .py file compiles;
  8. legacy layout: removed remnants (doc-check-*, doc-workflow, _tools) are gone;
  9. marketplace: doc-agent version matches the plugin manifest.

Usage:
    python tools/check_doc_agent_structure.py [--repo PATH]

Exit code 0 when no errors are found; 1 when consistency errors exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import re
import sys

SKILLS_ROOT = pathlib.Path("skills") / "doc-skills"
PLUGIN_ROOT = pathlib.Path("plugins") / "doc-agent"
AGENTS_ROOT = PLUGIN_ROOT / "agents"
CONTRACTS_ROOT = PLUGIN_ROOT / "contracts"
RULES_PATH = PLUGIN_ROOT / "rules" / "rules.yaml"
SKELETONS_PATH = PLUGIN_ROOT / "templates" / "skeletons.yaml"
MANIFEST_PATH = PLUGIN_ROOT / "templates" / "repo_manifest.yaml"
ENGINE_ROOT = PLUGIN_ROOT / "engine"
LINTERS_ROOT = PLUGIN_ROOT / "linters"
MARKETPLACE_PATH = pathlib.Path(".claude-plugin") / "marketplace.json"

ALLOWED_FUNCTIONS = {"common", "generation", "review"}
EXPECTED_SKILLS = {
    "doc-orchestrator",
    "doc-design",
    "doc-generate",
    "doc-update",
    "doc-quality-check",
}
PUBLIC_SKILLS = {"doc-orchestrator"}
PUBLIC_AGENTS = {"doc-orchestrator"}
LEGACY_DIRS = ["_tools", "doc-workflow", "doc-check-correctness", "doc-check-completeness",
               "doc-check-clarity", "doc-check-accessibility"]
CONTRACT_FILES = ["issue.json", "context_report.json", "design_plan.json",
                  "figure_requests.json", "quality_report.json", "memory_event.json",
                  "code_verify.json", "lint_report.json", "review_report.json"]
RULE_ID_RE = re.compile(r"^R-(COR|COM|CLA|ACC)-\d{3}$")


def parse_frontmatter(text: str) -> dict:
    """Parse the simple YAML subset used by SKILL.md/agent.md files."""
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    raw = m.group(1).splitlines()
    key_re = re.compile(r"^([A-Za-z_][\w-]*):[ \t]*(.*)$")
    data: dict = {}
    i = 0

    def parse_inline(value: str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return [x.strip() for x in value[1:-1].split(",") if x.strip()]
        return value

    while i < len(raw):
        line = raw[i]
        km = key_re.match(line)
        if not km:
            i += 1
            continue
        key, inline = km.group(1), km.group(2).strip()
        i += 1
        if inline:
            if inline.startswith((">", "|")):
                parts = []
                while i < len(raw):
                    ln = raw[i]
                    if ln.strip() == "":
                        i += 1
                        continue
                    if ln[0] in (" ", "\t"):
                        parts.append(ln.strip())
                        i += 1
                    else:
                        break
                data[key] = " ".join(parts)
            else:
                data[key] = parse_inline(inline)
            continue
        block = []
        while i < len(raw):
            ln = raw[i]
            if ln.strip() == "":
                i += 1
                continue
            if not ln[0].isspace() and key_re.match(ln):
                break
            block.append(ln)
            i += 1
        if not block:
            data[key] = []
            continue
        items = [re.match(r"^[ \t]+- (.+)$", b) for b in block]
        if all(items):
            data[key] = [it.group(1).strip() for it in items if it]
        else:
            data[key] = " ".join(b.strip() for b in block)
    return data


def _check_md_only(root: pathlib.Path, folder: pathlib.Path, errors: list[str]) -> None:
    """Role/knowledge layers must stay prompt files: only .md allowed."""
    if not folder.is_dir():
        return
    for f in sorted(folder.rglob('*')):
        if f.is_file() and f.suffix.lower() != '.md':
            errors.append(f"{f.relative_to(root)}: agents/skills layers allow only .md files "
                          f"(no code; see DESIGN.md role/knowledge boundary)")


def validate_skills(root: pathlib.Path, errors: list[str]) -> dict[str, pathlib.Path]:
    index: dict[str, pathlib.Path] = {}
    if not (root / SKILLS_ROOT).is_dir():
        errors.append(f"{SKILLS_ROOT}: skills directory missing")
        return index
    _check_md_only(root, root / SKILLS_ROOT, errors)
    for skill_md in sorted((root / SKILLS_ROOT).rglob("SKILL.md")):
        rel = skill_md.relative_to(root)
        fm = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = fm.get("name")
        if not name:
            errors.append(f"{rel}: frontmatter missing 'name'")
            continue
        if name != skill_md.parent.name:
            errors.append(f"{rel}: directory name '{skill_md.parent.name}' != skill name '{name}'")
        if name not in EXPECTED_SKILLS:
            errors.append(f"{rel}: unexpected skill '{name}' (current design keeps only {sorted(EXPECTED_SKILLS)})")
        desc = str(fm.get("description", ""))
        if not desc.startswith("This skill should be used when"):
            errors.append(f"{rel}: description must start with 'This skill should be used when'")
        if not re.match(r"^\d+\.\d+\.\d+$", str(fm.get("version", ""))):
            errors.append(f"{rel}: version must be x.y.z")
        if fm.get("domain") != "doc":
            errors.append(f"{rel}: domain must be 'doc'")
        if fm.get("function") not in ALLOWED_FUNCTIONS:
            errors.append(f"{rel}: function must be one of {sorted(ALLOWED_FUNCTIONS)}")
        if not fm.get("author"):
            errors.append(f"{rel}: frontmatter missing 'author'")
        visibility = str(fm.get("user-invocable", "")).lower()
        if visibility not in {"true", "false"}:
            errors.append(f"{rel}: user-invocable must be true or false")
        elif (name in PUBLIC_SKILLS) != (visibility == "true"):
            expected = "true" if name in PUBLIC_SKILLS else "false"
            errors.append(f"{rel}: user-invocable must be {expected} for its plugin boundary")
        body = text_after_fm(skill_md.read_text(encoding="utf-8"))
        if len(body.strip()) < 200:
            errors.append(f"{rel}: body too short ({len(body.strip())} chars)")
        index[name] = skill_md.parent
    missing = EXPECTED_SKILLS - set(index)
    if missing:
        errors.append(f"{SKILLS_ROOT}: missing skills: {sorted(missing)}")
    return index


def text_after_fm(text: str) -> str:
    m = re.match(r"^---\n.*?\n---\n?", text, re.S)
    return text[m.end():] if m else text


def validate_agents(root: pathlib.Path, skill_index: dict[str, pathlib.Path],
                    errors: list[str]) -> tuple[int, set[str]]:
    agent_dir = root / AGENTS_ROOT
    referenced: set[str] = set()
    if not agent_dir.is_dir():
        errors.append(f"{AGENTS_ROOT}: agents directory missing")
        return 0, referenced
    _check_md_only(root, agent_dir, errors)
    count = 0
    for folder in sorted(agent_dir.iterdir()):
        if not folder.is_dir():
            continue
        count += 1
        md = folder / "agent.md"
        rel = md.relative_to(root)
        if not md.exists():
            errors.append(f"{rel}: agent.md missing")
            continue
        text = md.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        name = fm.get("name")
        if not name:
            errors.append(f"{rel}: frontmatter missing 'name'")
        elif name != folder.name:
            errors.append(f"{rel}: folder name '{folder.name}' != agent name '{name}'")
        if not fm.get("description"):
            errors.append(f"{rel}: frontmatter missing 'description'")
        if not re.match(r"^\d+\.\d+\.\d+$", str(fm.get("version", ""))):
            errors.append(f"{rel}: version must be x.y.z")
        if not fm.get("author"):
            errors.append(f"{rel}: frontmatter missing 'author'")
        agent_visibility = str(fm.get("user-invocable", "")).lower()
        if agent_visibility not in {"true", "false"}:
            errors.append(f"{rel}: user-invocable must be true or false")
        elif (name in PUBLIC_AGENTS) != (agent_visibility == "true"):
            expected = "true" if name in PUBLIC_AGENTS else "false"
            errors.append(f"{rel}: user-invocable must be {expected} for its plugin boundary")
        skills = fm.get("skills", [])
        if not skills:
            errors.append(f"{rel}: skills list is empty")
        for skill in skills:
            if skill not in skill_index:
                errors.append(f"{rel}: agent references unknown skill '{skill}'")
            else:
                referenced.add(skill)
        tools = fm.get("tools", [])
        if not tools:
            errors.append(f"{rel}: tools list is empty")
        for tool in tools:
            if not tool:
                errors.append(f"{rel}: empty tool entry in tools list")
        if len(text_after_fm(text).strip()) < 300:
            errors.append(f"{rel}: body too short")
    return count, referenced


def _load_json_object(path: pathlib.Path, root: pathlib.Path,
                      errors: list[str]) -> dict:
    if not path.exists():
        errors.append(f"{path.relative_to(root)}: file missing")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.relative_to(root)}: invalid JSON ({exc})")
        return {}
    if not isinstance(data, dict):
        errors.append(f"{path.relative_to(root)}: root must be an object")
        return {}
    return data


def _same_directory_contents(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Return whether two materialized directory trees contain identical files."""
    try:
        left_files = {
            path.relative_to(left): path.read_bytes()
            for path in left.rglob("*") if path.is_file()
        }
        right_files = {
            path.relative_to(right): path.read_bytes()
            for path in right.rglob("*") if path.is_file()
        }
    except OSError:
        return False
    return left_files == right_files


def validate_plugins(root: pathlib.Path, skill_index: dict[str, pathlib.Path],
                     referenced: set[str], errors: list[str],
                     warnings: list[str]) -> str:
    claude_manifest = root / PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
    codex_manifest = root / PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
    claude = _load_json_object(claude_manifest, root, errors)
    codex = _load_json_object(codex_manifest, root, errors)
    if not claude or not codex:
        return ""

    for manifest, data in ((claude_manifest, claude), (codex_manifest, codex)):
        if data.get("name") != PLUGIN_ROOT.name:
            errors.append(f"{manifest.relative_to(root)}: plugin name must be '{PLUGIN_ROOT.name}'")
    claude_version = str(claude.get("version", ""))
    codex_version = str(codex.get("version", ""))
    if claude_version != codex_version:
        errors.append(f"doc-agent manifest versions differ: Claude '{claude_version}' != Codex '{codex_version}'")

    declared = claude.get("skills")
    if not isinstance(declared, list) or not all(isinstance(p, str) for p in declared):
        errors.append(f"{claude_manifest.relative_to(root)}: 'skills' must be a string list")
        declared = []
    expected_paths = {f"./skills/{name}" for name in EXPECTED_SKILLS}
    if set(declared) != expected_paths:
        errors.append(f"{claude_manifest.relative_to(root)}: skills must be exactly {sorted(expected_paths)}")
    for skill_path in declared:
        if ".." in pathlib.PurePosixPath(skill_path).parts:
            errors.append(f"{claude_manifest.relative_to(root)}: skill path escapes plugin root: {skill_path}")

    if codex.get("skills") != "./skills/":
        errors.append(f"{codex_manifest.relative_to(root)}: 'skills' must be './skills/'")
    interface = codex.get("interface")
    if not isinstance(interface, dict):
        errors.append(f"{codex_manifest.relative_to(root)}: interface object missing")
    else:
        for key in ("displayName", "shortDescription", "longDescription",
                    "developerName", "category", "capabilities", "defaultPrompt"):
            if not interface.get(key):
                errors.append(f"{codex_manifest.relative_to(root)}: interface.{key} missing")

    registered = {pathlib.PurePosixPath(p).name for p in declared}
    for skill in sorted(referenced - registered):
        errors.append(f"{claude_manifest.relative_to(root)}: agent-referenced skill '{skill}' not registered")
    for skill in sorted(EXPECTED_SKILLS - registered):
        errors.append(f"{claude_manifest.relative_to(root)}: expected skill '{skill}' not registered")

    aggregate = root / PLUGIN_ROOT / "skills"
    if not aggregate.is_dir():
        warnings.append(f"{aggregate.relative_to(root)}: generated skill links absent; run install.bat/install.sh")
    else:
        for name, source in sorted(skill_index.items()):
            link = aggregate / name
            if not link.is_dir():
                errors.append(f"{link.relative_to(root)}: generated skill link missing")
            elif link.resolve() == source.resolve():
                continue
            elif _same_directory_contents(link, source):
                warnings.append(
                    f"{link.relative_to(root)}: exact staged copy present instead of generated link"
                )
            else:
                errors.append(
                    f"{link.relative_to(root)}: neither canonical link nor exact staged copy"
                )

    if not (root / PLUGIN_ROOT / "AGENTS.md").is_file():
        errors.append(f"{PLUGIN_ROOT / 'AGENTS.md'}: Codex orchestration instructions missing")
    return claude_version


def validate_contracts(root: pathlib.Path, errors: list[str]) -> None:
    for fname in CONTRACT_FILES:
        path = root / CONTRACTS_ROOT / fname
        if not path.exists():
            errors.append(f"{path.relative_to(root)}: missing contract file")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(root)}: invalid JSON ({exc})")
            continue
        if not isinstance(data, dict) or data.get("type") != "object":
            errors.append(f"{path.relative_to(root)}: schema must be an object type")
        if "properties" not in data or "required" not in data:
            errors.append(f"{path.relative_to(root)}: schema missing properties/required")


def validate_rules(root: pathlib.Path, errors: list[str]) -> None:
    path = root / RULES_PATH
    if not path.exists():
        errors.append(f"{RULES_PATH}: rules.yaml missing")
        return
    text = path.read_text(encoding="utf-8")
    ids = re.findall(r"^\s*-?\s*id:\s*(R-(?:COR|COM|CLA|ACC)-\d{3})\s*$", text, re.M)
    for rid in ids:
        if not RULE_ID_RE.match(rid):
            errors.append(f"{RULES_PATH}: malformed rule id '{rid}'")
    if len(ids) != len(set(ids)):
        errors.append(f"{RULES_PATH}: duplicate rule ids")
    if len(ids) < 28:
        errors.append(f"{RULES_PATH}: expected 28 rules, found {len(ids)}")
    # deterministic rules must declare a linter
    for block in text.split("- id:"):
        block = block.lstrip()
        if not block.startswith("R-"):
            continue
        rid = block.split("\n", 1)[0].strip()
        if re.search(r"^\s*kind:\s*deterministic\s*$", block, re.M) and \
                not re.search(r"^\s*linter:\s*\S+\s*$", block, re.M):
            errors.append(f"{RULES_PATH}: deterministic rule {rid} missing linter")
    for key in ("weights:", "floors:", "pass_total:", "max_rounds:"):
        if key not in text:
            errors.append(f"{RULES_PATH}: scoring block missing '{key}'")


def validate_manifest(root: pathlib.Path, errors: list[str]) -> int:
    if (root / SKELETONS_PATH).exists():
        errors.append(f"{SKELETONS_PATH}: legacy skeleton registry still present (superseded by repo_manifest.yaml)")
    path = root / MANIFEST_PATH
    if not path.exists():
        errors.append(f"{MANIFEST_PATH}: repo_manifest.yaml missing")
        return 0
    text = path.read_text(encoding="utf-8")
    if "正式定义" not in text:
        errors.append(f"{MANIFEST_PATH}: missing formal-definition marker")
    if "required:" not in text or "optional:" not in text:
        errors.append(f"{MANIFEST_PATH}: missing required/optional partitions")
        return 0
    ids = re.findall(r"^  - id:\s*(D-[A-Z0-9-]+)\s*$", text, re.M)
    if len(ids) != len(set(ids)):
        errors.append(f"{MANIFEST_PATH}: duplicate entry ids")
    required_ids = re.findall(
        r"^  - id:\s*(D-[A-Z0-9-]+)\s*$",
        text.split("optional:")[0], re.M)
    optional_ids = re.findall(
        r"^  - id:\s*(D-[A-Z0-9-]+)\s*$",
        text.split("optional:", 1)[1], re.M)
    if len(required_ids) < 14:
        errors.append(f"{MANIFEST_PATH}: expected ≥14 required entries, found {len(required_ids)}")
    if len(optional_ids) < 6:
        errors.append(f"{MANIFEST_PATH}: expected ≥6 optional entries, found {len(optional_ids)}")
    # each entry must declare a name; `when` values are constrained
    for block in text.split("- id:"):
        block = block.lstrip()
        if not block.startswith("D-"):
            continue
        eid = block.split("\n", 1)[0].strip()
        if "name:" not in block:
            errors.append(f"{MANIFEST_PATH}: entry {eid} missing name")
        for when in re.findall(r"^  when:\s*(\S+)\s*$", block, re.M):
            if when not in ("repo_has_models", "repo_has_toolchain"):
                errors.append(f"{MANIFEST_PATH}: entry {eid} has unknown when condition '{when}'")
    return len(ids)


def validate_python(root: pathlib.Path, errors: list[str], warnings: list[str]) -> None:
    for folder in (ENGINE_ROOT, LINTERS_ROOT):
        for py_file in sorted((root / folder).glob("*.py")):
            rel = py_file.relative_to(root)
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"{rel}: compile error: {exc}")
            except OSError as exc:
                warnings.append(f"{rel}: cannot compile ({exc})")


def validate_legacy(root: pathlib.Path, errors: list[str]) -> None:
    base = root / SKILLS_ROOT
    for parent in (base, base / "common", base / "generation", base / "review"):
        for d in LEGACY_DIRS:
            if (parent / d).is_dir():
                errors.append(f"{parent.relative_to(root) / d}: legacy directory still present")
    if (base / "generation" / "references" / "design_decision_guide.md").exists():
        errors.append(f"{SKILLS_ROOT}/generation/references/design_decision_guide.md: legacy reference still present")


def validate_marketplace(root: pathlib.Path, plugin_version: str,
                         errors: list[str], warnings: list[str]) -> None:
    mp = root / MARKETPLACE_PATH
    if not mp.exists():
        warnings.append(f"{MARKETPLACE_PATH}: missing")
        return
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{MARKETPLACE_PATH}: invalid JSON ({exc})")
        return
    entry = next((p for p in data.get("plugins", []) if p.get("name") == "doc-agent"), None)
    if entry is None:
        errors.append("marketplace.json has no doc-agent entry")
    elif plugin_version and entry.get("version") != plugin_version:
        errors.append(f"marketplace doc-agent version '{entry.get('version')}' != plugin version '{plugin_version}'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="MindSpeed-Agent repo root")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.repo).resolve() if args.repo else pathlib.Path(__file__).resolve().parent.parent
    if not root.is_dir():
        print(f"ERROR: repo root not found: {root}")
        return 2
    errors: list[str] = []
    warnings: list[str] = []

    skill_index = validate_skills(root, errors)
    agent_count, referenced = validate_agents(root, skill_index, errors)
    plugin_version = validate_plugins(root, skill_index, referenced, errors, warnings)
    validate_contracts(root, errors)
    validate_rules(root, errors)
    manifest_entries = validate_manifest(root, errors)
    validate_python(root, errors, warnings)
    validate_legacy(root, errors)
    validate_marketplace(root, plugin_version, errors, warnings)

    print(f"Doc-Agent consistency check ({root})")
    print(f"  skills:   {len(skill_index)} found")
    print(f"  agents:   {agent_count} found")
    print(f"  manifest entries: {manifest_entries}")
    print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")
    else:
        print("ERRORS: none")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")
    print()
    ok = not errors
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
