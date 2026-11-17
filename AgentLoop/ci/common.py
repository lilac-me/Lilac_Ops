# CI Guard — 公共工具
# ===================
# frontmatter 解析（PyYAML 优先、正则回退）、检查结果结构、仓库发现工具。

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None

BLOCK = "block"
WARN = "warn"

KABAB_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
# CLAUDE.md 领域与功能分类体系（目录名 → 该领域允许的功能目录）
KNOWN_DOMAINS = {
    "llm-model-skills": {"deployment", "model-analysis", "model-adaptation", "alignment",
                         "performance-optimization", "community-operations", "common"},
    "mm-model-skills": {"deployment", "model-analysis", "model-adaptation", "alignment",
                        "performance-optimization", "community-operations", "common"},
    "rl-model-skills": {"deployment", "model-analysis", "model-adaptation", "alignment",
                        "performance-optimization", "community-operations", "common"},
    "inference-model-skills": {"deployment", "model-analysis", "model-adaptation", "alignment",
                               "performance-optimization", "community-operations", "common"},
    "ops-skills": {"ascendc", "triton", "tilelang", "common"},
    "common": {"common"},
    "vla-model-skills": {"vla-model-adaptation-optimization"},
    "doc-skills": {"common", "generation", "review", "_tools"},
    "ecosystem-code-repository-npu-migration": {"deployment", "model-analysis", "model-adaptation", "common"},
}
# 历史遗留目录名（分类体系漂移，仅提示、不阻塞）
KNOWN_DRIFT_DIRS = {"llm-skills", "common_tools", "supporting-skills", "model_migration",
                    "bridge_model_migration", "fsdp2-auto-optimization", "fsdp2-model-adaptation",
                    "npu-migration-skills", "veomni-skills", "rl-model-skills"}


@dataclass
class CheckResult:
    """单条检查结果：check=检查项 id，level=block/warn，message=描述，path=相关文件。"""
    check: str
    level: str
    message: str
    path: str = ""

    def to_dict(self) -> dict:
        return {"check": self.check, "level": self.level, "message": self.message, "path": self.path}


def parse_frontmatter(text: str) -> dict:
    """解析 SKILL.md/AGENT.md 的 YAML frontmatter。PyYAML 优先，缺失时正则回退。"""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    raw = m.group(1)
    if yaml is not None:
        try:
            data = yaml.safe_load(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    # 正则回退：支持 key: value、内联列表、引号字符串
    data: dict = {}
    key_re = re.compile(r"^([A-Za-z_][\w-]*):[ \t]*(.*)$")
    for line in raw.splitlines():
        km = key_re.match(line)
        if not km:
            continue
        key, value = km.group(1), km.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            data[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        elif value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1]
        else:
            data[key] = value
    return data


def strip_frontmatter(text: str) -> str:
    """去掉 frontmatter 后的正文。"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    return text[m.end():] if m else text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_kebab(name: str, min_len: int = 3, max_len: int = 40) -> bool:
    """kebab-case 校验：仅 [a-z0-9-]、无前导/尾随/连续连字符、长度区间。"""
    if not (min_len <= len(name) <= max_len):
        return False
    if not KABAB_RE.fullmatch(name):
        return False
    return "--" not in name


# ---------------------------------------------------------------------------
# 仓库发现
# ---------------------------------------------------------------------------

def discover_skills(repo: Path) -> dict:
    """遍历 skills/ 下所有 SKILL.md。返回 {skill_dir: frontmatter}。"""
    out: dict = {}
    skills_root = repo / "skills"
    if not skills_root.is_dir():
        return out
    for sk in sorted(skills_root.rglob("SKILL.md")):
        fm = parse_frontmatter(read_text(sk))
        out[sk.parent] = fm
    return out


def discover_plugins(repo: Path) -> list:
    """发现根插件 + plugins/* 插件。返回 [{dir, manifest, data, manifest_error}]。"""
    out: list = []

    def _entry(pdir: Path) -> dict:
        manifest = pdir / ".claude-plugin" / "plugin.json"
        data: Optional[dict] = None
        manifest_error = ""
        if manifest.exists():
            try:
                loaded = json.loads(read_text(manifest))
                data = loaded if isinstance(loaded, dict) else None
                if data is None:
                    manifest_error = "plugin.json 顶层不是 JSON 对象"
            except json.JSONDecodeError as e:
                manifest_error = f"plugin.json 解析失败: {e}"
        return {"dir": pdir, "manifest": manifest, "data": data, "manifest_error": manifest_error}

    # 根插件
    out.append(_entry(repo))
    # plugins/ 下的插件
    plugins_root = repo / "plugins"
    if plugins_root.is_dir():
        for pdir in sorted(p for p in plugins_root.iterdir() if p.is_dir()):
            out.append(_entry(pdir))
    return out


def find_skill_by_basename(repo: Path, basename: str) -> list:
    """在 skills/ 下按目录 basename 查找 skill 目录（用于 symlink 缺失时的疑似目标定位）。"""
    skills_root = repo / "skills"
    if not skills_root.is_dir():
        return []
    return sorted(d for d in skills_root.rglob(basename) if d.is_dir())
