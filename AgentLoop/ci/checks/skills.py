# CI Guard — L1 skill 静态结构检查 + L3 分类体系警告
# ==================================================
# 阻塞项对照 CLAUDE.md「技能评审清单」[阻塞] 部分：
#   frontmatter 完整性、命名规范、描述存在、正文非空、name 唯一、引用文件存在。
# 说明：description 的英文+前缀规范与分类体系一致性存在历史存量，
#       第一版以警告呈现，存量修复后再升级为阻塞。

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ci.common import (
    BLOCK,
    WARN,
    KNOWN_DOMAINS,
    KNOWN_DRIFT_DIRS,
    CheckResult,
    is_kebab,
    read_text,
    strip_frontmatter,
)

REF_RE = re.compile(r"(?:references|scripts|assets)/[A-Za-z0-9_./-]+")
DESC_PREFIX = "This skill should be used when"


def check_skills(repo: Path, skills: dict) -> list:
    results: list = []
    name_counter: dict = {}

    for skill_dir, fm in sorted(skills.items()):
        rel = skill_dir.relative_to(repo).as_posix()
        dirname = skill_dir.name
        body = strip_frontmatter(read_text(skill_dir / "SKILL.md"))

        # ---- 阻塞项 ----
        # 1. 目录命名规范
        if not is_kebab(dirname):
            results.append(CheckResult(
                "skill.naming", BLOCK,
                f"skill 目录名不符合 kebab-case 规范（3-40 字符、仅 [a-z0-9-]、无前导/尾随/连续连字符）：{dirname}", rel))
        # 2. frontmatter 完整性
        if not fm:
            results.append(CheckResult("skill.frontmatter", BLOCK, "frontmatter 缺失或无法解析", rel))
            continue  # 无法解析时后续检查无意义
        name = fm.get("name")
        if not isinstance(name, str) or not name.strip():
            results.append(CheckResult("skill.frontmatter", BLOCK, "frontmatter 缺少 name 字段", rel))
        elif name != dirname:
            results.append(CheckResult(
                "skill.naming", BLOCK,
                f"frontmatter name（{name}）与目录名（{dirname}）不一致", rel))
        description = fm.get("description")
        if not isinstance(description, str) or not description.strip():
            results.append(CheckResult("skill.frontmatter", BLOCK, "frontmatter 缺少 description 字段", rel))
        # 3. 正文非空
        if not body.strip():
            results.append(CheckResult("skill.content", BLOCK, "SKILL.md 正文为空", rel))

        # ---- 警告项（存量问题，逐步修复后可升级为阻塞）----
        if isinstance(description, str) and description.strip():
            if not description.startswith(DESC_PREFIX):
                results.append(CheckResult(
                    "skill.description-prefix", WARN,
                    f'description 未以 "{DESC_PREFIX}" 开头（i18n 规范）', rel))
            if len(description) > 1024:
                results.append(CheckResult(
                    "skill.description-len", WARN,
                    f"description 超过 1024 字符（当前 {len(description)}）", rel))
        version = fm.get("version")
        if version is not None and (not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version)):
            results.append(CheckResult("skill.version", WARN, f"version 非语义版本号：{version!r}", rel))
        tags = fm.get("tags")
        if tags is None:
            results.append(CheckResult("skill.tags", WARN, "frontmatter 缺少 tags 字段（推荐，最多 5 个）", rel))
        elif not isinstance(tags, list) or len(tags) > 5:
            results.append(CheckResult("skill.tags", WARN, f"tags 超过 5 个或不是数组（当前 {tags!r}）", rel))

        # 4. name 全局唯一（累计统计）
        if isinstance(name, str):
            name_counter.setdefault(name, []).append(rel)

        # 5. 正文引用的本地文件存在
        seen_refs = set()
        for m in REF_RE.findall(body):
            if m in seen_refs:
                continue
            seen_refs.add(m)
            if not (skill_dir / m).exists():
                results.append(CheckResult(
                    "skill.reference", WARN, f"正文引用的本地文件不存在：{m}", rel))

        # ---- L3 分类体系一致性（警告）----
        parts = rel.split("/")
        # 期望深度 3：skills/<领域>/<功能>/<skill>
        if len(parts) != 4:
            results.append(CheckResult(
                "skill.path-depth", WARN,
                f"目录深度异常（期望 skills/<领域>/<功能>/<skill> 三层，实际 {len(parts) - 1} 层）", rel))
        if len(parts) >= 3:
            domain_dir = parts[1]
            function_dir = parts[2]
            if domain_dir not in KNOWN_DOMAINS and domain_dir not in KNOWN_DRIFT_DIRS:
                results.append(CheckResult(
                    "skill.domain", WARN,
                    f"领域目录 {domain_dir} 不在 CLAUDE.md 分类体系中", rel))
            allowed = KNOWN_DOMAINS.get(domain_dir)
            if allowed is not None and function_dir not in allowed:
                results.append(CheckResult(
                    "skill.function", WARN,
                    f"功能目录 {function_dir} 不在领域 {domain_dir} 的分类体系中（允许：{sorted(allowed)}）", rel))

    # 重复 name → 阻塞
    for name, paths in sorted(name_counter.items()):
        if len(paths) > 1:
            results.append(CheckResult(
                "skill.duplicate", BLOCK,
                f"skill name {name!r} 重复出现于 {len(paths)} 个目录", paths[0]))

    return results


def collect_skill_names(skills: dict) -> set:
    """收集全部 skill 名（frontmatter name + 目录名），供 plugin/agent 引用校验。"""
    names: set = set()
    for skill_dir, fm in skills.items():
        names.add(skill_dir.name)
        name = fm.get("name") if fm else None
        if isinstance(name, str) and name.strip():
            names.add(name)
    return names


def count_skills_under(path: Path) -> int:
    """统计某目录下（含子目录）的 SKILL.md 数量。"""
    if not path.is_dir():
        return 0
    return len(list(path.rglob("SKILL.md")))


def iter_skill_dirs(skills: dict) -> Iterable:
    return skills.keys()
