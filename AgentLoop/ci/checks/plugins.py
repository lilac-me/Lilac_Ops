# CI Guard — L1 插件结构检查
# ==========================
# plugin.json 合法性、skills 路径解析（含 symlink 缺失判定）、
# agent 定义与引用校验、每插件 skill 数 ≤ 30（警告）。

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ci.common import (
    BLOCK,
    WARN,
    CheckResult,
    find_skill_by_basename,
    parse_frontmatter,
    read_text,
)

SKILL_CAP_PER_PLUGIN = 30  # CLAUDE.md：每个插件内的 skills 总数 ≤ 30


def check_plugins(repo: Path, plugins: list, skills: dict) -> list:
    results: list = []
    skill_names = {d.name for d in skills} | {fm.get("name") for fm in skills.values() if isinstance(fm.get("name"), str)}

    for p in plugins:
        pdir: Path = p["dir"]
        is_root = pdir == repo
        prel = pdir.relative_to(repo).as_posix() or "."
        data: Optional[dict] = p["data"]

        # 无 manifest：不作为插件发布，警告
        if not p["manifest"].exists():
            results.append(CheckResult(
                "plugin.manifest", WARN,
                "插件目录缺少 .claude-plugin/plugin.json（不作为插件发布？）", prel))
            continue

        # manifest 解析失败 → 阻塞
        if p["manifest_error"]:
            results.append(CheckResult("plugin.manifest", BLOCK, p["manifest_error"], prel))
            continue
        if data is None:
            results.append(CheckResult("plugin.manifest", BLOCK, "plugin.json 顶层不是 JSON 对象", prel))
            continue

        # name / version
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            results.append(CheckResult("plugin.manifest", BLOCK, "plugin.json 缺少 name", prel))
        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            results.append(CheckResult("plugin.manifest", BLOCK, "plugin.json 缺少 version", prel))
        elif not re.fullmatch(r"\d+\.\d+\.\d+", version):
            results.append(CheckResult(
                "plugin.version", WARN, f"version 非语义版本号：{version!r}", prel))

        # skills 路径解析
        declared = data.get("skills", [])
        if not isinstance(declared, list):
            results.append(CheckResult("plugin.skills", WARN, "skills 字段缺失或不是数组", prel))
            declared = []
        resolved_skill_dirs: list = []
        for entry in declared:
            target = (pdir / entry).resolve()
            entry_rel = (pdir / entry).as_posix()
            if target.is_dir():
                resolved_skill_dirs.append(target)
                continue
            # 缺失：疑似 install.sh 的 symlink 未创建（按 basename 找目标）
            basename = Path(entry).name
            candidates = find_skill_by_basename(repo, basename)
            if len(candidates) == 1:
                results.append(CheckResult(
                    "plugin.skills-symlink", WARN,
                    f"skills 路径 {entry_rel} 不存在（symlink 未创建？运行 install.sh），疑似目标：{candidates[0].relative_to(repo).as_posix()}", prel))
                resolved_skill_dirs.append(candidates[0])
            else:
                results.append(CheckResult(
                    "plugin.skills", BLOCK,
                    f"skills 路径 {entry_rel} 悬空：目录不存在且未找到同名 skill（匹配 {len(candidates)} 个）", prel))

        # 每插件 skill 数 ≤ 30（CLAUDE.md 会话上限约定）。
        # v2 按需安装架构下（docs/install-on-demand-design.md）每 unit 收敛到
        # ≤30 是硬约束——超限会导致 Claude Code 会话静默截断技能，故为 BLOCK。
        total = 0
        for d in resolved_skill_dirs:
            total += len(list(d.rglob("SKILL.md"))) if d.is_dir() else 0
        if total > SKILL_CAP_PER_PLUGIN:
            results.append(CheckResult(
                "plugin.skill-cap", BLOCK,
                f"插件挂载 {total} 个 skill，超过每插件 ≤ {SKILL_CAP_PER_PLUGIN} 的硬约束", prel))

        # agents 校验（AGENT.md 为 Claude Code 约定；agent.md 为 doc-agent/ops 历史约定）
        agents_dir = pdir / "agents"
        if agents_dir.is_dir():
            for adir in sorted(a for a in agents_dir.iterdir() if a.is_dir()):
                arel = adir.relative_to(repo).as_posix()
                md = adir / "AGENT.md"
                legacy_md = adir / "agent.md"
                if not md.exists() and not legacy_md.exists():
                    results.append(CheckResult("agent.missing", BLOCK, "agent 目录缺少 AGENT.md", arel))
                    continue
                if not md.exists():
                    md = legacy_md
                    results.append(CheckResult(
                        "agent.legacy-case", WARN,
                        "使用小写 agent.md——Linux 大小写敏感，Claude Code 可能无法发现，建议重命名为 AGENT.md", arel))
                text = read_text(md)
                fm = parse_frontmatter(text)
                if not fm:
                    results.append(CheckResult("agent.frontmatter", BLOCK, f"{md.name} frontmatter 缺失或无法解析", arel))
                agent_name = fm.get("name")
                if not isinstance(agent_name, str) or not agent_name.strip():
                    results.append(CheckResult("agent.frontmatter", BLOCK, f"{md.name} frontmatter 缺少 name", arel))
                # agent 引用的 skills 必须真实存在
                refs = fm.get("skills") or []
                if isinstance(refs, list):
                    for s in refs:
                        if s not in skill_names:
                            results.append(CheckResult(
                                "agent.skills", BLOCK,
                                f"agent 引用的 skill {s!r} 在全仓库不存在", arel))
                # agent.yaml（若存在）可解析
                yaml_path = adir / "agent.yaml"
                if yaml_path.exists():
                    try:
                        import yaml as _yaml
                    except ImportError:
                        _yaml = None
                    if _yaml is not None:
                        try:
                            spec = _yaml.safe_load(read_text(yaml_path))
                            if not isinstance(spec, dict):
                                results.append(CheckResult("agent.yaml", WARN, "agent.yaml 顶层不是映射", arel))
                            else:
                                yrefs = spec.get("skills") or []
                                if isinstance(yrefs, list):
                                    for s in yrefs:
                                        if s not in skill_names:
                                            results.append(CheckResult(
                                                "agent.skills", BLOCK,
                                                f"agent.yaml 引用的 skill {s!r} 在全仓库不存在", arel))
                        except Exception as e:
                            results.append(CheckResult("agent.yaml", WARN, f"agent.yaml 解析失败: {e}", arel))
                    else:
                        results.append(CheckResult("agent.yaml", WARN, "PyYAML 缺失，跳过 agent.yaml 解析", arel))

    return results
