# -*- coding: utf-8 -*-
"""catalog 校验核心：图谱事实源 + SKILL.md 引用 + 金句断言的确定性检查。

同时被两处消费：
  - catalog/lint_catalog.py（本地 CLI 入口）
  - ci/checks/knowledge_graph.py（挂入 ci/guard.py，转成 CheckResult）

检查项（check id）：
  catalog.repos.id-unique       BLOCK   repo id 重复/缺失/非 kebab
  catalog.repos.kind            WARN    kind 不在已知枚举
  catalog.repos.alias-conflict  BLOCK   别名/ID 归一化后跨仓冲突
  catalog.repos.repo-unaffined  WARN    无任何插件 repo_affinity 声明亲缘的仓
  catalog.plugins.affinity      BLOCK   repo_affinity 引用了不存在的 repo id
  catalog.plugins.skills        BLOCK   插件 skills 引用了不存在的 skill name
  catalog.plugins.install       BLOCK/WARN  install 元数据（unit/default_installed/
                                          skills_root/install_hint/core 唯一性）
  catalog.plugins.workflow      BLOCK   workflow 非空时文件必须存在（相对仓库根）
  catalog.plugins.execution     BLOCK/WARN  execution 取值；state-machine 插件必须声明
                                          workflow + entry_agent + agents，且 agents
                                          目录（plugins/<dir>/agents/<agent>）存在
  catalog.taskkinds.refs        BLOCK   task-kind 引用了不存在的 plugin/skill
  catalog.skill.targets         BLOCK   SKILL.md targets-repo 无法解析到任一 repo
  catalog.intents.repo          BLOCK   金句确定性消歧未命中 expected_repos 或误中无关仓
  catalog.intents.task          WARN    金句任务维度触发词未命中 expected_task
  catalog.intents.plugin        WARN    金句 R3 插件投影未命中 expected_plugins
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from catalog.loader import Catalog

BLOCK = "block"
WARN = "warn"

KIND_ENUM = {
    "training", "llm-training", "mm-training", "inference", "ops",
    "runtime", "profiling", "upstream", "doc",
}
KABAK_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


@dataclass
class CatalogResult:
    check: str
    level: str
    message: str
    path: str = ""


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-/.]+", "", str(s).lower())


def run_checks(repo_root: Path) -> list:
    """Run every catalog check; return list[CatalogResult]."""
    results: list = []
    cat = Catalog(repo_root)
    repo_ids = {r.get("id") for r in cat.repos if r.get("id")}
    plugin_ids = {p.get("id") for p in cat.plugins if p.get("id")}
    task_ids = {t.get("id") for t in cat.task_kinds if t.get("id")}
    skill_names = set()
    for rel, fm in cat.skills.items():
        nm = fm.get("name")
        if isinstance(nm, str) and nm.strip():
            skill_names.add(nm)
        else:
            results.append(CatalogResult("catalog.skill.targets", BLOCK,
                                         f"SKILL.md frontmatter 缺少 name：{rel}", rel))

    # ------------------------------------------------------------- repo checks
    seen_norm: dict = {}
    for r in cat.repos:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid.strip():
            results.append(CatalogResult("catalog.repos.id-unique", BLOCK,
                                         "repo 节点缺少 id", "catalog/repos.yaml"))
            continue
        if not KABAK_RE.fullmatch(rid):
            results.append(CatalogResult("catalog.repos.id-unique", BLOCK,
                                         f"repo id 非 kebab-case：{rid!r}", "catalog/repos.yaml"))
        kind = r.get("kind")
        if kind not in KIND_ENUM:
            results.append(CatalogResult("catalog.repos.kind", WARN,
                                         f"repo {rid} 的 kind 不在已知枚举 {sorted(KIND_ENUM)}：{kind!r}",
                                         "catalog/repos.yaml"))
        # alias/ID 跨仓冲突（同一仓内自身重复不算冲突）
        cands = [rid] + [a for a in r.get("aliases", []) if isinstance(a, str)]
        for c in cands:
            key = _norm(c)
            seen_norm.setdefault(key, []).append(rid)
    for key, owners in sorted(seen_norm.items()):
        if len(set(owners)) > 1:
            results.append(CatalogResult(
                "catalog.repos.alias-conflict", BLOCK,
                f"归一化别名 {key!r} 同时属于多个仓：{sorted(set(owners))}",
                "catalog/repos.yaml"))

    # 无插件亲缘的仓 → warn
    affined = set()
    for p in cat.plugins:
        for rid in p.get("repo_affinity", []) or []:
            affined.add(rid)
    for r in cat.repos:
        rid = r.get("id")
        if rid and rid not in affined:
            results.append(CatalogResult(
                "catalog.repos.repo-unaffined", WARN,
                f"repo {rid} 未被任何插件 repo_affinity 声明亲缘", "catalog/repos.yaml"))

    # ----------------------------------------------------------- plugin checks
    for p in cat.plugins:
        pid = p.get("id")
        for rid in p.get("repo_affinity", []) or []:
            if rid not in repo_ids:
                results.append(CatalogResult(
                    "catalog.plugins.affinity", BLOCK,
                    f"插件 {pid} repo_affinity 引用了不存在的 repo：{rid}",
                    "catalog/plugins.yaml"))
        for sname in p.get("skills", []) or []:
            if sname not in skill_names:
                results.append(CatalogResult(
                    "catalog.plugins.skills", BLOCK,
                    f"插件 {pid} skills 引用了不存在的 skill：{sname}",
                    "catalog/plugins.yaml"))

    # --------------------------------------------- plugin install 元数据校验
    core_units = []
    for p in cat.plugins:
        pid = p.get("id")
        inst = p.get("install") or {}
        if not isinstance(inst, dict):
            results.append(CatalogResult(
                "catalog.plugins.install", BLOCK,
                f"插件 {pid} 缺少 install 元数据（unit/default_installed/skills_root）",
                "catalog/plugins.yaml"))
            continue
        unit = inst.get("unit")
        if unit not in ("core", "on-demand"):
            results.append(CatalogResult(
                "catalog.plugins.install", BLOCK,
                f"插件 {pid} install.unit 取值非法：{unit!r}（core|on-demand）",
                "catalog/plugins.yaml"))
        default = inst.get("default_installed")
        if unit == "core":
            if default is not True:
                results.append(CatalogResult(
                    "catalog.plugins.install", BLOCK,
                    f"core 插件 {pid} 的 default_installed 必须为 true",
                    "catalog/plugins.yaml"))
            core_units.append(pid)
        elif unit == "on-demand" and default is not False:
            results.append(CatalogResult(
                "catalog.plugins.install", BLOCK,
                f"on-demand 插件 {pid} 的 default_installed 必须为 false",
                "catalog/plugins.yaml"))
        root = inst.get("skills_root")
        if not isinstance(root, str) or not (repo_root / root).is_dir():
            results.append(CatalogResult(
                "catalog.plugins.install", BLOCK,
                f"插件 {pid} install.skills_root 不存在或非目录：{root!r}",
                "catalog/plugins.yaml"))
        # source_dir（可选）：插件目录 != plugins/<id> 时记录真实目录（如 ecosystem-code-repository-npu-migration
        # 目录被改名）。缺省推导为 plugins/<id>；声明时必须存在。
        src = inst.get("source_dir")
        if src is not None:
            if not isinstance(src, str) or not (repo_root / src).is_dir():
                results.append(CatalogResult(
                    "catalog.plugins.install", BLOCK,
                    f"插件 {pid} install.source_dir 不存在或非目录：{src!r}",
                    "catalog/plugins.yaml"))
        hint = inst.get("install_hint") or {}
        for plat in ("claude", "codex", "dsh"):
            if not isinstance(hint.get(plat), str) or not hint.get(plat).strip():
                results.append(CatalogResult(
                    "catalog.plugins.install", WARN,
                    f"插件 {pid} install.install_hint 缺少 {plat} 命令",
                    "catalog/plugins.yaml"))
    if len(core_units) != 1:
        results.append(CatalogResult(
            "catalog.plugins.install", BLOCK,
            f"必须恰好一个 core 插件（当前 {len(core_units)} 个：{sorted(core_units)}）",
            "catalog/plugins.yaml"))

    # ------------------------------------- plugin workflow / state-machine 元数据
    for p in cat.plugins:
        pid = p.get("id")
        wf = p.get("workflow")
        if isinstance(wf, str) and wf.strip():
            if not (repo_root / wf.strip()).is_file():
                results.append(CatalogResult(
                    "catalog.plugins.workflow", BLOCK,
                    f"插件 {pid} workflow 文件不存在：{wf.strip()!r}",
                    "catalog/plugins.yaml"))
        execution = p.get("execution") or "skills"
        if execution not in ("skills", "state-machine"):
            results.append(CatalogResult(
                "catalog.plugins.execution", BLOCK,
                f"插件 {pid} execution 取值非法：{execution!r}（skills|state-machine）",
                "catalog/plugins.yaml"))
        if execution == "state-machine":
            if not isinstance(wf, str) or not wf.strip():
                results.append(CatalogResult(
                    "catalog.plugins.execution", BLOCK,
                    f"state-machine 插件 {pid} 必须声明非空 workflow",
                    "catalog/plugins.yaml"))
            entry = p.get("entry_agent")
            if not isinstance(entry, str) or not entry.strip():
                results.append(CatalogResult(
                    "catalog.plugins.execution", BLOCK,
                    f"state-machine 插件 {pid} 缺少 entry_agent",
                    "catalog/plugins.yaml"))
            agents = [a for a in (p.get("agents") or []) if isinstance(a, str)]
            if not agents:
                results.append(CatalogResult(
                    "catalog.plugins.execution", BLOCK,
                    f"state-machine 插件 {pid} 缺少 agents 列表",
                    "catalog/plugins.yaml"))
            src = (p.get("install") or {}).get("source_dir") or f"plugins/{pid}"
            ag_root = repo_root / src / "agents"
            for ag in dict.fromkeys([entry] + agents):
                if not (ag_root / ag).is_dir():
                    results.append(CatalogResult(
                        "catalog.plugins.execution", BLOCK,
                        f"state-machine 插件 {pid} agents 目录不存在："
                        f"{ag_root / ag}（entry_agent/agents 须对应 plugins/{pid}/agents/ 下目录）",
                        "catalog/plugins.yaml"))

    # ---------------------------------------------------------- task-kind refs
    for t in cat.task_kinds:
        tid = t.get("id")
        for pid in t.get("plugins", []) or []:
            if pid not in plugin_ids:
                results.append(CatalogResult(
                    "catalog.taskkinds.refs", BLOCK,
                    f"task-kind {tid} 引用了不存在的插件：{pid}", "catalog/task-kinds.yaml"))
        for sname in t.get("skills", []) or []:
            if sname not in skill_names:
                results.append(CatalogResult(
                    "catalog.taskkinds.refs", BLOCK,
                    f"task-kind {tid} 引用了不存在的 skill：{sname}", "catalog/task-kinds.yaml"))

    # ------------------------------------------------------ SKILL.md targets-repo
    norm_to_rid: dict = {}
    for r in cat.repos:
        rid = r.get("id")
        if not rid:
            continue
        for c in [rid] + [a for a in r.get("aliases", []) if isinstance(a, str)]:
            norm_to_rid.setdefault(_norm(c), rid)
    for rel, fm in sorted(cat.skills.items()):
        tr = fm.get("targets-repo")
        if tr is None:
            continue
        if not isinstance(tr, list):
            results.append(CatalogResult("catalog.skill.targets", BLOCK,
                                         f"targets-repo 必须是数组：{tr!r}", rel))
            continue
        for item in tr:
            if not isinstance(item, str) or _norm(item) not in norm_to_rid:
                results.append(CatalogResult(
                    "catalog.skill.targets", BLOCK,
                    f"targets-repo 无法解析到任一 repo（id/aliases）：{item!r}", rel))

    # --------------------------------------------------------- intent golden set
    for it in cat.intents:
        text = it.get("text") or ""
        exp_repos = [x for x in (it.get("expected_repos") or []) if isinstance(x, str)]
        resolved = cat.resolve_repos(text)
        if not exp_repos:
            results.append(CatalogResult("catalog.intents.repo", BLOCK,
                                         f"金句缺少 expected_repos：{text!r}",
                                         "catalog/intent-examples.yaml"))
            continue
        missing = [rid for rid in exp_repos if rid not in resolved]
        if missing:
            results.append(CatalogResult(
                "catalog.intents.repo", BLOCK,
                f"金句未消歧到期望仓 {missing}（实际命中 {resolved}）：{text!r}",
                "catalog/intent-examples.yaml"))
        # 误中无关仓：命中集合应 ⊆ 期望集合（歧义话术允许，见 expected_task: null）
        exp_task = it.get("expected_task")
        extra = [rid for rid in resolved if rid not in exp_repos]
        if extra and exp_task is not None:
            results.append(CatalogResult(
                "catalog.intents.repo", BLOCK,
                f"金句误中无关仓 {extra}（期望 {exp_repos}）：{text!r}",
                "catalog/intent-examples.yaml"))
        # 任务维度命中
        if exp_task is not None:
            tasks = cat.resolve_tasks(text)
            if exp_task not in tasks:
                results.append(CatalogResult(
                    "catalog.intents.task", WARN,
                    f"金句任务触发词未命中 {exp_task!r}（实际 {tasks}）：{text!r}",
                    "catalog/intent-examples.yaml"))
        # R3 插件投影（repo 命中话术才断言；纯模型名话术属语义候选 → 不判）
        exp_plugins = [x for x in (it.get("expected_plugins") or []) if isinstance(x, str)]
        resolved = cat.resolve_repos(text)
        if exp_plugins and resolved:
            plugins = cat.resolve_plugins(text)
            if not exp_plugins[0] in plugins:
                results.append(CatalogResult(
                    "catalog.intents.plugin", WARN,
                    f"金句 R3 插件投影未命中期望插件 {exp_plugins[0]!r}"
                    f"（实际 {plugins}）：{text!r}",
                    "catalog/intent-examples.yaml"))

    return results


def summarize(results: list) -> tuple:
    blocks = [r for r in results if r.level == BLOCK]
    warns = [r for r in results if r.level == WARN]
    return blocks, warns
