#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""catalog/generator.py — 汇总 repos/plugins/task-kinds + SKILL.md frontmatter，
输出一份跨平台统一索引 catalog/graph.json，并做一致性告警。

用法：
    python catalog/generator.py [--repo <仓库根>] [--out <输出 json>]

产物结构（nodes/edges 均为数组，方便任意平台/渲染器消费）：
    nodes.repos      Repo 节点（来自 repos.yaml）
    nodes.plugins    Plugin 节点（来自 plugins.yaml，含 execution/workflow/entry_agent/agents）
    nodes.task_kinds TaskKind 节点（来自 task-kinds.yaml）
    nodes.skills     由 SKILL.md frontmatter 扫描（含 domain/function/tags/targets-repo）
    edges.skill_targets_repo  skill name -> repo id（由 frontmatter targets-repo 解析）
    edges.plugin_affinity     plugin id -> repo id（由 plugins.yaml repo_affinity）
    edges.plugin_skills       plugin id -> skill name（plugins.yaml skills）
    edges.task_skills         task kind id -> skill name（task-kinds.yaml skills）
    edges.plugin_workflows    plugin id -> workflow 文件路径（plugins.yaml workflow，非空时）
    edges.plugin_agents       plugin id -> entry_agent + agents 列表（state-machine 插件）
    routing.hits              对 intent-examples 的确定性消歧结果（供回归，
                              resolved_plugins 为 R3 插件投影）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog.loader import Catalog  # noqa: E402


def normalize_ref(ref: str) -> str:
    return Catalog.normalize(ref)


def build(repo_root: Path) -> dict:
    cat = Catalog(repo_root)
    norm_to_rid: dict = {}
    for r in cat.repos:
        rid = r.get("id")
        if not rid:
            continue
        for c in [rid] + [a for a in r.get("aliases", []) if isinstance(a, str)]:
            norm_to_rid.setdefault(Catalog.normalize(c), rid)

    # skill nodes（仅收录有 name 的；正文语义不改动）
    skill_nodes = []
    for rel, fm in sorted(cat.skills.items()):
        name = fm.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        targets = fm.get("targets-repo")
        resolved_targets = []
        if isinstance(targets, list):
            for t in targets:
                rid = norm_to_rid.get(normalize_ref(str(t))) if isinstance(t, str) else None
                if rid:
                    resolved_targets.append(rid)
        skill_nodes.append({
            "name": name,
            "path": rel,
            "domain": fm.get("domain"),
            "function": fm.get("function"),
            "tags": fm.get("tags") or [],
            "targets-repo": targets or [],
            "targets-repo-resolved": sorted(set(resolved_targets)),
        })

    # edges
    skill_targets = []
    for s in skill_nodes:
        for rid in s["targets-repo-resolved"]:
            skill_targets.append({"skill": s["name"], "repo": rid})
    plugin_affinity = []
    for p in cat.plugins:
        pid = p.get("id")
        for rid in p.get("repo_affinity", []) or []:
            plugin_affinity.append({"plugin": pid, "repo": rid})
    plugin_skills = []
    for p in cat.plugins:
        pid = p.get("id")
        for sname in p.get("skills", []) or []:
            plugin_skills.append({"plugin": pid, "skill": sname})
    task_skills = []
    for t in cat.task_kinds:
        tid = t.get("id")
        for sname in t.get("skills", []) or []:
            task_skills.append({"task": tid, "skill": sname})

    # state-machine 插件：workflow 文件 + entry/agents（供路由终点判定）
    plugin_workflows = []
    plugin_agents = []
    for p in cat.plugins:
        pid = p.get("id")
        wf = p.get("workflow")
        if isinstance(wf, str) and wf.strip():
            plugin_workflows.append({"plugin": pid, "workflow": wf.strip()})
        if (p.get("execution") or "skills") == "state-machine":
            agents = [a for a in p.get("agents") or [] if isinstance(a, str)]
            entry = p.get("entry_agent")
            if entry and entry not in agents:
                agents = [entry] + agents
            for ag in agents:
                plugin_agents.append({"plugin": pid, "agent": ag})

    # routing hits（R1 repo 消歧 + R2 任务命中 + R3 插件投影），供回归与人工检视
    hits = []
    for it in cat.intents:
        text = it.get("text") or ""
        hits.append({
            "text": text,
            "expected_repos": it.get("expected_repos") or [],
            "resolved_repos": cat.resolve_repos(text),
            "expected_task": it.get("expected_task"),
            "resolved_tasks": cat.resolve_tasks(text),
            "expected_plugins": it.get("expected_plugins") or [],
            "resolved_plugins": cat.resolve_plugins(text),
        })

    return {
        "generator": "catalog/generator.py",
        "schema": "v2",
        "nodes": {
            "repos": cat.repos,
            "plugins": cat.plugins,
            "task_kinds": cat.task_kinds,
            "skills": skill_nodes,
        },
        "edges": {
            "skill_targets_repo": skill_targets,
            "plugin_affinity": plugin_affinity,
            "plugin_skills": plugin_skills,
            "task_skills": task_skills,
            "plugin_workflows": plugin_workflows,
            "plugin_agents": plugin_agents,
        },
        "routing": {"hits": hits},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 catalog/graph.json 统一索引")
    ap.add_argument("--repo", default=".", help="仓库根路径（默认当前目录）")
    ap.add_argument("--out", default=None, help="输出 json 路径（默认 catalog/graph.json）")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    out = Path(args.out) if args.out else repo / "catalog" / "graph.json"
    try:
        graph = build(repo)
    except Exception as e:  # 脚本错误 → 2
        import traceback
        traceback.print_exc()
        print(f"generator 运行错误: {e}", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    n_skills = len(graph["nodes"]["skills"])
    print(f"written: {out}")
    print(f"repos={len(graph['nodes']['repos'])} plugins={len(graph['nodes']['plugins'])} "
          f"task_kinds={len(graph['nodes']['task_kinds'])} skills={n_skills}")
    print(f"edges: skill_targets_repo={len(graph['edges']['skill_targets_repo'])} "
          f"plugin_affinity={len(graph['edges']['plugin_affinity'])} "
          f"plugin_skills={len(graph['edges']['plugin_skills'])} "
          f"task_skills={len(graph['edges']['task_skills'])} "
          f"plugin_workflows={len(graph['edges']['plugin_workflows'])} "
          f"plugin_agents={len(graph['edges']['plugin_agents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
