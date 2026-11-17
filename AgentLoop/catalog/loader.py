# -*- coding: utf-8 -*-
"""catalog loader: parse catalog/*.yaml + skills/**/SKILL.md frontmatter.

Deterministic disambiguation helpers used by lint_catalog.py, generator.py and
the CI check (ci/checks/knowledge_graph.py).

R1 repo disambiguation: normalize the utterance (lowercase, strip spaces,
hyphens, underscores) then match each repo candidate (id + aliases,
normalized) with longest-match-first while skipping spans already covered by
a longer candidate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CATALOG_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    """Load a YAML file; minimal fallback raises on failure."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load catalog YAML files")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


class Catalog:
    """Aggregated catalog state: repos / plugins / task kinds / intent examples."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        cdir = self.repo_root / "catalog"
        self.repos = load_yaml(cdir / "repos.yaml").get("repos", [])
        self.plugins = load_yaml(cdir / "plugins.yaml").get("plugins", [])
        self.task_kinds = load_yaml(cdir / "task-kinds.yaml").get("task-kinds", [])
        self.intents = load_yaml(cdir / "intent-examples.yaml").get("intents", [])
        self.skills = self._discover_skills()

    # ------------------------------------------------------------------ skills
    def _discover_skills(self) -> dict:
        """Return {skill_rel_posix_path: frontmatter} for skills/**/SKILL.md."""
        out: dict = {}
        root = self.repo_root / "skills"
        if not root.is_dir():
            return out
        for sk in sorted(root.rglob("SKILL.md")):
            fm = self._parse_frontmatter(sk)
            if fm:
                out[sk.parent.relative_to(self.repo_root).as_posix()] = fm
        return out

    @staticmethod
    def _parse_frontmatter(path: Path) -> dict:
        text = path.read_text(encoding="utf-8", errors="replace")
        m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
        if not m:
            return {}
        raw = m.group(1)
        try:
            data = yaml.safe_load(raw) if yaml is not None else None
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------- repos
    def repo_by_id(self, rid: str) -> Optional[dict]:
        return next((r for r in self.repos if r.get("id") == rid), None)

    def all_repo_tokens(self) -> list:
        """[(normalized_token, repo_id)] for id + aliases of every repo."""
        toks: list = []
        for r in self.repos:
            rid = r.get("id")
            if not rid:
                continue
            cands = [rid] + [a for a in r.get("aliases", []) if isinstance(a, str)]
            for c in cands:
                toks.append((Catalog.normalize(c), rid))
        # longest first; stable by repo id for equal length
        toks.sort(key=lambda t: (-len(t[0]), t[1]))
        return toks

    @staticmethod
    def normalize(s: str) -> str:
        return re.sub(r"[\s_\-/.]+", "", str(s).lower())

    def resolve_repos(self, utterance: str) -> list:
        """R1: return repo ids mentioned in the utterance (longest-match)."""
        text = Catalog.normalize(utterance)
        if not text:
            return []
        spans: list = []  # (start, end, repo_id)
        for tok, rid in self.all_repo_tokens():
            if not tok:
                continue
            for m in re.finditer(re.escape(tok), text):
                s, e = m.start(), m.end()
                if any(not (e <= os_ or s >= oe_) for os_, oe_, _ in spans):
                    # already covered by a longer match
                    continue
                spans.append((s, e, rid))
        # left-to-right by start, dedupe repo id
        result: list = []
        seen: set = set()
        for s, _, rid in sorted(spans, key=lambda t: t[0]):
            if rid not in seen:
                seen.add(rid)
                result.append(rid)
        return result

    # -------------------------------------------------------------- task kinds
    def task_by_id(self, tid: str) -> Optional[dict]:
        return next((t for t in self.task_kinds if t.get("id") == tid), None)

    def resolve_tasks(self, utterance: str) -> list:
        """R2: return task-kind ids whose trigger words appear in the utterance."""
        text = Catalog.normalize(utterance)
        hits: list = []
        for t in self.task_kinds:
            words = [Catalog.normalize(w) for w in t.get("trigger_words", [])]
            if any(w and w in text for w in words):
                hits.append(t.get("id"))
        return hits

    # ------------------------------------------------------------------ plugins
    def plugin_by_id(self, pid: str) -> Optional[dict]:
        return next((p for p in self.plugins if p.get("id") == pid), None)

    def resolve_plugins(self, utterance: str) -> list:
        """R3: return plugin ids servicing the utterance.

        Plugin candidate = repo_affinity ∩ resolved repo(s) ∩ task-kind
        `plugins` 列表（R3 链路组装的确定性投影）。repo 无命中（纯模型名话术）
        时按任务候选插件 + 任务触发词给出（仍属语义候选，须经 R4 确认）。

        规则：
        - repo 命中时：优先取 repo_affinity 与 resolved repo 相交的插件；
          无 repo-affined 候选时，兜底取无 repo_affinity 的插件（repo 无关
          插件，如 doc-agent 面向任意仓）。
        - 命中多个候选且存在 on-demand 时，优先 on-demand（core
          mindspeed-agent 是全仓路由底座，声明了对所有仓的亲缘，会吞掉领域
          插件；如 vllm-ascend+perf-optimize → inference-model-optimize）。
        """
        repos = set(self.resolve_repos(utterance))
        tasks = set(self.resolve_tasks(utterance))
        task_plugins = set()
        for t in self.task_kinds:
            if t.get("id") in tasks:
                task_plugins.update(t.get("plugins") or [])

        def _is_core(p: dict) -> bool:
            return (p.get("install") or {}).get("unit") == "core"

        affined, agnostic = [], []
        for p in self.plugins:
            pid = p.get("id")
            if not pid or pid not in task_plugins:
                continue
            aff = set(p.get("repo_affinity") or [])
            if repos:
                if aff & repos:
                    affined.append(p)
                elif not aff:
                    agnostic.append(p)
            elif not _is_core(p):
                # 无 repo 命中：on-demand 插件可作语义候选（无 repo_affinity 限制）
                agnostic.append(p)

        pool = affined or agnostic
        on_demand = [p for p in pool if not _is_core(p)]
        chosen = on_demand or pool
        return sorted({p.get("id") for p in chosen})

    # ------------------------------------------------------------------- skills
    def skills_for_repo(self, rid: str) -> list:
        """Skills declaring targets-repo == rid (or matching its alias)."""
        out = []
        for rel, fm in self.skills.items():
            tr = fm.get("targets-repo")
            if not isinstance(tr, list):
                continue
            norm = [Catalog.normalize(x) for x in tr if isinstance(x, str)]
            if Catalog.normalize(rid) in norm:
                out.append(rel)
        return out
