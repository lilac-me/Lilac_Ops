# CI Guard 自测 — catalog 知识图谱检查
# ======================================
# 覆盖 ci/checks/knowledge_graph.py（M3 门禁）：
#   - 无 catalog/ 目录的最小仓库 → no-op（不崩溃）
#   - 合法 catalog → 0 阻塞
#   - 别名跨仓冲突 / targets-repo 引用非法 / 金句消歧失败 → 阻塞

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ci import guard  # noqa: E402
from ci.common import BLOCK, discover_plugins, discover_skills  # noqa: E402
from ci.checks import knowledge_graph as kg_check  # noqa: E402

GOOD_REPOS = """\
repos:
  - id: mindspeed-llm
    name: MindSpeed-LLM
    aliases: ["ModelLink", "mindspeed-llm"]
    kind: llm-training
    backends: [mcore]
    urls: {}
    related: []
  - id: vllm-ascend
    name: vLLM-Ascend
    aliases: ["vllm-ascend"]
    kind: inference
    backends: [vllm]
    urls: {}
    related: []
"""

GOOD_PLUGINS = """\
plugins:
  - id: mindspeed-agent
    name: mindspeed-agent
    platforms: [claude, codex, dsh]
    repo_affinity: [mindspeed-llm]
    skills: []
    workflow: ""
    install:
      unit: core
      default_installed: true
      skills_root: skills
      install_hint:
        claude: "claude plugin install mindspeed-agent@X"
        codex: "codex plugin add mindspeed-agent@X"
        dsh: "python tools/install-unit.py dsh mindspeed-agent"
"""

# 一个额外 on-demand 插件（用于多插件/非 core 场景测试）
ONDEMAND_PLUGIN = """\
plugins:
  - id: mindspeed-agent
    name: mindspeed-agent
    platforms: [claude, codex, dsh]
    repo_affinity: [mindspeed-llm]
    skills: []
    workflow: ""
    install:
      unit: core
      default_installed: true
      skills_root: skills
      install_hint:
        claude: "claude plugin install mindspeed-agent@X"
        codex: "codex plugin add mindspeed-agent@X"
        dsh: "python tools/install-unit.py dsh mindspeed-agent"
  - id: demo-llm
    name: demo-llm
    platforms: [claude, codex, dsh]
    repo_affinity: [mindspeed-llm]
    skills: [demo-llm-skill]
    workflow: ""
    install:
      unit: on-demand
      default_installed: false
      skills_root: skills
      install_hint:
        claude: "claude plugin install demo-llm@X"
        codex: "codex plugin add demo-llm@X"
        dsh: "python tools/install-unit.py dsh demo-llm"
"""

GOOD_TASKS = """\
task-kinds:
  - id: perf-optimize
    trigger_words: [优化, 性能]
    plugins: [mindspeed-agent]
    skills: []
"""

GOOD_INTENTS = """\
intents:
  - text: "我想优化 mindspeed-llm 的模型"
    expected_repos: [mindspeed-llm]
    expected_task: null
    expected_plugins: [mindspeed-agent]
    note: "验收金句"
"""

GOOD_SKILL = """\
---
name: demo-llm-skill
description: This skill should be used when the user asks to demo.
targets-repo:
  - mindspeed-llm
---

# Demo

正文。
"""


def _write(root: Path, name: str, content: str) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _setup_catalog(root: Path, repos=GOOD_REPOS, plugins=GOOD_PLUGINS,
                   tasks=GOOD_TASKS, intents=GOOD_INTENTS) -> None:
    cdir = root / "catalog"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "repos.yaml").write_text(textwrap.dedent(repos), encoding="utf-8")
    (cdir / "plugins.yaml").write_text(textwrap.dedent(plugins), encoding="utf-8")
    (cdir / "task-kinds.yaml").write_text(textwrap.dedent(tasks), encoding="utf-8")
    (cdir / "intent-examples.yaml").write_text(textwrap.dedent(intents), encoding="utf-8")


def _levels(results, level):
    return [r for r in results if r.level == level]


class TestKnowledgeGraphCheck:
    def test_no_catalog_noop(self, tmp_path):
        assert kg_check.check_knowledge_graph(tmp_path) == []

    def test_clean_catalog_passes(self, tmp_path):
        _setup_catalog(tmp_path)
        _write(tmp_path, "skills/demo-skills/common/demo-llm-skill/SKILL.md", GOOD_SKILL)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert _levels(results, BLOCK) == []

    def test_alias_conflict_blocks(self, tmp_path):
        repos = GOOD_REPOS.replace('aliases: ["ModelLink", "mindspeed-llm"]',
                                   'aliases: ["vllm-ascend"]')  # collides with other id/alias
        _setup_catalog(tmp_path, repos=repos)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.repos.alias-conflict" for r in _levels(results, BLOCK))

    def test_bad_targets_repo_blocks(self, tmp_path):
        _setup_catalog(tmp_path)
        bad_skill = GOOD_SKILL.replace("mindspeed-llm", "ghost-repo")
        _write(tmp_path, "skills/demo-skills/common/demo-llm-skill/SKILL.md", bad_skill)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.skill.targets" for r in _levels(results, BLOCK))

    def test_intent_miss_blocks(self, tmp_path):
        _setup_catalog(tmp_path, intents=GOOD_INTENTS.replace("mindspeed-llm", "mindspeed-mm"))
        _write(tmp_path, "skills/demo-skills/common/demo-llm-skill/SKILL.md", GOOD_SKILL)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.intents.repo" for r in _levels(results, BLOCK))

    def test_plugin_affinity_missing_repo_blocks(self, tmp_path):
        _setup_catalog(tmp_path, plugins=GOOD_PLUGINS.replace("mindspeed-llm", "ghost-repo"))
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.plugins.affinity" for r in _levels(results, BLOCK))

    # ---------- install 元数据（core/on-demand 按需安装）----------
    def test_core_plus_ondemand_passes(self, tmp_path):
        _setup_catalog(tmp_path, plugins=ONDEMAND_PLUGIN)
        _write(tmp_path, "skills/demo-skills/common/demo-llm-skill/SKILL.md", GOOD_SKILL)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert _levels(results, BLOCK) == []

    def test_two_core_plugins_block(self, tmp_path):
        plugins = ONDEMAND_PLUGIN.replace("unit: on-demand", "unit: core")  # 两个 core
        _setup_catalog(tmp_path, plugins=plugins)
        _write(tmp_path, "skills/demo-skills/common/demo-llm-skill/SKILL.md", GOOD_SKILL)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.plugins.install" for r in _levels(results, BLOCK))

    def test_missing_install_metadata_blocks(self, tmp_path):
        plugins = GOOD_PLUGINS.split("    install:")[0]  # 砍掉 install 块
        _setup_catalog(tmp_path, plugins=plugins)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.plugins.install" for r in _levels(results, BLOCK))

    def test_bad_skills_root_blocks(self, tmp_path):
        plugins = GOOD_PLUGINS.replace("skills_root: skills", "skills_root: skills/nope")
        _setup_catalog(tmp_path, plugins=plugins)
        results = kg_check.check_knowledge_graph(tmp_path)
        assert any(r.check == "catalog.plugins.install" for r in _levels(results, BLOCK))


class TestGuardWithCatalog:
    def _setup_minimal(self, root: Path) -> None:
        _write(root, "skills/demo-skills/common/demo-llm-skill/SKILL.md", GOOD_SKILL)
        _setup_catalog(root)
        mp = root / ".claude-plugin"
        mp.mkdir(parents=True)
        (mp / "plugin.json").write_text(
            '{"name": "mindspeed-agent", "version": "1.0.0", "skills": ["./skills"]}', encoding="utf-8")
        (mp / "marketplace.json").write_text(
            '{"name": "test", "repository": "x", "plugins": [{"name": "mindspeed-agent", "version": "1.0.0", "source": "."}]}',
            encoding="utf-8")

    def test_guard_exit_0_with_catalog(self, tmp_path):
        self._setup_minimal(tmp_path)
        assert guard.main(["--repo", str(tmp_path)]) == 0

    def test_guard_exit_1_on_catalog_block(self, tmp_path):
        self._setup_minimal(tmp_path)
        # break catalog alias uniqueness -> guard should block
        repos = GOOD_REPOS.replace('aliases: ["ModelLink", "mindspeed-llm"]',
                                   'aliases: ["vllm-ascend"]')
        (tmp_path / "catalog" / "repos.yaml").write_text(textwrap.dedent(repos), encoding="utf-8")
        assert guard.main(["--repo", str(tmp_path)]) == 1

    def test_marketplace_missing_unit_blocks(self, tmp_path):
        self._setup_minimal(tmp_path)
        # catalog 声明了 on-demand unit，但 marketplace 未登记 → 一致性 BLOCK
        (tmp_path / "catalog" / "plugins.yaml").write_text(
            textwrap.dedent(ONDEMAND_PLUGIN), encoding="utf-8")
        assert guard.main(["--repo", str(tmp_path)]) == 1
