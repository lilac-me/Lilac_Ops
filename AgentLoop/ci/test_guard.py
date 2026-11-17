# CI Guard 自测
# ============
# 用临时目录构造合法/非法样例，验证各检查的判定与 guard 入口的退出码，
# 防止看护脚本自身腐化。

import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ci import guard  # noqa: E402
from ci.common import (  # noqa: E402
    BLOCK,
    WARN,
    discover_plugins,
    discover_skills,
    is_kebab,
    parse_frontmatter,
)
from ci.checks import plugins as plugins_check  # noqa: E402
from ci.checks import skills as skills_check  # noqa: E402
from ci.checks import marketplace as marketplace_check  # noqa: E402

GOOD_SKILL = textwrap.dedent("""\
    ---
    name: demo-skill
    description: This skill should be used when the user asks to demo.
    version: "1.0.0"
    tags: [demo, ci]
    ---

    # Demo Skill

    正文内容。
    """)


def _write_skill(root: Path, rel: str, content: str = GOOD_SKILL, name: str = "demo-skill") -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def _write_plugin(root: Path, name: str, version: str = "1.0.0", skills: list = None,
                  manifest_ok: bool = True) -> Path:
    pdir = root / "plugins" / name
    mp = pdir / ".claude-plugin"
    mp.mkdir(parents=True, exist_ok=True)
    if manifest_ok:
        (mp / "plugin.json").write_text(
            json.dumps({"name": name, "version": version, "skills": skills or []}), encoding="utf-8")
    return pdir


def _levels(results, level):
    return [r for r in results if r.level == level]


# ---------------------------------------------------------------------------
# common 工具
# ---------------------------------------------------------------------------

class TestCommon:
    def test_is_kebab(self):
        assert is_kebab("demo-skill")
        assert not is_kebab("Demo-Skill")     # 大写
        assert not is_kebab("-demo-skill")    # 前导连字符
        assert not is_kebab("demo--skill")    # 连续连字符
        assert not is_kebab("ab")             # 过短
        assert not is_kebab("a" * 41)         # 过长

    def test_parse_frontmatter(self):
        fm = parse_frontmatter(GOOD_SKILL)
        assert fm["name"] == "demo-skill"
        assert fm["description"].startswith("This skill should be used when")
        assert fm["tags"] == ["demo", "ci"]


# ---------------------------------------------------------------------------
# L1 skill 检查
# ---------------------------------------------------------------------------

class TestSkillChecks:
    def test_good_skill_passes(self, tmp_path):
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill")
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert _levels(results, BLOCK) == []

    def test_bad_dirname_blocks(self, tmp_path):
        _write_skill(tmp_path, "skills/demo-skills/common/Demo-Skill")
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("kebab-case" in r.message for r in _levels(results, BLOCK))

    def test_missing_description_blocks(self, tmp_path):
        content = GOOD_SKILL.replace('description: This skill should be used when the user asks to demo.\n', "")
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill", content)
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("description" in r.message for r in _levels(results, BLOCK))

    def test_name_mismatch_blocks(self, tmp_path):
        content = GOOD_SKILL.replace("name: demo-skill", "name: other-skill")
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill", content)
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("不一致" in r.message for r in _levels(results, BLOCK))

    def test_empty_body_blocks(self, tmp_path):
        content = GOOD_SKILL.split("---\n\n")[0] + "---\n\n   \n"
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill", content)
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("正文为空" in r.message for r in _levels(results, BLOCK))

    def test_duplicate_name_blocks(self, tmp_path):
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill")
        _write_skill(tmp_path, "skills/demo-skills/deployment/demo-skill")
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("重复" in r.message for r in _levels(results, BLOCK))

    def test_chinese_description_warns_only(self, tmp_path):
        content = GOOD_SKILL.replace(
            'description: This skill should be used when the user asks to demo.',
            "description: 面向中文场景的演示技能。")
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill", content)
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert _levels(results, BLOCK) == []
        assert any("skill.description-prefix" == r.check for r in _levels(results, WARN))

    def test_missing_referenced_file_warns(self, tmp_path):
        content = GOOD_SKILL.replace("正文内容。", "参见 references/nonexistent.md。")
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill", content)
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert any("引用" in r.message for r in _levels(results, WARN))

    def test_path_depth_anomaly_warns(self, tmp_path):
        _write_skill(tmp_path, "skills/demo-skills/demo-skill")  # 2 层
        results = skills_check.check_skills(tmp_path, discover_skills(tmp_path))
        assert _levels(results, BLOCK) == []
        assert any("深度" in r.message for r in _levels(results, WARN))


# ---------------------------------------------------------------------------
# L1 plugin 检查
# ---------------------------------------------------------------------------

class TestPluginChecks:
    def test_dangling_skill_path_blocks(self, tmp_path):
        _write_plugin(tmp_path, "demo-plugin", skills=["./skills/not-exist"])
        results = plugins_check.check_plugins(tmp_path, discover_plugins(tmp_path), discover_skills(tmp_path))
        assert any("悬空" in r.message for r in _levels(results, BLOCK))

    def test_missing_symlink_warns_when_basename_exists(self, tmp_path):
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill")
        _write_plugin(tmp_path, "demo-plugin", skills=["./skills/demo-skill"])  # symlink 未建
        results = plugins_check.check_plugins(tmp_path, discover_plugins(tmp_path), discover_skills(tmp_path))
        assert _levels(results, BLOCK) == []
        assert any("symlink" in r.message for r in _levels(results, WARN))

    def test_agent_unknown_skill_blocks(self, tmp_path):
        pdir = _write_plugin(tmp_path, "demo-plugin", skills=["./skills/demo-skill"])
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill")
        adir = pdir / "agents" / "demo-agent"
        adir.mkdir(parents=True)
        (adir / "AGENT.md").write_text(
            "---\nname: demo-agent\ndescription: demo\nskills:\n  - ghost-skill\n---\n\n正文。", encoding="utf-8")
        results = plugins_check.check_plugins(tmp_path, discover_plugins(tmp_path), discover_skills(tmp_path))
        assert any("ghost-skill" in r.message for r in _levels(results, BLOCK))

    def test_agent_known_skill_passes(self, tmp_path):
        pdir = _write_plugin(tmp_path, "demo-plugin", skills=["./skills/demo-skill"])
        _write_skill(tmp_path, "skills/demo-skills/common/demo-skill")
        adir = pdir / "agents" / "demo-agent"
        adir.mkdir(parents=True)
        (adir / "AGENT.md").write_text(
            "---\nname: demo-agent\ndescription: demo\nskills:\n  - demo-skill\n---\n\n正文。", encoding="utf-8")
        results = plugins_check.check_plugins(tmp_path, discover_plugins(tmp_path), discover_skills(tmp_path))
        assert _levels(results, BLOCK) == []

    def test_manifest_missing_warns(self, tmp_path):
        (tmp_path / "plugins" / "no-manifest-plugin").mkdir(parents=True)
        results = plugins_check.check_plugins(tmp_path, discover_plugins(tmp_path), discover_skills(tmp_path))
        assert any("缺少 .claude-plugin" in r.message for r in _levels(results, WARN))


# ---------------------------------------------------------------------------
# L2 marketplace 检查
# ---------------------------------------------------------------------------

class TestMarketplaceChecks:
    def _write_marketplace(self, root: Path, entries: list) -> None:
        mp = root / ".claude-plugin"
        mp.mkdir(parents=True, exist_ok=True)
        (mp / "marketplace.json").write_text(
            json.dumps({"name": "test", "repository": "x", "plugins": entries}), encoding="utf-8")

    def test_version_mismatch_blocks(self, tmp_path):
        _write_plugin(tmp_path, "demo-plugin", version="2.0.0")
        self._write_marketplace(tmp_path, [{"name": "demo-plugin", "version": "1.0.0", "source": "./plugins/demo-plugin"}])
        results = marketplace_check.check_marketplace(tmp_path, discover_plugins(tmp_path))
        assert any("不一致" in r.message for r in _levels(results, BLOCK))

    def test_version_match_passes(self, tmp_path):
        _write_plugin(tmp_path, "demo-plugin", version="2.0.0")
        self._write_marketplace(tmp_path, [{"name": "demo-plugin", "version": "2.0.0", "source": "./plugins/demo-plugin"}])
        results = marketplace_check.check_marketplace(tmp_path, discover_plugins(tmp_path))
        assert _levels(results, BLOCK) == []

    def test_source_missing_blocks(self, tmp_path):
        self._write_marketplace(tmp_path, [{"name": "ghost", "version": "1.0.0", "source": "./plugins/ghost"}])
        results = marketplace_check.check_marketplace(tmp_path, discover_plugins(tmp_path))
        assert any("source 路径不存在" in r.message for r in _levels(results, BLOCK))

    def test_marketplace_missing_blocks(self, tmp_path):
        results = marketplace_check.check_marketplace(tmp_path, discover_plugins(tmp_path))
        assert any("缺少" in r.message for r in _levels(results, BLOCK))


# ---------------------------------------------------------------------------
# guard 入口
# ---------------------------------------------------------------------------

class TestGuardEntry:
    def _setup_minimal_repo(self, root: Path) -> None:
        """构造最小合法仓库：根 manifest + marketplace + 一个正常 skill。"""
        _write_skill(root, "skills/demo-skills/common/demo-skill")
        mp = root / ".claude-plugin"
        mp.mkdir(parents=True)
        (mp / "plugin.json").write_text(
            json.dumps({"name": "test-root", "version": "1.0.0", "skills": ["./skills"]}), encoding="utf-8")
        (mp / "marketplace.json").write_text(
            json.dumps({"name": "test", "repository": "x",
                        "plugins": [{"name": "test-root", "version": "1.0.0", "source": "."}]}),
            encoding="utf-8")

    def test_clean_repo_exit_0(self, tmp_path):
        self._setup_minimal_repo(tmp_path)
        rc = guard.main(["--repo", str(tmp_path)])
        assert rc == 0

    def test_broken_repo_exit_1(self, tmp_path):
        self._setup_minimal_repo(tmp_path)
        _write_skill(tmp_path, "skills/demo-skills/common/Bad-Name")
        rc = guard.main(["--repo", str(tmp_path)])
        assert rc == 1

    def test_json_report_written(self, tmp_path):
        self._setup_minimal_repo(tmp_path)
        out = tmp_path / "report.json"
        rc = guard.main(["--repo", str(tmp_path), "--json", str(out)])
        assert rc == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["summary"]["skills"] == 1
        assert "results" in report
