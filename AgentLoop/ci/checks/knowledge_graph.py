# CI Guard - catalog 知识图谱检查（M3 + v2 安装一致性）
# ======================================================
# 复用 catalog/checks_core 的确定性检查（同一份实现），转成 CheckResult
# 挂入 ci/guard.py 的 run_all_checks。
#
# 检查项（check id）：
#   catalog.repos.id-unique / kind / alias-conflict / repo-unaffined
#   catalog.plugins.affinity / skills / install（元数据 + core 唯一性）
#   catalog.taskkinds.refs
#   catalog.skill.targets（SKILL.md targets-repo 引用合法性）
#   catalog.intents.repo / task（金句断言）
#   catalog.marketplace.mismatch（marketplace 插件集合 == catalog install units）

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ci.common import BLOCK, WARN, CheckResult  # noqa: E402
from catalog.checks_core import run_checks  # noqa: E402
from catalog.loader import load_yaml  # noqa: E402


def _check_marketplace_units(repo: Path) -> list:
    """marketplace.json 插件集合必须与 catalog/plugins.yaml 的 install units 一致。

    防止新增/改名 unit 时漏登记 marketplace（会导致按需安装命令失效）。
    """
    results: list = []
    catalog_path = repo / "catalog" / "plugins.yaml"
    market_path = repo / ".claude-plugin" / "marketplace.json"
    if not catalog_path.exists() or not market_path.exists():
        return results
    try:
        units = {p.get("id") for p in
                 load_yaml(catalog_path).get("plugins", []) if p.get("id")}
        mp = json.loads(market_path.read_text(encoding="utf-8"))
        entries = {e.get("name") for e in mp.get("plugins", []) if isinstance(e, dict)}
    except Exception as e:
        results.append(CheckResult("catalog.marketplace.mismatch", BLOCK,
                                   f"marketplace/catalog 解析失败: {e}", str(market_path)))
        return results
    missing_in_mp = sorted(units - entries)
    extra_in_mp = sorted(entries - units)
    if missing_in_mp:
        results.append(CheckResult(
            "catalog.marketplace.mismatch", BLOCK,
            f"catalog 中的 unit 未登记到 marketplace.json：{missing_in_mp}", str(market_path)))
    if extra_in_mp:
        results.append(CheckResult(
            "catalog.marketplace.mismatch", BLOCK,
            f"marketplace.json 中存在 catalog 未声明的插件：{extra_in_mp}", str(market_path)))
    return results


def check_knowledge_graph(repo: Path) -> list:
    """Run catalog checks; map CatalogResult to CheckResult.

    Gracefully no-op when the repo has no catalog/ directory (e.g. minimal
    fixture repos in tests) so the check never crashes the guard.
    """
    results: list = []
    if not (repo / "catalog").is_dir():
        return results
    for r in run_checks(repo):
        level = BLOCK if r.level == BLOCK else WARN
        results.append(CheckResult(r.check, level, r.message, r.path))
    results += _check_marketplace_units(repo)
    return results
