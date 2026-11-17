#!/usr/bin/env python3
# MindSpeed-Agent CI Guard
# ========================
# 看护仓库中 plugins / skills 的安装有效性：
#   L1 静态结构（命名、frontmatter、引用、plugin.json、agent）
#   L2 一致性（marketplace 版本、状态机 YAML、YAML↔编译产物、JS 语法）
#   L3 分类体系一致性（警告）
#
# 用法：
#   python ci/guard.py --repo <仓库路径> [--json report.json]
#
# 退出码：0 = 通过（可含警告）；1 = 存在阻塞问题；2 = 脚本运行错误。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根，供 from ci... import

from ci.checks import marketplace as _marketplace  # noqa: E402
from ci.checks import plugins as _plugins  # noqa: E402
from ci.checks import skills as _skills  # noqa: E402
from ci.checks import workflows as _workflows  # noqa: E402
from ci.checks import knowledge_graph as _knowledge_graph  # noqa: E402
from ci.common import BLOCK, WARN, CheckResult, discover_plugins, discover_skills  # noqa: E402


def run_all_checks(repo: Path) -> list:
    """执行全部检查，返回 CheckResult 列表。"""
    results: list = []
    skills = discover_skills(repo)
    results += _skills.check_skills(repo, skills)
    plugins = discover_plugins(repo)
    results += _plugins.check_plugins(repo, plugins, skills)
    results += _marketplace.check_marketplace(repo, plugins)
    results += _workflows.check_workflows(repo)
    results += _knowledge_graph.check_knowledge_graph(repo)
    return results


def print_report(repo: Path, results: list) -> None:
    blocks = [r for r in results if r.level == BLOCK]
    warns = [r for r in results if r.level == WARN]

    print(f"=== MindSpeed-Agent CI Guard: {repo} ===")
    print(f"skill 总数: {len(discover_skills(repo))} | 插件目录: {len(discover_plugins(repo))}")
    print(f"结果: {len(blocks)} 阻塞 / {len(warns)} 警告\n")

    if blocks:
        print("--- 阻塞问题 ---")
        for r in sorted(blocks, key=lambda x: (x.check, x.path)):
            where = f" [{r.path}]" if r.path else ""
            print(f"  ✗ {r.check}{where}: {r.message}")
        print()
    if warns:
        print("--- 警告 ---")
        for r in sorted(warns, key=lambda x: (x.check, x.path)):
            where = f" [{r.path}]" if r.path else ""
            print(f"  ! {r.check}{where}: {r.message}")
        print()

    if blocks:
        print(f"结论: 未通过（{len(blocks)} 个阻塞问题）")
    else:
        print(f"结论: 通过（{len(warns)} 个警告，不阻塞）")


def write_json_report(repo: Path, results: list, out: Path) -> None:
    blocks = [r for r in results if r.level == BLOCK]
    warns = [r for r in results if r.level == WARN]
    report = {
        "repo": str(repo),
        "summary": {"skills": len(discover_skills(repo)), "plugins": len(discover_plugins(repo)),
                    "block": len(blocks), "warn": len(warns)},
        "results": [r.to_dict() for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MindSpeed-Agent CI Guard — plugins/skills 安装有效性看护")
    ap.add_argument("--repo", default=".", help="仓库根路径（默认当前目录）")
    ap.add_argument("--json", default=None, help="JSON 报告输出路径（供 CI artifact）")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"错误: 仓库路径不存在: {repo}", file=sys.stderr)
        return 2

    try:
        results = run_all_checks(repo)
        print_report(repo, results)
        if args.json:
            write_json_report(repo, results, Path(args.json))
    except Exception as e:  # 脚本自身错误 → 退出码 2
        import traceback
        traceback.print_exc()
        print(f"guard 运行错误: {e}", file=sys.stderr)
        return 2

    blocks = [r for r in results if r.level == BLOCK]
    return 1 if blocks else 0


if __name__ == "__main__":
    sys.exit(main())
