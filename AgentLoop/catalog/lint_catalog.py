#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""catalog 本地校验 CLI。

用法：
    python catalog/lint_catalog.py [--repo <仓库根>] [--json report.json]

退出码：0 = 通过（可含警告）；1 = 存在阻塞问题；2 = 脚本运行错误。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog.checks_core import BLOCK, run_checks, summarize  # noqa: E402


def print_report(results: list) -> None:
    blocks, warns = summarize(results)
    print(f"=== MindSpeed-Agent catalog lint ===")
    print(f"结果: {len(blocks)} 阻塞 / {len(warns)} 警告\n")
    if blocks:
        print("--- 阻塞问题 ---")
        for r in sorted(blocks, key=lambda x: (x.check, x.path)):
            where = f" [{r.path}]" if r.path else ""
            print(f"   ✗ {r.check}{where}: {r.message}")
        print()
    if warns:
        print("--- 警告 ---")
        for r in sorted(warns, key=lambda x: (x.check, x.path)):
            where = f" [{r.path}]" if r.path else ""
            print(f"   ! {r.check}{where}: {r.message}")
        print()
    if blocks:
        print(f"结论: 未通过（{len(blocks)} 个阻塞问题）")
    else:
        print(f"结论: 通过（{len(warns)} 个警告，不阻塞）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="catalog 知识图谱数据校验")
    ap.add_argument("--repo", default=".", help="仓库根路径（默认当前目录）")
    ap.add_argument("--json", default=None, help="JSON 报告输出路径")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not (repo / "catalog").is_dir():
        print(f"错误: 仓库路径下不存在 catalog/ 目录: {repo}", file=sys.stderr)
        return 2
    try:
        results = run_checks(repo)
    except Exception as e:  # 脚本自身错误 → 2
        import traceback
        traceback.print_exc()
        print(f"catalog lint 运行错误: {e}", file=sys.stderr)
        return 2

    print_report(results)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        blocks, warns = summarize(results)
        report = {
            "repo": str(repo),
            "summary": {"block": len(blocks), "warn": len(warns)},
            "results": [r.__dict__ for r in results],
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    blocks, _ = summarize(results)
    return 1 if blocks else 0


if __name__ == "__main__":
    sys.exit(main())
