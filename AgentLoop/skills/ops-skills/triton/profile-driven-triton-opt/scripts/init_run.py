#!/usr/bin/env python3
"""init_run.py — 为 profile-driven-triton-opt 创建唯一的本轮运行目录(RUN_DIR)。

保证「每次运行的结果隔离，且默认不看之前的结果」：
- 默认(fresh)总是新建一个 <out_root>/profile_opt_<ts>_<rid>/ 目录并返回其路径，
  **绝不返回/复用任何已存在的运行目录**——杜绝代理把历史 run 的 summary/candidates
  当成本轮交付(曾因平铺产物被静默继承导致契约闸校验陈旧结果)。
- 只有显式传 --reuse-last 才返回上一个 run 目录，用于刻意「续跑」场景。

用法：
  RUN_DIR=$(python3 scripts/init_run.py <out_root> [--label <name>])
  # 续跑(显式，不推荐缺省):
  RUN_DIR=$(python3 scripts/init_run.py <out_root> --reuse-last)

退出码:0 = 成功(RUN_DIR 打印到 stdout);1 = 错误。
"""

import argparse
import datetime
import json
import os
import random
import string
import sys

_rs = random.SystemRandom()
_B62 = string.digits + string.ascii_uppercase + string.ascii_lowercase


def _ts():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rid(n=4):
    return "".join(_rs.choice(_B62) for _ in range(n))


def _list_runs(out_root):
    """按目录 mtime 升序返回（--reuse-last 取最后一项 = 最新创建的 run）。"""
    if not os.path.isdir(out_root):
        return []
    runs = [p for p in os.listdir(out_root) if p.startswith("profile_opt_")]
    return sorted(runs, key=lambda p: os.path.getmtime(os.path.join(out_root, p)))


def _resolve_fresh(out_root, label):
    # 与 random 无关：每次新建，绝不命中已存在目录；同秒冲突由随机 rid 规避
    while True:
        ts = _ts()
        rid = _rid()
        name = f"profile_opt_{ts}_{rid}"
        if label:
            name = f"{name}_{label}"
        run_dir = os.path.join(out_root, name)
        if not os.path.exists(run_dir):
            return run_dir, "fresh run"


def _resolve_reuse(out_root):
    runs = _list_runs(out_root)
    if not runs:
        sys.exit(
            "ERROR: --reuse-last 但 <out_root> 下没有历史 run 目录；"
            "续跑前需先有一次 fresh run。建议重跑用默认(fresh)。"
        )
    return os.path.join(out_root, runs[-1]), f"reused {runs[-1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_root", help="产物根目录(如 out/)")
    ap.add_argument("--label", default="", help="可选标签附加到目录名(便于识别这一轮主题)")
    ap.add_argument(
        "--reuse-last",
        action="store_true",
        help="显式续跑：返回上一个 run 目录(默认与绝不复用，保持 fresh)",
    )
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    if args.reuse_last:
        run_dir, note = _resolve_reuse(out_root)
    else:
        run_dir, note = _resolve_fresh(out_root, args.label)
        os.makedirs(run_dir, exist_ok=True)

    meta = {
        "rid": os.path.basename(run_dir),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "out_root": out_root,
        "run_dir": run_dir,
        "note": note,
    }
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    hist = _list_runs(out_root)
    print(f"RUN_DIR={run_dir}", flush=True)
    print(
        f"[init_run] {note}; out_root 下现有 {len(hist)} 个 run 目录(fresh 模式不复用任何已有产物)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())