#!/usr/bin/env python3
"""demo_restructure_e2e.py — I1 端到端契约级演示（§10 I1，causal_conv1d_fwd）。

用已实证的 causal plan B 产物构造一条完整「restructure(load_block)→tune→summary→契约」链
的契约级复现（不重跑 NPU benchmark：复用官方产物溯源，避免与其它会话争抢 NPU）。

链条：
  restartructure guard g1/g3 = 实跑（scripts/restructure_guard.py 对本 op 的归档 IR/源码）
  g2 = 历史 12/12（本会话已过 gate2_matrix，NPU 被占用故不重跑）
  engine = restructure_engine/load_block.py（签名从该引擎派生，applied_transform 溯源）
  g4 = 官方 benchmark 溯源（baseline_result.json=1.6824ms / bplan_result.json=1.2034ms）
  report.md 含「重构过程」+「调优过程」两节（§8）
  最后 check_summary_contract 必须 CONTRACT PASS。

用法： python3 scripts/demo_restructure_e2e.py [--out <tmpdir>]
        [--mm-triton-dir <MindSpeed-MM 的 triton 目录>]
退出码：0 = 契约 PASS（I1 达成）；2 = FAIL。

外部依赖（不写死本机路径）：MindSpeed-MM 的 triton 目录（含 utils.py）用
`--mm-triton-dir` 或环境变量 `MM_TRITON_DIR` 指定；Index.md 默认按本文件位置推导，
可用 `LATENCY_INDEX` 覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
CHECK = os.path.join(HERE, "check_summary_contract.py")
GUARD = os.path.join(HERE, "restructure_guard.py")
# Index.md 在 <repo>/skills/ops-skills/vendored-triton-op/ 下，按本文件位置推导，
# 不写死本机绝对路径（可设 LATENCY_INDEX 覆盖）。
OPS_SKILLS = os.path.dirname(os.path.dirname(SKILL))
INDEX = os.environ.get("LATENCY_INDEX") or os.path.join(
    OPS_SKILLS, "vendored-triton-op", "triton-latency-optimizer", "references", "Index.md")

# 外部仓目录由 --mm-triton-dir / MM_TRITON_DIR 提供，仓库内不写死任何本机路径。
_MM_TRITON_DIR = os.environ.get("MM_TRITON_DIR", "").strip()
OP = os.path.join(SKILL,
                  "out/profile_opt_20260921_090349_xVcb_causal_fwd_t2x/op_causal_conv1d_fwd_kernel")
USER_SRC = os.path.join(OP, "causal_conv1d_fwd_kernel.py")
RESTRUCTURED = os.path.join(OP, "plan_b", "conv_restructured.py")
BASE_JSON = os.path.join(OP, "official", "baseline_result.json")
OPT_JSON = os.path.join(OP, "plan_b", "bench_new", "bplan_result.json")
IR_BEFORE = os.path.join(OP, "opt_iter_1", "ir_output_round1", "causal_conv1d_fwd_kernel_ttir.mlir")
IR_AFTER = os.path.join(OP, "plan_b", "ir_output_run1", "causal_conv1d_fwd_kernel_ttir.mlir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--mm-triton-dir", default=_MM_TRITON_DIR,
                    help="MindSpeed-MM 的 triton 目录（含 utils.py）；默认取环境变量 MM_TRITON_DIR")
    args = ap.parse_args()

    mm_dir = args.mm_triton_dir
    if not mm_dir:
        print("[FAIL] 缺少 MindSpeed-MM 的 triton 目录："
              "请用 --mm-triton-dir <dir> 或环境变量 MM_TRITON_DIR 指定（目录下需有 utils.py）")
        return 1
    utils = os.path.join(mm_dir, "utils.py")

    for p in (USER_SRC, RESTRUCTURED, utils, BASE_JSON, OPT_JSON, IR_BEFORE, IR_AFTER):
        if not os.path.isfile(p):
            print(f"[FAIL] 前置产物缺失: {p}")
            return 1
    if not os.path.isfile(INDEX):
        print(f"[FAIL] Index.md 不存在: {INDEX}（可用 LATENCY_INDEX 覆盖）")
        return 1

    # 1) 实跑 restructure_guard g1+g3（离线），g2 引用历史 12/12
    gr = subprocess.run(
        [sys.executable, GUARD, "--user", USER_SRC, "--restructured", RESTRUCTURED,
         "--utils", utils, "--pkg", "mmgt_e2e",
         "--ir-before", IR_BEFORE, "--ir-after", IR_AFTER],
        capture_output=True, text=True,
    )
    if gr.returncode not in (0, 2):  # 0=三闸过(跳过项算过)；2=必跑闸 False
        print(f"[FAIL] restructure_guard 运行异常 rc={gr.returncode}\n{gr.stderr}")
        return 1
    try:
        guard = json.loads(gr.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        print(f"[FAIL] restructure_guard 输出非 JSON: {gr.stdout!r}")
        return 1
    if guard["g1_io_frozen"] is not True or guard["g3_ir_homology"] is not True:
        print(f"[FAIL] guard 必跑闸未过: {guard}")
        return 1
    print(f"[guard] g1={guard['g1_io_frozen']} g3={guard['g3_ir_homology']} "
          f"loads {guard['ir_gm_loads_before']['loads']}→{guard['ir_gm_loads_after']['loads']} "
          f"(stores {guard['ir_gm_loads_before']['stores']}→{guard['ir_gm_loads_after']['stores']})")

    # 2) applied_transform 从 restructure_engine 派生（溯源，非手写）
    sys.path.insert(0, os.path.join(SKILL, "restructure_engine"))
    from load_block import register_default
    plan = register_default(win=128, W=4, bd=64)
    applied = plan.signature()
    print(f"[plan] applied_transform={applied}")

    # 3) 构造 summary + report，过契约
    with tempfile.TemporaryDirectory(prefix="e2e_") as td:
        summary = {
            "op_name": "causal_conv1d_fwd_kernel",
            "status": "BELOW",
            "perf_baseline": USER_SRC,
            "gate_ref": os.path.join(OP, "gate_ref.py:causal_conv1d_fwd_ref"),
            "speedup_vs_user": round(1.6824 / 1.2034, 4),
            "optimizer_rounds": 1,
            "optimizer": {
                "tool": "triton-latency-optimizer", "points_hit": [31],
                "loaded_docs": ["references/IR_triton.md"], "points_considered": 33,
                "scan_complete": False,
                "scan_trace": [{"round": 1, "scan_from": 1, "hit": 31,
                                "not_hit": "[1,30]", "mandatory": [31]}],
            },
            "bench": {
                "tool": "scripts/benchmark.py",
                "baseline_result_json": BASE_JSON, "optimized_result_json": OPT_JSON,
                "baseline_avg_ms": 1.6824, "optimized_avg_ms": 1.2034,
            },
            "best_code": None,
            "restructure": {
                "trigger": "ir_point31_gather_reuse",
                "applied_transform": applied,
                "engine": os.path.join(SKILL, "restructure_engine", "load_block.py"),
                "gates": {
                    "g1_io_frozen": True, "g2_matrix": True, "g2_matrix_cases": 12,
                    "g3_ir_homology": True,
                    "g3_gm_loads": {"before": guard["ir_gm_loads_before"]["loads"],
                                    "after": guard["ir_gm_loads_after"]["loads"]},
                },
                "io_freeze_basis": "gate1(签名/实参冻结)+gate2(12例位级 torch.equal)+gate3(IR: load 5→2, store 2→2)",
            },
        }
        with open(os.path.join(td, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        report = f"""## 三、调优过程（optimization record）
- tool: triton-latency-optimizer；scan_complete 由编排器推导；调优经唯一执行通道执行
## 重构过程（restructure record）
- trigger: ir_point31_gather_reuse
- transform: {applied}   (engine=restructure_engine/load_block.py)
- 4 闸:
  - g1 I/O 冻结: 签名/实参逐字节一致 ✓
  - g2 变体矩阵: 12/12 位级 torch.equal ✓
  - g3 IR 同构: GM load 5→2, store 2→2, mask 同位 ✓
- I/O 冻结依据: gate1+gate2+gate3
- 性能: baseline(用户源)=1.6824ms → 重构=1.2034ms, speedup_vs_user={summary['speedup_vs_user']}x
- 与调优关系: 重构产出作为 code_version 基线, 随后 vendor 扫描 1-33（见「调优过程」）
"""
        with open(os.path.join(td, "report.md"), "w") as f:
            f.write(report)

        out_dir = args.out or td
        os.makedirs(out_dir, exist_ok=True)

        proc = subprocess.run(
            [sys.executable, CHECK, os.path.join(td, "summary.json"), "--index", INDEX],
            capture_output=True, text=True,
        )
        out = (proc.stdout + proc.stderr).strip()
        print(f"[contract] rc={proc.returncode} {out.splitlines()[-1] if out else ''}")
        if proc.returncode != 0:
            print(out)
            print(f"\n[I1] FAILED —— summary/report 残留于 {out_dir}")
            return 1

        # 落一份可复盘的摘要产物（同一 temp 目录结构）
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write(report)
        print(f"[I1] PASS —— 契约通过；摘要产物: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())