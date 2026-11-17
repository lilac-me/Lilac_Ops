#!/usr/bin/env python3
"""test_restructure_contract.py — restructure-stage 契约单测（§10 U1–U7，无需 NPU）。

对每个用例构造一个临时 op 工作目录（summary.json + report.md + 官方 bench result JSON），
用 subprocess 调 `scripts/check_summary_contract.py --index <Index.md>`，断言退出码。

覆盖：
  U1 纯 tune（无 restructure）+ 合规 report → PASS（restructure 闸为空操作，不回归）
  U2 restructure + optimizer_rounds>0 + 双节 report → PASS
  U3 restructure 但 optimizer_rounds<=0 → FAIL（反滥用：禁止"重构+直接判 REACHED"）
  U4 restructure.gates.g2_matrix=false → FAIL（4 闸未全过）
  U5 restructure 缺 trigger → FAIL（缺字段）
  U6 restructure 但 report 缺「重构过程」→ FAIL（用户报告记录缺失）
  U7 optimizer_rounds>0 但 report 缺「调优过程」→ FAIL（用户报告记录缺失）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
CHECK = os.path.join(HERE, "check_summary_contract.py")
# Index.md 在 <repo>/skills/ops-skills/vendored-triton-op/ 下，按本文件位置推导，
# 不写死本机绝对路径（可设 LATENCY_INDEX 覆盖）。
OPS_SKILLS = os.path.dirname(os.path.dirname(SKILL))
INDEX = os.environ.get("LATENCY_INDEX") or os.path.join(
    OPS_SKILLS, "vendored-triton-op", "triton-latency-optimizer", "references", "Index.md")
ENGINE = os.path.join(SKILL, "restructure_engine", "load_block.py")

N = 33  # vendor Index.md 最大点号


def write_bench_json(op_dir):
    """写官方产物 result JSON（供 bench 溯源闸读取 implementation.avg_latency_ms）。"""
    base = {"implementation": {"avg_latency_ms": 1.6824}}
    opt = {"implementation": {"avg_latency_ms": 1.2034}}
    with open(os.path.join(op_dir, "baseline_result.json"), "w") as f:
        json.dump(base, f)
    with open(os.path.join(op_dir, "optimized_result.json"), "w") as f:
        json.dump(opt, f)


# —— 一段合法且穷尽自洽的 scan_trace（rounds=1，应用轮命中强制点 31）——
def valid_scan_trace():
    return [{
        "round": 1, "scan_from": 1, "hit": 31, "not_hit": "[1,30]",
        "mandatory": [31], "scan_complete": False,
    }]


def base_summary(**over):
    s = {
        "op_name": "op_x",
        "status": "BELOW",
        "perf_baseline": "official/op_x.py",
        "gate_ref": "gate_ref.py:op_x_ref",
        "speedup_vs_user": 1.398,
        "bench": {
            "tool": "scripts/benchmark.py",
            "baseline_result_json": "baseline_result.json",
            "optimized_result_json": "optimized_result.json",
            "baseline_avg_ms": 1.6824,
            "optimized_avg_ms": 1.2034,
        },
    }
    s.update(over)
    return s


def run_case(name, summary, report_body, expect_fail, rounds):
    with tempfile.TemporaryDirectory(prefix="contract_u_") as td:
        write_bench_json(td)
        with open(os.path.join(td, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(td, "report.md"), "w") as f:
            f.write(report_body)
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, CHECK, os.path.join(td, "summary.json"), "--index", INDEX],
            capture_output=True, text=True, env=env,
        )
        out = (proc.stdout + proc.stderr).strip()
        rc = proc.returncode
        ok = (rc == 2) if expect_fail else (rc == 0)
        if not ok:
            print(f"[FAIL] {name}: expect rc={'2' if expect_fail else '0'} got {rc}\n      {out}")
            sys.exit(1)
        tag = "FAIL(契约)" if expect_fail else "PASS"
        print(f"[PASS] {name}: rc={rc} {tag}")
    return 0


# —— report 模板 ——
TUNE_ONLY = (
    "## 三、调优过程\n"
    "- tool: triton-latency-optimizer\n"
    "- scan_complete=true / points_hit=[31]\n"
    "- 调优经 triton-latency-optimizer 执行\n"
)
RESTRUCTURE_AND_TUNE = (
    TUNE_ONLY +
    "## 重构过程\n"
    "- trigger: ir_point31_gather_reuse\n"
    "- g1: I/O 冻结 ✓\n"
    "- g2: 位级矩阵 ✓\n"
    "- g3: IR 同构 ✓\n"
    "- speedup_vs_user: 1.398x\n"
)
RESTRUCTURE_MISSING = TUNE_ONLY  # 有调优无重构
TUNE_MISSING = "## 只有一些其他内容\n没有调优过程节。\n"


def main():
    idx = "--index {}已存在".format(INDEX)
    if not os.path.isfile(INDEX):
        print(f"[FAIL] Index.md 不存在: {INDEX}")
        return 1

    # U1 纯 tune，无 restructure，report 合规 → PASS（restructure 空操作）
    run_case("U1 纯tune合规report",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }),
             TUNE_ONLY, expect_fail=False, rounds=1)

    # U2 restructure + rounds=1 + 双节 report → PASS
    restructure_ok = {
        "trigger": "ir_point31_gather_reuse",
        "applied_transform": "load_block{win=128, W=4, bd=64}",
        "engine": ENGINE,
        "gates": {"g1_io_frozen": True, "g2_matrix": True, "g3_ir_homology": True},
        "io_freeze_basis": "gate1+gate2+gate3",
    }
    run_case("U2 restructure+rondOrds>0 双节report",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }, restructure=restructure_ok),
             RESTRUCTURE_AND_TUNE, expect_fail=False, rounds=1)

    # U3 restructure 但 rounds<=0 → FAIL（反滥用）
    run_case("U3 restructure但rounds=0",
             base_summary(optimizer_rounds=0, restructure=restructure_ok),
             RESTRUCTURE_AND_TUNE, expect_fail=True, rounds=0)

    # U4 gates.g2_matrix=false → FAIL
    bad_gates = dict(restructure_ok, gates={"g1_io_frozen": True, "g2_matrix": False, "g3_ir_homology": True})
    run_case("U4 gates.g2=false",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }, restructure=bad_gates),
             RESTRUCTURE_AND_TUNE, expect_fail=True, rounds=1)

    # U5 restructure 缺 trigger → FAIL
    no_trig = dict(restructure_ok, trigger="")
    run_case("U5 restructure缺trigger",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }, restructure=no_trig),
             RESTRUCTURE_AND_TUNE, expect_fail=True, rounds=1)

    # U6 restructure 但 report 缺「重构过程」→ FAIL
    run_case("U6 report缺重构过程",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }, restructure=restructure_ok),
             RESTRUCTURE_MISSING, expect_fail=True, rounds=1)

    # U7 rounds>0 但 report 缺「调优过程」→ FAIL
    run_case("U7 report缺调优过程",
             base_summary(optimizer_rounds=1, optimizer={
                 "tool": "triton-latency-optimizer", "points_hit": [31],
                 "loaded_docs": ["index"], "points_considered": N,
                 "scan_complete": False, "scan_trace": valid_scan_trace(),
             }),
             TUNE_MISSING, expect_fail=True, rounds=1)

    print("\n=== U1–U7 all PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())