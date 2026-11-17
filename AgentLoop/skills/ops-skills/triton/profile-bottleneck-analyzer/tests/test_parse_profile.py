# -*- coding: utf-8 -*-
"""parse_profile 单测：合成 CSV 验证两档判定 / 内建排除 / 跨 step 占比平均 / 阈值过滤 / shape 去重。

不依赖真实 zip。真实数据冒烟需显式提供数据目录（环境变量 `PROFILE_SMOKE_DIR`），
未提供时自动跳过，保证在任意机器 / CI 上都能跑通。
"""

import csv
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import parse_profile as pp  # noqa: E402

KERNEL_HEADER = [
    "Step Id", "Device_id", "Task ID", "Name", "Accelerator Core",
    "Duration(us)", "Input Shapes", "Input Data Types",
]


def write_csv(dirpath, fname, header, rows):
    path = os.path.join(dirpath, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def make_tree(kernel_rows, step_rows, steady=True):
    """在临时目录下构造一个 ASCEND_PROFILER_OUTPUT 风格的 profile 目录。"""
    tmp = tempfile.mkdtemp(prefix="parse_prof_")
    asc = os.path.join(tmp, "run1", "ASCEND_PROFILER_OUTPUT")
    os.makedirs(asc)
    write_csv(asc, "kernel_details.csv", KERNEL_HEADER, kernel_rows)
    if step_rows is not None:
        write_csv(asc, "step_trace_time.csv",
                  ["Device_id", "Step", "Computing"], step_rows)
    return tmp


def run(tmp, threshold=0.03, max_shapes=3):
    """直接调内部解析函数（不经 argparse），返回通过阈值过滤的 candidates。"""
    kd = os.path.join(tmp, "run1", "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    st = os.path.join(tmp, "run1", "ASCEND_PROFILER_OUTPUT", "step_trace_time.csv")
    per_step_op, per_step_total, samples = pp.load_kernel_details(kd)
    steps = pp.load_steady_steps(st, set(per_step_total.keys()))
    cfg = dict(pp.DEFAULT_CONFIG)
    cfg["bottleneck_threshold"] = threshold
    cfg["max_shapes"] = max_shapes
    cands = pp.build_candidates(per_step_op, per_step_total, steps, samples, cfg)
    return [c for c in cands if c["mean_op_ratio"] >= threshold]


def rows():
    """合成数据口径：
     step1 总 200us: causal=40(0.20) warp=60(0.30) aclnn=100(内建排除)
     step2 总 100us: causal=20(0.20)          aclnn=80(内建排除)
     causal 跨 step 均值 (0.20+0.20)/2=0.20, coverage=1.0
     warp   step2 缺失 → 占比 0 → 均值 (0.30+0)/2=0.15, coverage=0.5
    """
    return [
        ["1", "0", "1", "causal_fwd_kernel", "AI_CORE", "40.0", '"40,1,128"', "FLOAT"],
        ["1", "0", "1", "warp_flatten", "AI_VECTOR_CORE", "60.0", '"64,64"', "FLOAT"],
        ["1", "0", "1", "aclnnLayerNorm", "AI_CORE", "100.0", "N/A", "N/A"],
        ["2", "0", "1", "causal_fwd_kernel", "AI_CORE", "20.0", '"40,1,128"', "FLOAT"],
        ["2", "0", "1", "aclnnLayerNorm", "AI_CORE", "80.0", "N/A", "N/A"],
    ], [["0", "1", "200"], ["0", "2", "100"]]


def test_builtin_excluded_and_strong_detected():
    kr, st = rows()
    tmp = make_tree(kr, st)
    cands = run(tmp)
    names = [c["name"] for c in cands]
    assert "causal_fwd_kernel" in names       # _kernel → strong
    assert "aclnnLayerNorm" not in names      # 内建前缀排除
    causal = next(c for c in cands if c["name"] == "causal_fwd_kernel")
    assert causal["tier"] == "strong"


def test_ratio_averaged_across_steps():
    kr, st = rows()
    tmp = make_tree(kr, st)
    cands = run(tmp)
    causal = next(c for c in cands if c["name"] == "causal_fwd_kernel")
    # step1: 40/200=0.20, step2: 20/100=0.20 → mean=0.20
    assert causal["mean_op_ratio"] == pytest.approx(0.20, abs=1e-4)
    assert causal["coverage"] == 1.0
    warp = next(c for c in cands if c["name"] == "warp_flatten")
    assert warp["mean_op_ratio"] == pytest.approx(0.15, abs=1e-4)  # step2 缺失→0，均值被稀释
    assert warp["coverage"] == 0.5


def test_non_builtin_non_pattern_is_weak():
    kr, st = rows()
    tmp = make_tree(kr, st)
    cands = run(tmp)
    warp = next(c for c in cands if c["name"] == "warp_flatten")
    # warp_flatten 非内建前缀、无 _kernel/trion 特征 → weak（待人工核定）
    assert warp["tier"] == "weak"


def test_threshold_filtering():
    kr, st = rows()
    tmp = make_tree(kr, st)
    # causal=0.20, warp=0.15
    cands = run(tmp, threshold=0.17)   # causal 保留，warp 被滤
    assert any(c["name"] == "causal_fwd_kernel" for c in cands)
    assert not any(c["name"] == "warp_flatten" for c in cands)
    cands_all = run(tmp, threshold=0.10)
    assert any(c["name"] == "warp_flatten" for c in cands_all)


def test_shape_dedup_and_max():
    kr, st = rows()
    # 同一 kernel 多次出现不同 shape，去重后保留前 max_shapes 个
    kr_extra = kr + [
        ["1", "0", "1", "causal_fwd_kernel", "AI_CORE", "1.0", '"40,1,128"', "FLOAT"],
        ["1", "0", "1", "causal_fwd_kernel", "AI_CORE", "1.0", '"80,2,256"', "FLOAT"],
    ]
    tmp = make_tree(kr_extra, st)
    cands = run(tmp, max_shapes=3)
    causal = next(c for c in cands if c["name"] == "causal_fwd_kernel")
    shapes = [s["shape"] for s in causal["shapes"]]
    # 按 CSV 出现顺序去重：'40,1,128' 先于 '80,2,256'
    assert shapes == ["40,1,128", "80,2,256"]
    assert causal["n_samples_shapes"] == 2
    assert len(shapes) <= 3


def test_steady_steps_fallback_when_no_trace():
    kr, _st = rows()
    # step_trace 缺失 → 回退到 kernel_details 的 step 集
    tmp = make_tree(kr, None)
    cands = run(tmp)
    names = [c["name"] for c in cands]
    assert "causal_fwd_kernel" in names
    # 仅 step1、step2 在内 → step2 占比也算进均值
    causal = next(c for c in cands if c["name"] == "causal_fwd_kernel")
    assert causal["coverage"] > 0


# 真实 profile 冒烟数据目录由环境变量提供，仓库内不写死任何本机绝对路径：
#   PROFILE_SMOKE_DIR=/path/to/profiles python -m pytest tests/test_parse_profile.py
# 未设置或目录不存在时自动跳过（CI / 他人机器无需该数据）。
REAL_PROFILE_DIR = os.environ.get("PROFILE_SMOKE_DIR", "").strip()
_HAS_REAL_PROFILE = bool(REAL_PROFILE_DIR) and os.path.isdir(REAL_PROFILE_DIR)


@pytest.mark.skipif(
    not _HAS_REAL_PROFILE,
    reason="未设置 PROFILE_SMOKE_DIR（指向含 *.zip 的 profile 目录）或目录不存在，跳过真实数据冒烟",
)
def test_real_zips_smoke(tmp_path):
    """对真实 profile zip 逐个冒烟：能在候选人里找到 'kernel' 结尾的 strong 项，且不抛异常。"""
    zips = [os.path.join(REAL_PROFILE_DIR, f) for f in os.listdir(REAL_PROFILE_DIR)
            if f.endswith(".zip")]
    if not zips:
        pytest.skip(f"PROFILE_SMOKE_DIR={REAL_PROFILE_DIR} 下无 zip，跳过真实数据冒烟")
    seen_strong = 0
    for z in zips:
        out = os.path.join(tmp_path, os.path.basename(z).replace(".zip", ""))
        root, is_temp, cleanup = pp.open_profile(z)
        try:
            kd = pp.locate_csv(root, "kernel_details.csv")
            if not kd:
                continue
            per_step_op, per_step_total, samples = pp.load_kernel_details(kd)
            steps = pp.load_steady_steps(pp.locate_csv(root, "step_trace_time.csv"),
                                         set(per_step_total.keys()))
            cands = pp.build_candidates(per_step_op, per_step_total, steps, samples,
                                        dict(pp.DEFAULT_CONFIG))
            if any(c["tier"] == "strong" and c["name"].endswith("kernel") for c in cands):
                seen_strong += 1
        finally:
            cleanup()
    assert seen_strong >= 1, "真实 zip 未识别到任何 _kernel strong 候选"


def test_zip_input_extracts_and_parses(tmp_path):
    # 打包成 zip 也应能通过 open_profile → locate_csv 正常定位 CSV
    kr, st = rows()
    tree = make_tree(kr, st)
    zpath = shutil.make_archive(str(tmp_path / "prof"), "zip", tree)
    root, is_temp, cleanup = pp.open_profile(zpath)
    try:
        assert is_temp is True
        kd = pp.locate_csv(root, "kernel_details.csv")
        assert kd is not None
        per_step_op, per_step_total, _ = pp.load_kernel_details(kd)
        assert "causal_fwd_kernel" in per_step_op["1"]
    finally:
        cleanup()