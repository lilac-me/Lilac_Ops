#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile → 瓶颈 Triton 算子识别（S2 前端解析器）。

语义对齐 docs/agent_workflow/profile-driven-triton-opt_design.md：
  - 定位 profile 里的 kernel_details.csv / step_trace_time.csv（zip 或已解压目录均可，不写死路径）。
  - 稳态 step 内按 op 类型聚合耗时，求 op 占比，跨 step 先算占比再平均。
  - 两档 Triton 判定：strong（可直接进 S4）/ weak（归入「待人工核定」，不改判）；
    内建 CANN 前缀（aclnn/Hccl/hcom 等）直接排除。
  - 输出 candidates.json（按占比降序），并保留命中该 candidate 的 Input Shapes 前 N 个去重样本。

用法：
  python parse_profile.py <profile.zip|dir> -o <outdir> [--config config.json] [--max-shapes N]
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import tempfile
import zipfile

from collections import defaultdict
from opportunities import identify

DEFAULT_CONFIG = {
    "bottleneck_threshold": 0.03,
    "builtin_prefixes": [
        "aclnn", "hcom", "Hccl", "aicpu", "Atc", "acl", "aclnnGroupedMatmul",
        "ForeachCopy", "RmsNorm",
    ],
    # strong 命中规则：非内建前缀且命中其一（正则）
    "strong_patterns": [
        r"_kernel$",          # causal_conv1d_fwd_kernel 等以 _kernel 结尾
        r"kernel",            # 含 kernel 字样兜底（用户已核实后进 S4）
        r"triton",
    ],
    # 内建但名字可疑、可能需要人工核定的"次内建"前缀（weak，不直接排除）
    "semi_builtin_prefixes": ["ascend", "npu", "aco_", "mep"],
    "max_shapes": 5,
    "min_shapes_str_length": 1,
}


def locate_csv(extracted_root, basename):
    """在解压根下递归寻找指定 CSV，返回绝对路径（找不到返回 None）。"""
    hits = glob.glob(os.path.join(extracted_root, "**", basename), recursive=True)
    return hits[0] if hits else None


def open_profile(source):
    """source 为 .zip 时解压到临时目录；否则原样返回目录。返回 (root, is_temp, cleanup_fn)。"""
    if os.path.isdir(source):
        return source, False, lambda: None
    if zipfile.is_zipfile(source):
        tmp = tempfile.mkdtemp(prefix="profile_opt_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp)
        return tmp, True, lambda: shutil.rmtree(tmp, ignore_errors=True)
    raise ValueError(f"输入既不是目录也不是 zip: {source}")


def load_kernel_details(csv_path):
    """读 kernel_details，返回 {step: {name: duration_sum}} + 每 step 列 Name/Step Id/Duration。

    以 header 名取列；Duration 数字转换失败的行跳过；Step Id 非数字的行（异步/N/A）无法归 step 跳过。
    """
    per_step_op = defaultdict(lambda: defaultdict(float))  # step -> name -> duration
    per_step_total = defaultdict(float)
    name_samples = defaultdict(list)                        # name -> (shape, dtype, core)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = str(row.get("Step Id", "")).strip()
            name = (row.get("Name") or "").strip()
            if step in ("", "N/A", "None") or name == "":
                continue
            try:
                step_f = float(step)
            except ValueError:
                continue
            step = str(int(step_f))
            try:
                dur = float(str(row.get("Duration(us)", "")).strip())
            except ValueError:
                continue
            if dur <= 0:
                continue
            per_step_op[step][name] += dur
            per_step_total[step] += dur
            shape = str(row.get("Input Shapes", "")).strip().strip('"')
            dtype = str(row.get("Input Data Types", "")).strip().strip('"')
            core = str(row.get("Accelerator Core", "")).strip()
            if shape and len(shape) >= DEFAULT_CONFIG["min_shapes_str_length"]:
                name_samples[name].append({"shape": shape, "dtype": dtype, "core": core})
    return per_step_op, per_step_total, name_samples


def load_steady_steps(step_trace_path, kernel_steps, fallback_all=True):
    """稳态 step 集 = step_trace_time.csv 的 Step 列；缺失则回退到 kernel_details 的 step 集。"""
    if step_trace_path and os.path.exists(step_trace_path):
        steps = []
        with open(step_trace_path, newline="") as f:
            for row in csv.DictReader(f):
                v = str(row.get("Step", "")).strip()
                try:
                    steps.append(str(int(float(v))))
                except ValueError:
                    continue
        if steps:
            return steps
    if fallback_all:
        return list(kernel_steps)
    return []


def classify_tier(name, cfg):
    """两档 + 排除。返回 'strong' / 'weak' / None(None=内建排除)。"""
    low = name.lower()
    if any(low.startswith(p.lower()) for p in cfg["builtin_prefixes"]):
        return None
    if any(low.startswith(p.lower()) for p in cfg["semi_builtin_prefixes"]):
        return "weak"
    if any(re.search(p, name, flags=re.IGNORECASE) for p in cfg["strong_patterns"]):
        return "strong"
    # 非内建但无强特征命名 → 弱候选（待人工核定，不改判）
    return "weak"


def build_candidates(per_step_op, per_step_total, steady_steps, name_samples, cfg):
    candidates = []
    names = {n for step in per_step_op.values() for n in step}
    for name in names:
        tier = classify_tier(name, cfg)
        if tier is None:
            continue  # 内建，排除
        per_step = []
        present = 0
        total_dur = 0.0
        for s in steady_steps:
            op_dur = per_step_op.get(s, {}).get(name, 0.0)
            st_total = per_step_total.get(s, 0.0)
            ratio = op_dur / st_total if st_total > 0 else 0.0
            if op_dur > 0:
                present += 1
                total_dur += op_dur
            per_step.append({"step": s, "op_total_us": round(op_dur, 3),
                             "step_total_us": round(st_total, 3), "ratio": round(ratio, 6)})
        coverage = present / len(steady_steps) if steady_steps else 0.0
        mean_ratio = (sum(p["ratio"] for p in per_step) / len(per_step)) if per_step else 0.0
        # 去重后的 Input Shapes 前 N 个
        _seen = set()
        shapes = []
        for smp in name_samples.get(name, []):
            key = (smp["shape"], smp["dtype"], smp["core"])
            if key in _seen:
                continue
            _seen.add(key)
            shapes.append(smp)
            if len(shapes) >= cfg["max_shapes"]:
                break
        candidates.append({
            "name": name,
            "tier": tier,
            "mean_op_ratio": round(mean_ratio, 6),
            "coverage": round(coverage, 4),
            "total_duration_us": round(total_dur, 3),
            "n_samples_shapes": len(shapes),
            "shapes": shapes,
            "per_step": per_step,
        })
    candidates.sort(key=lambda c: c["mean_op_ratio"], reverse=True)
    return candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="profile.zip 或已解压目录（不写死文件名）")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--config", default=None, help="可选 config.json 覆盖默认阈值/规则")
    ap.add_argument("--max-shapes", type=int, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    if args.max_shapes is not None:
        cfg["max_shapes"] = args.max_shapes

    os.makedirs(args.outdir, exist_ok=True)
    root, is_temp, cleanup = open_profile(args.source)
    try:
        kd_path = locate_csv(root, "kernel_details.csv")
        st_path = locate_csv(root, "step_trace_time.csv")
        if not kd_path:
            raise SystemExit("FAIL: profile 内找不到 kernel_details.csv，请检查 profiler 产物。")
        per_step_op, per_step_total, name_samples = load_kernel_details(kd_path)
        steady_steps = load_steady_steps(st_path, set(per_step_total.keys()))
        candidates = build_candidates(per_step_op, per_step_total, steady_steps,
                                      name_samples, cfg)
        threshold = float(cfg["bottleneck_threshold"])
        if threshold > 0:
            candidates = [c for c in candidates if c["mean_op_ratio"] >= threshold]
        opportunities = identify(kd_path, steady_steps, threshold,
                                 cfg.get("fusion_max_gap_us", 5.0), cfg["max_shapes"])
        matmul_names = {item["name"] for item in opportunities["matmul_mfu"]}
        for candidate in candidates:
            candidate["route"] = "matmul_mfu" if candidate["name"] in matmul_names else "latency"
        weak = [c for c in candidates if c["tier"] == "weak"]
        strong = [c for c in candidates if c["tier"] == "strong"]

        result = {
            "profile_source": args.source,
            "csvs": {"kernel_details": kd_path,
                     "step_trace_time": st_path or None},
            "config_used": {k: cfg[k] for k in
                            ("bottleneck_threshold", "builtin_prefixes",
                             "strong_patterns", "semi_builtin_prefixes")},
            "steady_steps": steady_steps,
            "summary": {
                "n_steady_steps": len(steady_steps),
                "n_strong": len(strong),
                "n_weak": len(weak),
            },
            "note": "weak 归入待人工核定，不改判，不自动进 S4。",
            "candidates": candidates,
        }
        with open(os.path.join(args.outdir, "opportunities.json"), "w") as f:
            json.dump(opportunities, f, indent=2, ensure_ascii=False, allow_nan=False)
        with open(os.path.join(args.outdir, "candidates.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"OK: {len(strong)} strong / {len(weak)} weak candidates (≥ {threshold:.1%}) "
              f"across {len(steady_steps)} steady steps")
        print(f"   top3: " + ", ".join(f"{c['name']}({c['mean_op_ratio']:.1%},{c['tier']})"
                                       for c in candidates[:3]))
        print(f"   -> {os.path.join(args.outdir, 'candidates.json')}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()