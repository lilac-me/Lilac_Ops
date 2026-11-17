#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Step7 (retest): 复测汇总 —— 对比参考报告失败项与本次复测结果, 逐项判定:
#   稳定复现（复测再次失败）| 未复现（本次通过）| 无法判定（无结果/未执行/桶超时/文件缺失）
# 用法: python summarize_retest.py <retest_dir> <ref_dir> [container]
#   产物: <retest_dir>/retest_report.json + retest_report.md
# 输入: run_retest.sh 生成的复测结果日志（result_ut_retest.log / result_atk_{acc,perf}_retest.log
#       / atk_logs/retest_{acc,perf}_*.log / timing.txt / op_timing_{acc,perf}.txt）
#       与参考报告 <ref_dir>/report.json（失败项清单 + 基准原因 + OWNER_MAP 负责人）
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize import OWNER_MAP, fmt_dur, read_timing, read_env_versions, classify_fail

VERDICT_STABLE = "稳定复现"
VERDICT_FIXED = "未复现（本次通过）"
VERDICT_UNKNOWN = "无法判定"


def owner_of_ut_file(file):
    base = os.path.basename(str(file)).replace("unit_tests/", "").replace(".py", "")
    if base.startswith("test_"):
        return OWNER_MAP.get(base[len("test_"):], "未知")
    return OWNER_MAP.get(base, "未知")


def parse_ut_retest(retest_dir):
    """return (still_failed:{file:count}, bucket_files:{file:part}, part_summary:{i:bool}, missing:set)"""
    path = os.path.join(retest_dir, "result_ut_retest.log")
    still, bucket_files, part_summary, missing = {}, {}, {}, set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            m = re.match(r"^rpart_files(\d+):\s*(.*)$", line)
            if m:
                for f in m.group(2).split():
                    bucket_files[f] = int(m.group(1))
                continue
            m = re.match(r"^rpart(\d+):", line)
            if m:
                part_summary[int(m.group(1))] = ("无汇总行" not in line)
                continue
            m = re.match(r"^SKIP missing file:\s*(.*)$", line)
            if m:
                missing.add(m.group(1).strip())
                continue
            m = re.match(r"^FAILED\s+(.*)$", line)
            if m:
                f = re.split(r"::", m.group(1))[0]
                still[f] = still.get(f, 0) + 1
    return still, bucket_files, part_summary, missing


def verdict_for_ut(file, still, bucket_files, part_summary, missing):
    if file in still:
        return VERDICT_STABLE, "复测仍 %d 条失败" % still[file]
    if file in missing:
        return VERDICT_UNKNOWN, "复测时文件缺失（未执行）"
    if file in bucket_files:
        i = bucket_files[file]
        if part_summary.get(i, True):
            return VERDICT_FIXED, "复测无 FAILED 用例（本次通过）"
        return VERDICT_UNKNOWN, "所在桶（rpart%d）超时/中断, 无法判定" % i
    return VERDICT_UNKNOWN, "未出现在复测运行清单"


def parse_atk_retest(retest_dir, kind):
    """return {op: status_str} for kind=acc|perf"""
    path = os.path.join(retest_dir, "result_atk_%s_retest.log" % kind)
    ops = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            m = re.match(r"\[(.*?)\]\s*status:\s*(\S+)", line)
            if m:
                ops[m.group(1)] = m.group(2)
                continue
            m = re.match(r"\[(.*?)\]\s*case gen FAILED", line)
            if m:
                ops[m.group(1)] = "CASE_GEN_FAILED"
    return ops


def verdict_for_atk(op, status, kind):
    low = (status or "").lower()
    if low in ("pass", "true"):
        return VERDICT_FIXED, "复测状态 %s=%s（本次通过）" % (kind, status)
    if low in ("failed", "false"):
        return VERDICT_STABLE, "复测状态 %s=%s（再次失败）" % (kind, status)
    if status == "CASE_GEN_FAILED":
        return VERDICT_UNKNOWN, "复测 case gen 失败, 未执行任务"
    return VERDICT_UNKNOWN, "复测无结果（%s=%s）" % (kind, status or "NO_RESULT")


def main():
    retest_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else ""
    container = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    ref = {}
    if ref_dir and os.path.exists(os.path.join(ref_dir, "report.json")):
        ref = json.load(open(os.path.join(ref_dir, "report.json"), encoding="utf-8"))
    ref_meta = ref.get("meta", {})

    wall, op_timing = read_timing(retest_dir)
    versions = read_env_versions(retest_dir)

    # ---------- 参考失败清单 ----------
    ref_ut_counts = {ff["file"]: ff["count"] for ff in ref.get("ut", {}).get("failed_files", [])}
    ref_acc = {f["op"]: f for f in ref.get("atk_accuracy", {}).get("failed_ops", [])}
    ref_perf = {f["op"]: f for f in ref.get("atk_performance", {}).get("failed_ops", [])}

    # ---------- 复测结果判定 ----------
    ut_items = []
    still, bucket_files, part_summary, missing = parse_ut_retest(retest_dir)
    for file in ref_ut_counts:
        verdict, detail = verdict_for_ut(file, still, bucket_files, part_summary, missing)
        ut_items.append({
            "file": file, "owner": owner_of_ut_file(file),
            "verdict": verdict, "detail": detail,
            "base": "基准失败 %d 条" % ref_ut_counts[file],
            "seconds": None,
        })

    acc_items, perf_items = [], []
    for kind, items, refmap, opkey in (
            ("acc", acc_items, ref_acc, "acc"),
            ("perf", perf_items, ref_perf, "perf")):
        ops = parse_atk_retest(retest_dir, kind)
        for op, base in sorted(refmap.items()):
            status = ops.get(op)
            if status is None:
                verdict, detail = VERDICT_UNKNOWN, "复测结果日志无该算子（未执行）"
            else:
                verdict, detail = verdict_for_atk(op, status, kind)
            items.append({
                "op": op, "owner": base.get("owner", OWNER_MAP.get(op, "未知")),
                "verdict": verdict, "detail": detail,
                "base": base.get("reason", "失败"),
                "seconds": op_timing[opkey].get(op),
            })

    acc_perf_items = ([{"kind": "accuracy", "name": it["op"], **it} for it in acc_items] +
                      [{"kind": "performance", "name": it["op"], **it} for it in perf_items])
    all_items = ([{"kind": "ut", "name": it["file"], **it} for it in ut_items] + acc_perf_items)
    stable = [it for it in all_items if it["verdict"] == VERDICT_STABLE]
    fixed = [it for it in all_items if it["verdict"] == VERDICT_FIXED]
    unknown = [it for it in all_items if it["verdict"] == VERDICT_UNKNOWN]

    by = defaultdict(list)
    for it in all_items:
        if it["verdict"] == VERDICT_STABLE:
            by[it["owner"]].append({"kind": it["kind"], "name": it["name"], "verdict": it["verdict"],
                                    "detail": it["detail"]})
    by_owner = [{"owner": k, "items": v} for k, v in sorted(by.items())]

    # ---------- retest_report.json ----------
    report = {
        "meta": {
            "date": os.path.basename(retest_dir).rstrip("_retest"),
            "retest_dir": retest_dir,
            "ref_dir": ref_dir,
            "ref_date": ref_meta.get("date"),
            "ref_guard_type": ref_meta.get("guard_type"),
            "ref_container": ref_meta.get("container"),
            "container": container,
            "env_versions": versions,
            "timing_seconds": wall,
            "timing_total_seconds": sum(wall.values()),
        },
        "totals": {"stable": len(stable), "not_reproduced": len(fixed), "undetermined": len(unknown)},
        "ut": {"items": ut_items},
        "atk_accuracy": {"items": acc_items},
        "atk_performance": {"items": perf_items},
        "by_owner": by_owner,
        "report_files": {
            "report": f"{retest_dir}/retest_report.md",
            "logs": [f"{retest_dir}/result_ut_retest.log", f"{retest_dir}/result_atk_retest.log",
                     f"{retest_dir}/result_atk_perf_retest.log", f"{retest_dir}/atk_logs/"],
        },
    }
    with open(os.path.join(retest_dir, "retest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- retest_report.md（五段, 对齐日常报告风格） ----------
    def cnt(items):
        return (sum(1 for x in items if x["verdict"] == VERDICT_STABLE),
                sum(1 for x in items if x["verdict"] == VERDICT_FIXED),
                sum(1 for x in items if x["verdict"] == VERDICT_UNKNOWN))

    s_ut, f_ut, u_ut = cnt(ut_items)
    s_acc, f_acc, u_acc = cnt(acc_items)
    s_perf, f_perf, u_perf = cnt(perf_items)
    ref_label = ""
    if ref_meta:
        ref_label = "（%s · 容器 %s）" % (ref_meta.get("guard_type", "?") or "?",
                                       ref_meta.get("container", "?") or "?")

    md = ["# 复测报告（失败项稳定性判定 · 参考 %s%s）" % (ref_meta.get("date", "?"), ref_label), ""]
    md += ["## 一、复测摘要", "",
           "- 复测范围: UT 失败文件 %d · ATK 精度失败算子 %d · ATK 性能失败算子 %d（只重跑失败项）"
           % (len(ut_items), len(acc_items), len(perf_items)),
           "- 总判定: **稳定复现 %d** · 未复现 %d · 无法判定 %d" % (len(stable), len(fixed), len(unknown)),
           "", "### UT",
           "- 稳定复现 %d · 未复现 %d · 无法判定 %d" % (s_ut, f_ut, u_ut)]
    for it in ut_items:
        md.append("- %s@%s — %s" % (os.path.basename(it["file"]), it["owner"], it["verdict"]))
    md += ["", "### ATK 精度",
           "- 稳定复现 %d · 未复现 %d · 无法判定 %d" % (s_acc, f_acc, u_acc)]
    for it in acc_items:
        md.append("- %s@%s — %s" % (it["op"], it["owner"], it["verdict"]))
    md += ["", "### ATK 性能",
           "- 稳定复现 %d · 未复现 %d · 无法判定 %d" % (s_perf, f_perf, u_perf)]
    for it in perf_items:
        md.append("- %s@%s — %s" % (it["op"], it["owner"], it["verdict"]))
    if ref.get("atk_performance", {}).get("noresult_ops"):
        md.append("- 注: 性能无结果 %d 项（首轮无基准）非失败项, 按『只重跑失败项』未纳入复测"
                  % len(ref.get("atk_performance", {}).get("noresult_ops", [])))

    md += ["", "## 二、元信息", "",
           "- 复测容器: %s（A3 Ascend910） · 复测目录: `%s`" % (container, retest_dir),
           "- 参考报告: `%s`（%s · 容器 %s）" % (ref_dir, ref_meta.get("guard_type", "?"),
                                            ref_meta.get("container", "?"))]
    if versions:
        md.append("- 环境: " + " · ".join("%s %s" % (k, v) for k, v in versions.items()))
    md.append("- 阶段耗时: " + (" · ".join("%s %s" % (k, fmt_dur(v)) for k, v in sorted(wall.items())) or "未记录"))
    if wall:
        md.append("- 总耗时: %s" % fmt_dur(sum(wall.values())))

    md += ["", "## 三、逐项判定详情", ""]
    for it in ut_items:
        md += ["### [ut] %s — 负责人: %s" % (it["file"], it["owner"]), "",
               "- 判定: **%s**" % it["verdict"],
               "- 基准: %s" % it["base"],
               "- 复测: %s" % it["detail"],
               "- 复测日志: `result_ut_retest.log`（含各桶 rpart 明细）", ""]
    for it in acc_perf_items:
        lp = os.path.join(retest_dir, "atk_logs", "retest_%s_%s.log" % (
            "acc" if it["kind"] == "accuracy" else "perf", it["name"]))
        sec = "，耗时 %s" % fmt_dur(it["seconds"]) if it.get("seconds") else ""
        extra = ""
        if it["verdict"] == VERDICT_STABLE:
            reason, _ = classify_fail(lp)
            extra = "；失败分类: %s" % reason
        md += ["### [%s] %s — 负责人: %s" % (it["kind"], it["name"], it["owner"]), "",
               "- 判定: **%s**" % it["verdict"],
               "- 基准原因: %s" % it["base"],
               "- 复测: %s%s%s" % (it["detail"], sec, extra),
               "- 复测日志: `%s`" % lp, ""]

    md += ["", "## 四、按负责人聚合（稳定复现项）", ""]
    if not by_owner:
        md.append("无稳定复现项 ✅")
    else:
        for o in by_owner:
            md.append("- **%s（%d 项）**" % (o["owner"], len(o["items"])))
            for it in o["items"]:
                md.append("  - [%s] %s — %s" % (it["kind"], it["name"], it["detail"]))

    md += ["", "## 五、报告与归档位置", "",
           "- 本报告: `%s/retest_report.md`、`%s/retest_report.json`" % (retest_dir, retest_dir),
           "- 复测日志: `%s/result_ut_retest.log`、`result_atk_acc_retest.log`、`result_atk_perf_retest.log`、`atk_logs/`" % retest_dir,
           "- 参考报告: `%s/report.md`（原始看护）" % ref_dir, ""]
    with open(os.path.join(retest_dir, "retest_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- console ----------
    print("=" * 72)
    print("复测参考: %s（%s · 容器 %s）→ 失败项 UT %d 文件 / ACC %d / PERF %d" % (
        ref_meta.get("date", "?"), ref_meta.get("guard_type", "?"), ref_meta.get("container", "?"),
        len(ut_items), len(acc_items), len(perf_items)))
    print("总判定: 稳定复现 %d · 未复现 %d · 无法判定 %d" % (len(stable), len(fixed), len(unknown)))
    print("UT: 稳定复现 %d / 未复现 %d / 无法判定 %d" % (s_ut, f_ut, u_ut))
    for it in ut_items:
        print("失败用例：%s@%s — %s" % (os.path.basename(it["file"]).replace(".py", ""), it["owner"], it["verdict"]))
    print("ATK 精度: 稳定复现 %d / 未复现 %d / 无法判定 %d" % (s_acc, f_acc, u_acc))
    for it in acc_items:
        print("失败用例：%s@%s — %s" % (it["op"], it["owner"], it["verdict"]))
    print("ATK 性能: 稳定复现 %d / 未复现 %d / 无法判定 %d" % (s_perf, f_perf, u_perf))
    for it in perf_items:
        print("失败用例：%s@%s — %s" % (it["op"], it["owner"], it["verdict"]))
    print("=" * 72)
    print("复测报告已写入: %s/retest_report.md, retest_report.json" % retest_dir)


if __name__ == "__main__":
    main()
