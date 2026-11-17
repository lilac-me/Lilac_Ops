#!/usr/bin/env python
# Step6: summarize UT + ATK acc + ATK perf results; owner_map responsibility.
# Report format (user spec 2026-08-26 v4, skill 化反馈后定稿):
#  Part1 结果摘要（三栏统计 + 失败项@负责人速览）
#  Part2 元信息（日期/看护路径/容器/环境版本/阶段耗时/总耗时）
#  Part3 失败项详情（原因 + 日志线索 + 耗时）
#  Part4 按负责人聚合视图（by_owner）
#  - UT 失败明细精简到"文件→条数"（markdown），完整用例清单保留于 report.json 与 result_ut.log
#  - ATK status None/NO_RESULT 独立列为无结果并归类原因（perf 首轮无基准），不并入失败
# 用法: python summarize.py <date_dir> [guard_type] [container_ip]
#       guard_type: daily(默认, 日常看护) | version(版本看护)
# 2026-09-08 脱敏改造: 负责人映射不再内嵌（开源合规），运行时从
#   /mnt/share/ops_nightly_agent/owner_map/owner_map.json 读取（key=算子名, value=负责人标识）；
#   文件缺失/解析失败时回退为空映射（owner 显示"未知"），不影响报告生成。
import json
import os
import re
import sys
from collections import Counter, defaultdict

OWNER_MAP_FILE = "/mnt/share/ops_nightly_agent/owner_map/owner_map.json"


def _load_owner_map():
    try:
        with open(OWNER_MAP_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


OWNER_MAP = _load_owner_map()

TIMING_FILE = "timing.txt"           # 每行: <phase> <seconds>   (ut|acc|perf)
OP_TIMING = {"acc": "op_timing_acc.txt", "perf": "op_timing_perf.txt"}


def fmt_dur(sec):
    sec = int(sec or 0)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def read_timing(date_dir):
    """wall: {ut|acc|perf: seconds}; ops: {acc|perf: {op: seconds}}"""
    # 2026-08-29 修复: wall 初始为空 dict 时 `p[0] in wall` 恒 False → 阶段耗时丢失。
    wall = {"ut": 0, "acc": 0, "perf": 0}
    ops = {k: {} for k in OP_TIMING}
    tf = os.path.join(date_dir, TIMING_FILE)
    if os.path.exists(tf):
        for line in open(tf, encoding="utf-8", errors="ignore"):
            p = line.split()
            if len(p) >= 2 and p[0] in wall and p[1].isdigit():
                wall[p[0]] = int(p[1])
    for phase, fn in OP_TIMING.items():
        pf = os.path.join(date_dir, fn)
        if os.path.exists(pf):
            for line in open(pf, encoding="utf-8", errors="ignore"):
                s = line.split()
                if len(s) >= 3 and s[1] and s[2].isdigit():
                    ops[phase][s[1]] = int(s[2])
    return wall, ops


def read_env_versions(date_dir):
    """Best effort from $WS/versions.txt (written by env install), else {}.
    2026-08-29 修复: versions.txt 是 `KEY=value`（set -x 追踪行以 + 开头），
    原实现只解析 `:` → 环境版本恒为空。现兼容 `=` 与 `:`，过滤 +/note 行。
    """
    vf = os.path.join(os.path.dirname(date_dir), "versions.txt")
    if not os.path.exists(vf):
        return {}
    out = {}
    for line in open(vf, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("+") or line.startswith("note"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        k = k.strip()
        if k and not k.startswith("#"):
            out[k] = v.strip()
    return out


def parse_ut(path):
    if not os.path.exists(path):
        return {"ok": False, "text": "UT 日志不存在"}
    content = open(path, encoding="utf-8", errors="ignore").read()
    failed_cases = re.findall(r"^FAILED\s+(.*)$", content, re.MULTILINE)
    f_prog = 0
    for line in content.splitlines():
        if line.strip().startswith("unit_tests/") or re.match(r"^[.FE s]+$", line.strip()):
            f_prog += line.count("F") + line.count("E")
    summaries = re.findall(
        r"(\d+)\s+failed\b.*?(\d+)\s+passed\b|(\d+)\s+passed\b.*?(\d+)\s+failed\b|(\d+)\s+passed\b|(\d+)\s+failed\b",
        content,
    )
    note = ""
    if summaries:
        last = summaries[-1]
        failed = last[0] or last[3] or last[5] or "0"
        passed = last[1] or last[2] or last[4] or "0"
    else:
        passed = failed = "?"
        note = "（首轮 pytest 在 test_sinkhorn 挂起被中断，无汇总行；失败数按进度字符统计）"
        failed = str(f_prog)
    skipped = re.findall(r"(\d+)\s+skipped\b", content)
    skipped = skipped[-1] if skipped else "0"
    runs = content.count("test session starts")
    files = Counter(re.split(r"::", cc)[0] for cc in failed_cases)
    failed_files = [{"file": f, "count": c} for f, c in files.most_common()]
    dist = Counter(owner_of_ut_case(cc) for cc in failed_cases)
    return {"ok": True, "passed": str(passed), "failed": str(failed), "skipped": str(skipped),
            "runs": runs, "note": note, "failed_cases": failed_cases,
            "failed_files": failed_files, "owner_dist": dict(dist),
            "timed_out": ("SINKHORN_TIMED_OUT" in content or "SINKHORN_RC=124" in content)}


def op_log_path(date_dir, kind, case):
    return os.path.join(date_dir, "atk_logs", f"{kind}_{case}.log")


def classify_fail(op_log):
    if not op_log or not os.path.exists(op_log):
        return "无日志", []
    c = open(op_log, encoding="utf-8", errors="ignore").read()
    lines_detail = []
    m = re.search(r"Total Task:\s*(\d+), success (\d+), failed (\d+)", c)
    if m:
        tot, suc, fl = m.group(1), m.group(2), m.group(3)
        lines_detail.append(f"Total Task: {tot}, success {suc}, failed {fl}")
        base = f"{suc}/{tot} 算例通过"
        if fl == tot and ("run opp task failed" in c or "RuntimeError" in c):
            return f"运行期异常（{suc}/{tot} 算例）", lines_detail + ["含 run opp task failed / RuntimeError"]
        if "float16" in c:
            base += ", 含 fp16 失败"
            lines_detail.append("日志含 float16 基准/用例（fp16 分支受影响）")
        return base, lines_detail
    if "run opp task failed" in c or "RuntimeError" in c:
        return "运行期异常", ["含 run opp task failed / RuntimeError"]
    return "未知", ["见算子日志"]


def parse_atk(path, label, kind, op_timing):
    if not os.path.exists(path):
        return {"ok": False, "text": f"{label} 日志不存在"}
    content = open(path, encoding="utf-8", errors="ignore").read()
    pass_count = fail_count = noresult = 0
    failed, noresult_ops = [], []
    for line in content.splitlines():
        line = line.strip()
        if re.search(r"\[.*\].*status:\s*Pass", line):
            pass_count += 1
        elif re.search(r"\[.*\].*status:\s*Failed", line):
            fail_count += 1
            m = re.search(r"\[(.*?)\]", line)
            if m:
                case = m.group(1)
                reason, detail = classify_fail(op_log_path(os.path.dirname(path), kind, case))
                failed.append({"op": case, "owner": OWNER_MAP.get(case, "未知"),
                               "reason": reason, "detail": detail,
                               "seconds": op_timing.get(case)})
        elif "NO_RESULT" in line or re.search(r"status:\s*None", line):
            noresult += 1
            m = re.search(r"\[(.*?)\]", line)
            if m:
                case = m.group(1)
                noresult_ops.append({"op": case, "owner": OWNER_MAP.get(case, "未知"),
                                     "reason": "首轮无基准（无 xlsx 对比）" if kind == "perf" else "无结果",
                                     "seconds": op_timing.get(case)})
    return {"ok": True, "passed": pass_count, "failed": fail_count, "noresult": noresult,
            "failed_ops": failed, "noresult_ops": noresult_ops}


def owner_of_ut_case(cc):
    path = re.split(r"::", cc)[0].replace("unit_tests/", "").replace(".py", "")
    base = os.path.basename(path)
    if base.startswith("test_"):
        return OWNER_MAP.get(base[len("test_"):], "未知")
    return OWNER_MAP.get(base, "未知")


def build_by_owner(acc, perf, ut):
    by = defaultdict(list)
    for f in acc.get("failed_ops") or []:
        by[f["owner"]].append({"kind": "accuracy", "op": f["op"], "reason": f["reason"],
                               "seconds": f.get("seconds")})
    for f in perf.get("failed_ops") or []:
        by[f["owner"]].append({"kind": "performance", "op": f["op"], "reason": f["reason"],
                               "seconds": f.get("seconds")})
    # 2026-09-07: 无结果项不再写入报告（负责人聚合 / 摘要 / 速览均不体现）
    for ff in ut.get("failed_files") or []:
        by[owner_of_ut_case(ff["file"] + "::x")].append(
            {"kind": "ut", "op": ff["file"], "reason": f"{ff['count']} 条（明细见 result_ut.log）"})
    return [{"owner": k, "items": v} for k, v in sorted(by.items())]


def main():
    date_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    guard_type = sys.argv[2] if len(sys.argv) > 2 else "daily"
    container = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    acc_file = "result_atk.log"
    perf_file = "result_atk_perf.log"

    wall, op_timing = read_timing(date_dir)
    ut = parse_ut(os.path.join(date_dir, "result_ut.log"))
    acc = parse_atk(os.path.join(date_dir, acc_file), "ATK 精度", "acc", op_timing["acc"])
    perf = parse_atk(os.path.join(date_dir, perf_file), "ATK 性能", "perf", op_timing["perf"])
    versions = read_env_versions(date_dir)
    by_owner = build_by_owner(acc, perf, ut)
    ut_failed = ut.get("failed_cases") or []
    ut_dist = Counter(ut.get("owner_dist") or {})

    dedup = []
    seeset = set()
    for kind, fl in (("accuracy", acc.get("failed_ops") or []), ("performance", perf.get("failed_ops") or [])):
        for f in fl:
            key = (f["op"], kind)
            if key not in seeset:
                seeset.add(key)
                dedup.append({**f, "kind": kind})

    # ---------- report.json ----------
    report = {
        "meta": {
            "date": os.path.basename(date_dir),
            "guard_type": "version" if guard_type == "version" else "daily",
            "container": container,
            "env_versions": versions,
            "timing_seconds": wall,
            "timing_total_seconds": sum(wall.values()),
        },
        "ut": {"passed": ut.get("passed"), "failed": ut.get("failed"),
               "skipped": ut.get("skipped"), "failed_cases": ut_failed,
               "failed_files": ut.get("failed_files") or [], "owner_dist": ut_dist},
        "atk_accuracy": {"passed": acc.get("passed"), "failed": acc.get("failed"),
                         "noresult": acc.get("noresult", 0),
                         "failed_ops": acc.get("failed_ops") or [],
                         "noresult_ops": acc.get("noresult_ops") or []},
        "atk_performance": {"passed": perf.get("passed"), "failed": perf.get("failed"),
                            "noresult": perf.get("noresult", 0),
                            "failed_ops": perf.get("failed_ops") or [],
                            "noresult_ops": perf.get("noresult_ops") or []},
        "by_owner": by_owner,
        "report_files": {
            "report": f"{date_dir}/report.md",
            "logs": [f"{date_dir}/result_ut.log", f"{date_dir}/{acc_file}", f"{date_dir}/{perf_file}",
                     f"{date_dir}/atk_logs/"],
            "perf_baseline": "/mnt/share/ops_nightly_agent/triton/",
        },
    }
    with open(os.path.join(date_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- report.md ----------
    guard_label = "版本看护（全量环境更新）" if guard_type == "version" else "日常看护（仅更新 MindSpeed-Ops）"
    md = [f"# 每日看护测试报告（{os.path.basename(date_dir)} · {guard_label}）", ""]

    md += ["## 一、结果摘要", ""]
    md += ["### UT", "",
           f"- 通过 {ut['passed']} · 失败 {ut['failed']} · 跳过 {ut['skipped']}"]
    if ut["runs"] > 1:
        md.append("- （含 2 段运行——首轮在 test_sinkhorn 挂起被中断，补跑合并）")
    if ut["note"]:
        md.append(f"- {ut['note']}")
    if ut_failed:
        md.append("- 失败负责人分布: " + ", ".join(f"{k} {v}" for k, v in ut_dist.most_common()))
    md += ["", "### ATK 精度", "",
           f"- 通过 {acc['passed']} · 失败 {acc['failed']}"]
    if acc["failed_ops"]:
        md.append("- 失败算子: " + "; ".join(f"{f['op']}@{f['owner']}({f['reason']})" for f in acc["failed_ops"]))
    md += ["", "### ATK 性能", "",
           f"- 通过 {perf['passed']} · 失败 {perf['failed']}"]
    if perf["failed_ops"]:
        md.append("- 失败算子: " + "; ".join(f"{f['op']}@{f['owner']}({f['reason']})" for f in perf["failed_ops"]))
    md += ["", "### 失败项与负责人（速览）", ""]
    if not dedup and not ut_failed:
        md.append("- 无失败项 ✅")
    else:
        for f in dedup:
            md.append(f"- [{f['kind']}] {f['op']} @{f['owner']} — {f['reason']}")
        for ff in ut.get("failed_files") or []:
            md.append(f"- [ut] {ff['file']} 共 {ff['count']} 条 @{owner_of_ut_case(ff['file'] + '::x')}")

    md += ["", "## 二、元信息", ""]
    md += [f"- 容器: {container}（A3 Ascend910） · 看护路径: {guard_label}"]
    if versions:
        md.append("- 环境: " + " · ".join(f"{k} {v}" for k, v in versions.items()))
    md.append("- 阶段耗时: " + (" · ".join(f"{k} {fmt_dur(v)}" for k, v in sorted(wall.items())) or "未记录"))
    if wall:
        md.append(f"- 总耗时: {fmt_dur(sum(wall.values()))}")

    md += ["", "## 三、失败项详情", ""]
    for f in dedup:
        dur = f"，耗时 {fmt_dur(f['seconds'])}" if f.get("seconds") else ""
        md += [f"### [{f['kind']}] {f['op']} — 负责人: {f['owner']}", "",
               f"- 原因: {f['reason']}{dur}",
               f"- 详情: " + "；".join(f.get("detail") or []) if f.get("detail") else "- 详情: 见算子日志",
               f"- 算子日志: `atk_logs/{f['kind'][:3]}_{f['op']}.log`", ""]
    if ut_failed:
        md += ["### [ut] 单测失败（按文件汇总，明细见 result_ut.log）", ""]
        for ff in ut.get("failed_files") or []:
            sample = next((cc for cc in ut_failed if cc.startswith(ff["file"])), "")
            md += [f"- **{ff['file']}** — {ff['count']} 条 @{owner_of_ut_case(ff['file'] + '::x')}",
                   f"- 样例: {sample}" if sample else "", ""]
        md.append(f"- 全部 {len(ut_failed)} 条 FAILED 明细见 `result_ut.log`")

    md += ["", "## 四、按负责人聚合", ""]
    if not by_owner:
        md.append("无失败项 ✅")
    else:
        for o in by_owner:
            md.append(f"- **{o['owner']}（{len(o['items'])} 项）**")
            for it in o["items"]:
                dur = f"，耗时 {fmt_dur(it['seconds'])}" if it.get("seconds") else ""
                md.append(f"  - [{it['kind']}] {it['op']} — {it['reason']}{dur}")

    md += ["", "## 五、报告与归档位置", "",
           f"- 本报告: `{date_dir}/report.md`、`{date_dir}/report.json`",
           f"- 原始日志: `{date_dir}/result_ut.log`、`{date_dir}/{acc_file}`、`{date_dir}/{perf_file}`、`{date_dir}/atk_logs/`",
           f"- 性能基准 xlsx: `/mnt/share/ops_nightly_agent/triton/`", ""]
    with open(os.path.join(date_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- console ----------
    print("=" * 44)
    print(f"UT: 通过 {ut['passed']} / 失败 {ut['failed']} / 跳过 {ut['skipped']}")
    for ff in ut.get("failed_files") or []:
        fname = os.path.basename(ff["file"]).replace(".py", "")
        print(f"失败用例：{fname} @{owner_of_ut_case(ff['file'] + '::x')}")
    print(f"ATK 精度: 通过 {acc['passed']} / 失败 {acc['failed']}")
    for f in acc.get("failed_ops") or []:
        print(f"失败用例：{f['op']} @{f['owner']}")
    print(f"ATK 性能: 通过 {perf['passed']} / 失败 {perf['failed']}")
    for f in perf.get("failed_ops") or []:
        print(f"失败用例：{f['op']} @{f['owner']}")
    print("=" * 44)
    print(f"报告已写入: {date_dir}/report.md, report.json")


if __name__ == "__main__":
    main()
