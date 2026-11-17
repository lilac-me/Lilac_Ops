#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 report.json 结构并打印关键字段（agent 取结果时核对用）。
用法: python check_report_json.py <report.json路径>
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else ""
if not path:
    print("[check] 用法: python check_report_json.py <report.json路径>")
    sys.exit(2)
with open(path, encoding="utf-8") as f:
    r = json.load(f)
print("meta:", json.dumps(r["meta"], ensure_ascii=False))
print("ut:", r["ut"]["passed"], r["ut"]["failed"], r["ut"]["skipped"])
print("acc:", r["atk_accuracy"]["passed"], r["atk_accuracy"]["failed"], r["atk_accuracy"]["noresult"])
print("perf:", r["atk_performance"]["passed"], r["atk_performance"]["failed"], r["atk_performance"]["noresult"])
print("acc_failed_ops:", [(f["op"], f["owner"]) for f in r["atk_accuracy"]["failed_ops"]])
print("perf_failed_ops:", [(f["op"], f["owner"]) for f in r["atk_performance"]["failed_ops"]])
print("perf_noresult_ops:", [(f["op"], f["owner"]) for f in r["atk_performance"]["noresult_ops"]])
print("by_owner:", [(o["owner"], len(o["items"])) for o in r["by_owner"]])
print("[check] report.json OK")
