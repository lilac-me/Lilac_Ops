#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证新格式 result_ut.log（nightly_ut.sh 6 卡合并版）能被 summarize.py 正确解析。
用途: 修改合并格式后回归验证, 确保 summarize.py summaries[-1] 解析不破。
用法: python verify_ut_parse.py
"""
import re

sample = """###### UT PARALLEL (6 cards, sinkhorn excluded) ######
part0: 3 failed, 16 passed, 2 skipped
part1: 0 failed, 19 passed, 2 skipped
part2: 5 failed, 14 passed, 2 skipped
part3: 4 failed, 15 passed, 2 skipped
part4: 6 failed, 13 passed, 2 skipped
part5: 2 failed, 17 passed, 2 skipped
###### FAILED CASES (merged) ######
FAILED unit_tests/triton/test_wy_fast.py::TestWyFast::test_recompute_w_u_fwd[B1_T64_mha4-bf16]
FAILED unit_tests/triton/test_sort_chunks_by_idx.py::TestSortChunksByIdx::test_case1
===== TOTAL: 20 failed, 109 passed, 11 skipped (6 parallel parts) =====
SINKHORN_TIMED_OUT_TRITON
SINKHORN_TIMED_OUT_TILELANG
"""

content = sample
failed_cases = re.findall(r"^FAILED\s+(.*)$", content, re.MULTILINE)
summaries = re.findall(
    r"(\d+)\s+failed\b.*?(\d+)\s+passed\b|(\d+)\s+passed\b.*?(\d+)\s+failed\b|(\d+)\s+passed\b|(\d+)\s+failed\b",
    content)
print("failed_cases:", failed_cases)
print("summaries[-1]:", summaries[-1] if summaries else None)
skipped = re.findall(r"(\d+)\s+skipped\b", content)
print("skipped[-1]:", skipped[-1] if skipped else "0")
timed_out = ("SINKHORN_TIMED_OUT" in content or "SINKHORN_RC=124" in content)
print("timed_out:", timed_out)

last = summaries[-1]
failed = last[0] or last[3] or last[5] or "0"
passed = last[1] or last[2] or last[4] or "0"
print("parsed failed:", failed, "passed:", passed, "skipped:", skipped[-1])
assert failed == "20" and passed == "109" and skipped[-1] == "11" and timed_out
print("PARSE_VERIFY_OK")
