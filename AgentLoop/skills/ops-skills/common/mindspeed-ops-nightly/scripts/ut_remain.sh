#!/bin/bash
# UT remainder — 补跑挂起文件 test_sinkhorn.py（triton + tilelang 两个）, 各 900s 限时。
# 2026-08-29 改造: nightly_ut.sh 已 N 卡分桶并行覆盖全部非挂起 test_*.py（含原 PART1 三文件），
# 故本脚本只负责两个 sinkhorn；结果仅追加 SINKHORN 标记行到 result_ut.log，
# 不拼 pytest 数字行 → summarize.py summaries[-1] 仍取 TOTAL 汇总行（防污染）。
set -x
WS=/mnt/share/ops_nightly_agent
DATE="${1:-${DATE:-$WS/$(date +%Y%m%d)}}"
source $WS/env.sh
export SOC_VERSION=ascend910_93
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"

cd $WS/MindSpeed-Ops/tests || exit 1
UTREM=$DATE/ut_remain.log
: > $UTREM

echo "###### SINKHORN REMAINDER (timeout 900s each) ######" >> $UTREM
timeout 900 pytest unit_tests/triton/test_sinkhorn.py -q >> $UTREM 2>&1
RC1=${PIPESTATUS[0]}
echo "triton_sinkhorn_rc=$RC1" >> $UTREM
timeout 900 pytest unit_tests/tilelang/test_sinkhorn.py -q >> $UTREM 2>&1
RC2=${PIPESTATUS[0]}
echo "tilelang_sinkhorn_rc=$RC2" >> $UTREM

# 只追加标记行（无数字行；"SINKHORN_TIMED_OUT"/"SINKHORN_OK" 与 summarize.py 检测兼容）
M1="SINKHORN_TIMED_OUT_TRITON";  [ $RC1 -eq 0 ] && M1="SINKHORN_OK_TRITON";  [ $RC1 -gt 124 ] && M1="SINKHORN_RC=${RC1}_TRITON"
M2="SINKHORN_TIMED_OUT_TILELANG"; [ $RC2 -eq 0 ] && M2="SINKHORN_OK_TILELANG"; [ $RC2 -gt 124 ] && M2="SINKHORN_RC=${RC2}_TILELANG"
echo "$M1" >> $DATE/result_ut.log
echo "$M2" >> $DATE/result_ut.log
echo UT_REMAINDER_DONE
