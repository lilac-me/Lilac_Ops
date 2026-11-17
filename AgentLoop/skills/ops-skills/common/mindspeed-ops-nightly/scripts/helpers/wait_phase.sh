#!/bin/bash
# 等待 OUT 目录的 .phase 不再是 <cur_phase>（阶段推进）或运行结束。
# 用法: bash wait_phase.sh <OUT_DIR> <cur_phase>
#   例: bash wait_phase.sh /mnt/share/ops_nightly_agent/<DATE> acc
# 说明: 换容器续跑/容器失联场景下, agent 用本脚本等待 .phase 推进后再取结果。
OUT="${1:?OUT_DIR 必填}"
PH="${2:?cur_phase 必填}"
while [ "$(cat "$OUT/.phase" 2>/dev/null || echo none)" = "$PH" ]; do
  sleep 30
done
echo "PHASE_CHANGED -> $(cat "$OUT/.phase" 2>/dev/null || echo none)"
echo "====HEARTBEAT_TAIL===="
tail -3 "$OUT/heartbeat" 2>/dev/null
