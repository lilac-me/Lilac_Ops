#!/bin/bash
# 等待 perf 阶段结束（run_retest / nightly_atk_perf 进程消失且 .phase 推进到 summarize）。
# 用法: bash wait_perf_proc.sh <OUT_DIR>
#   例: bash wait_perf_proc.sh /mnt/share/ops_nightly_agent/<DATE>_retest
OUT="${1:?OUT_DIR 必填}"
while :; do
  P=$(ps aux | grep -cE 'run_retest\.sh|nightly_atk_perf\.sh')
  PH=$(cat "$OUT/.phase" 2>/dev/null || echo none)
  if [ "$PH" = "summarize" ] || { [ "$PH" = "perf" ] && [ "$P" -eq 0 ]; }; then
    echo "PERF_DONE phase=$PH procs=$P"; break
  fi
  sleep 30
done
echo "====RESULT_ATK_PERF===="
tail -6 "$OUT/result_atk_perf.log" 2>/dev/null
echo "====HEARTBEAT_TAIL===="
tail -3 "$OUT/heartbeat" 2>/dev/null
