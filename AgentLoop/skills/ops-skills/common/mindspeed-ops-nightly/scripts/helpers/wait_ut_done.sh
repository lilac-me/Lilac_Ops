#!/bin/bash
# 等待 ut 阶段结束（nightly_ut.sh / ut_remain.sh 进程消失且 .phase 推进）。
# 用法: bash wait_ut_done.sh <OUT_DIR>
#   例: bash wait_ut_done.sh /mnt/share/ops_nightly_agent/<DATE>
OUT="${1:?OUT_DIR 必填}"
while :; do
  P=$(ps aux | grep -cE '[n]ightly_ut\.sh|[u]t_remain\.sh')
  PH=$(cat "$OUT/.phase" 2>/dev/null || echo none)
  if [ "$PH" = "acc" ] || [ "$PH" = "perf" ] || [ "$PH" = "summarize" ] || { [ "$PH" = "install" ] && [ "$P" -eq 0 ]; }; then
    echo "UT_DONE phase=$PH procs=$P"; break
  fi
  sleep 30
done
echo "====RESULT_UT===="
tail -8 "$OUT/result_ut.log" 2>/dev/null
