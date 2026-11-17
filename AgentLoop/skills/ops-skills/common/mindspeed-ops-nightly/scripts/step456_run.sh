#!/bin/bash
# Master runner: [update] Step4 (UT) -> Step5a (ATK acc) -> Step5b (ATK perf) -> Step6 (summarize)
# usage: bash step456_run.sh [daily|version] [CONTAINER_IP]
set -x
WS=/mnt/share/ops_nightly_agent
GUARD=${1:-daily}
CONTAINER=${2:-unknown}
mkdir -p $WS/logs
LOG=$WS/logs/step456_$(date +%Y%m%d_%H%M%S).log
exec > "$LOG" 2>&1

source $WS/env.sh
export SOC_VERSION=ascend910_93
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATE=$WS/$(date +%Y%m%d)
mkdir -p "$DATE/atk_logs" "$WS/triton"

echo "############ UPDATE: $GUARD ############"
if [ "$GUARD" = "version" ]; then
  bash $WS/scripts/version_update_all.sh || { echo VERSION_UPDATE_FAILED; }
else
  bash $WS/scripts/daily_update_mindspeed.sh || { echo DAILY_UPDATE_FAILED; }
fi

echo "############ STEP4: UT ############"
bash $WS/scripts/nightly_ut.sh || { echo UT_FAILED; }
echo "############ STEP5a: ATK accuracy ############"
bash $WS/scripts/nightly_atk_acc.sh || { echo ATK_ACC_FAILED; }
echo "############ STEP5b: ATK performance ############"
bash $WS/scripts/nightly_atk_perf.sh || { echo ATK_PERF_FAILED; }
echo "############ STEP6: summarize ############"
python $WS/scripts/summarize.py "$DATE" "$GUARD" "$CONTAINER" 2>&1
echo "############ STEP7: pack logs ############"
bash $WS/scripts/pack_logs.sh "$DATE" || { echo PACK_LOGS_FAILED; }
echo "############ ALL DONE ############"
echo STEP456_ALL_DONE
