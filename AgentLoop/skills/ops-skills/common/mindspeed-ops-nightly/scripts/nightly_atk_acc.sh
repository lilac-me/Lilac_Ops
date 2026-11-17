#!/bin/bash
# Step5a: ATK accuracy, 26 ops, N-card parallel pool (2026-08-29 占用率改造; 2026-09-09 卡数参数化).
# 原: 26 op 全并发 i%8 绑 8 卡 → 快 op 先完, 尾部只剩慢 op 在少数卡, 占用率 <5% 被释放。
# 现在: ① ASCEND_RT_VISIBLE_DEVICES 可覆盖（默认 0..5, 6 卡）; ② 慢 op 按历史耗时降序先入池;
#       ③ NCARDS 并发作业池（默认 6, 服务器方式按实际可传 NCARDS=16 等）→ 全程满载。
# atk 命令/参数与用户脚本镜像一致 (唯一差异仍: 全 op --max_task 10 -to 1200)。
set -x
WS=/mnt/share/ops_nightly_agent
DATE="${1:-${DATE:-$WS/$(date +%Y%m%d)}}"
source $WS/env.sh
export SOC_VERSION=ascend910_93
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
# 2026-08-28 修复: editable finder 不再挂仓库根 → atk worker 需 PYTHONPATH 才能 import tests
export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"

cd $WS/MindSpeed-Ops/tests || exit 1
mkdir -p $DATE/atk_logs

echo "=======start atk accuracy test======="
T0=$(date +%s)
TARGET_DIR="atk_tests/triton"
RESULT_SUMMARY="$DATE/result_atk.log"
> "$RESULT_SUMMARY"
LOG_DIR="$DATE/atk_logs"
mkdir -p "$LOG_DIR"
mkdir -p $WS/triton
NCARDS="${NCARDS:-6}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR! TARGET_DIR not exists"; exit 1
fi

# 收集有效 op 目录（yaml + generate + triton 三件套齐全）
mapfile -t dirs < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d)
NAMES=()
for dir_path in "${dirs[@]}"; do
  dir_name=$(basename "$dir_path")
  yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
  py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
  triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"
  if [ -f "$yaml_file" ] && [ -f "$py_script" ] && [ -f "$triton_file" ]; then
    NAMES+=("$dir_name")
  fi
done

# 慢 op 排序: 读最近一轮的 op_timing_acc.txt（$WS/20*/ 下最新修改）, 秒数降序（慢的先入池）,
# 无历史记录的（本轮新增 op）按 0 秒排最后, 保证尾部无长单。
TIMING_REF=""
for f in $WS/20*/op_timing_acc.txt; do
  if [ -f "$f" ] && { [ -z "$TIMING_REF" ] || [ "$f" -nt "$TIMING_REF" ]; }; then TIMING_REF="$f"; fi
done
declare -A OP_SEC
if [ -n "$TIMING_REF" ]; then
  while read -r _ op sec; do [ -n "$op" ] && [ "$sec" -eq "$sec" ] 2>/dev/null && OP_SEC[$op]=$sec; done < "$TIMING_REF"
fi
mapfile -t SORTED < <(for n in "${NAMES[@]}"; do echo "${OP_SEC[$n]:-0} $n"; done | sort -k1,1 -rn | awk '{print $2}')

# 6 并发作业池: 满 6 个活跃就 wait -n 等一个完成
pids=()
for idx in "${!SORTED[@]}"; do
  dir_name="${SORTED[$idx]}"
  LOG_FILE="${LOG_DIR}/acc_${dir_name}.log"
  yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
  py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
  triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"

  # 并发池空位等待: 过滤已结束 pid, 活跃 >=6 则 wait -n 等任一完成
  while :; do
    ALIVE=0; NEW=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { ALIVE=$((ALIVE+1)); NEW+=("$p"); }; done
    pids=("${NEW[@]}")
    [ "$ALIVE" -lt "$NCARDS" ] && break
    wait -n 2>/dev/null || sleep 2
  done

  device_id=$((idx % NCARDS))
  rm -rf result/${dir_name}
  if ! atk case -f "$yaml_file" -p "$py_script" > /dev/null 2>&1; then
    echo "[$dir_name] case gen FAILED" >> "$RESULT_SUMMARY"
    continue
  fi

  # unified params per user request; per-op timing recorded
  (
    t_op=$(date +%s)
    atk node --backend triton --devices "$device_id" node --backend cpu --devices "$device_id" \
      task --max_task 10 -c result/${dir_name}/json/all_${dir_name}.json --task accuracy \
      -tup ./ -p "$triton_file" -to 1200 > "$LOG_FILE" 2>&1
    echo "acc $dir_name $(( $(date +%s) - t_op ))" >> "$DATE/op_timing_acc.txt"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null
done

# 结果提取（按 find 原始顺序, case gen FAILED 的 op 无日志 → NO_RESULT）
for dir_path in "${dirs[@]}"; do
  dir_name=$(basename "$dir_path")
  LOG_FILE="${LOG_DIR}/acc_${dir_name}.log"
  yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
  py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
  triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"
  if [ ! -f "$yaml_file" ] || [ ! -f "$py_script" ] || [ ! -f "$triton_file" ]; then
    continue
  fi
  STATUS=$(grep -oP 'acc_pass_result:\K[^\s]+' "$LOG_FILE" | tail -1)
  if [ -z "$STATUS" ]; then STATUS="NO_RESULT"; fi
  echo "[$dir_name] status: $STATUS" >> "$RESULT_SUMMARY"
done
echo "acc $(( $(date +%s) - T0 ))" >> "$DATE/timing.txt"
echo "finish all acc task"
echo ATK_ACC_PHASE_DONE
