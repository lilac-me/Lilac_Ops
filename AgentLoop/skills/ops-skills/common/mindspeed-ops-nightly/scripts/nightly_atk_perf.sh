#!/bin/bash
# Step5b: ATK performance, 26 ops, 6-card parallel pool (2026-08-29 占用率改造).
# 原: 26 op 全并发 i%8 绑 8 卡 → 尾部只剩慢 op 在少数卡, 占用率 <5% 被释放。
# 现在: ① ASCEND_RT_VISIBLE_DEVICES=0..5 (6 卡); ② 慢 op 按历史耗时降序先入池;
#       ③ 6 并发作业池 (最多 6 个 op 同时跑) → 6 卡全程满载。
# xlsx 对比分支保留原脚本语义（嵌套路径 ../../triton/<op>/<op>.xlsx 永不命中 → 恒生成模式，
# 生成后 excel mv 到 $WS/triton/<op>.xlsx 扁平基准，跨天累积供下轮对比）。
set -x
WS=/mnt/share/ops_nightly_agent
DATE="${1:-${DATE:-$WS/$(date +%Y%m%d)}}"
source $WS/env.sh
export SOC_VERSION=ascend910_93
# 2026-09-07 参数化: 2 卡容器收尾时 NCARDS=2 ASCEND_RT_VISIBLE_DEVICES=0,1 覆盖; 默认 6 卡
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
# 2026-08-28 修复: editable finder 不再挂仓库根 → atk worker 需 PYTHONPATH 才能 import tests
export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"

cd $WS/MindSpeed-Ops/tests || exit 1
mkdir -p $DATE/atk_logs

echo "=======start atk performance test======="
T0=$(date +%s)
TARGET_DIR="atk_tests/triton"
RESULT_SUMMARY="$DATE/result_atk_perf.log"
> "$RESULT_SUMMARY"
LOG_DIR="$DATE/atk_logs"
mkdir -p "$LOG_DIR"
mkdir -p $WS/triton
NCARDS="${NCARDS:-6}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR! TARGET_DIR not exists"; exit 1
fi

# 收集有效 op 目录（三件套齐全）
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

# 慢 op 排序: 读最近一轮的 op_timing_perf.txt, 秒数降序先入池; 无历史按 0 排最后
TIMING_REF=""
for f in $WS/20*/op_timing_perf.txt; do
  if [ -f "$f" ] && { [ -z "$TIMING_REF" ] || [ "$f" -nt "$TIMING_REF" ]; }; then TIMING_REF="$f"; fi
done
declare -A OP_SEC
if [ -n "$TIMING_REF" ]; then
  while read -r _ op sec; do [ -n "$op" ] && [ "$sec" -eq "$sec" ] 2>/dev/null && OP_SEC[$op]=$sec; done < "$TIMING_REF"
fi
mapfile -t SORTED < <(for n in "${NAMES[@]}"; do echo "${OP_SEC[$n]:-0} $n"; done | sort -k1,1 -rn | awk '{print $2}')

# 6 并发作业池
pids=()
for idx in "${!SORTED[@]}"; do
  dir_name="${SORTED[$idx]}"
  LOG_FILE="${LOG_DIR}/perf_${dir_name}.log"
  yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
  py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
  triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"
  performance_file="../../triton/${dir_name}/${dir_name}.xlsx"

  # 2026-09-07 per-op 断点: 容器频繁释放场景下, 上轮已完成（扁平基准 xlsx 已生成 + perf 日志非空）
  # 的算子直接跳过重跑——结果提取阶段仍从既有 perf_<op>.log grep, 报告不丢算子。
  if [ -f "$WS/triton/${dir_name}.xlsx" ] && [ -s "$LOG_FILE" ]; then
    echo "perf $dir_name skip(existing baseline+log)" >> "$DATE/op_timing_perf.txt"
    continue
  fi

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

  if [ ! -f "$performance_file" ]; then
    # generate mode (first run), unified params; per-op timing recorded
    (
      t_op=$(date +%s)
      atk node --backend triton --devices "$device_id" node --backend npu --devices "$device_id" \
        task --max_task 10 -c result/${dir_name}/json/all_${dir_name}.json --task performance_device \
        -tup ./ -p "$triton_file" -to 1200 > "$LOG_FILE" 2>&1
      echo "perf $dir_name $(( $(date +%s) - t_op ))" >> "$DATE/op_timing_perf.txt"
    ) &
    pids+=($!)
  else
    # compare mode (baseline exists), unified params; per-op timing recorded
    (
      t_op=$(date +%s)
      atk node --backend triton --devices "$device_id" node --backend npu --is_compare False \
        node --backend triton --devices "$device_id" --bm_file "$performance_file" \
        task -c result/${dir_name}/json/all_${dir_name}.json --task performance_device \
        --max_task 10 -tup "${TARGET_DIR}/${dir_name}/" -p "$triton_file" -s 0 -e 3 -to 1200 > "$LOG_FILE" 2>&1
      echo "perf $dir_name $(( $(date +%s) - t_op ))" >> "$DATE/op_timing_perf.txt"
    ) &
    pids+=($!)
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null
done

# move generated excel baselines (mirror original: flat names under $WS/triton)
for dir_path in "${dirs[@]}"; do
  dir_name=$(basename "$dir_path")
  performance_file="../../triton/${dir_name}/${dir_name}.xlsx"
  if [ ! -f "$performance_file" ]; then
    find "./result/${dir_name}/excel/" -maxdepth 1 -name "${dir_name}*.xlsx" \
      -exec mv {} "../../triton/${dir_name}.xlsx" \; 2>/dev/null
  fi
done

# 结果提取（按 find 原始顺序）
for dir_path in "${dirs[@]}"; do
  dir_name=$(basename "$dir_path")
  LOG_FILE="${LOG_DIR}/perf_${dir_name}.log"
  yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
  py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
  triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"
  if [ ! -f "$yaml_file" ] || [ ! -f "$py_script" ] || [ ! -f "$triton_file" ]; then
    continue
  fi
  STATUS=$(grep -oP 'is_device_perf_pass:\K[^\s]+' "$LOG_FILE" | tail -1)
  if [ -z "$STATUS" ]; then STATUS="NO_RESULT"; fi
  echo "[$dir_name] status: $STATUS" >> "$RESULT_SUMMARY"
done
echo "perf $(( $(date +%s) - T0 ))" >> "$DATE/timing.txt"
echo "finish all perf task"
echo ATK_PERF_PHASE_DONE
