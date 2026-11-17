#!/bin/bash
# Step4: UT full run — N 卡并行分桶（2026-08-29 占用率改造; 2026-09-09 卡数参数化）
# 默认 6 卡: NCARDS 环境变量可覆盖（服务器方式按实际可用逻辑卡数传入, 如 NCARDS=16）。
# 排除挂起文件 test_sinkhorn.py（两个）, 其余 test_*.py 字典序轮转分 NCARDS 桶,
# 每个 pytest 进程绑逻辑卡 i（ASCEND_RT_VISIBLE_DEVICES=$i）, 结果合并 → result_ut.log
# （最后一行合成总汇总, 兼容 summarize.py summaries[-1] 解析）。sinkhorn 由 ut_remain.sh 补跑。
set -x
WS=/mnt/share/ops_nightly_agent
DATE="${1:-${DATE:-$WS/$(date +%Y%m%d)}}"
source $WS/env.sh
export SOC_VERSION=ascend910_93
# 2026-08-28 修复: editable finder 不再挂仓库根, 文件模式 pytest 需 PYTHONPATH
export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"

T0=$(date +%s)
cd $WS/MindSpeed-Ops/tests || exit 1
mkdir -p $DATE
echo "=======start ut test (NCARDS=$NCARDS parallel buckets)======="

# 收集测试文件（排除两个挂起 sinkhorn），字典序轮转分 NCARDS 桶
mapfile -t FILES < <(find unit_tests -name 'test_*.py' | grep -v '/test_sinkhorn.py$' | sort)
NCARDS="${NCARDS:-6}"
declare -a BUCKETS
for i in $(seq 0 $((NCARDS-1))); do BUCKETS[$i]=""; done
idx=0
for f in "${FILES[@]}"; do
  BUCKETS[$((idx % NCARDS))]+=" $f"
  idx=$((idx+1))
done

pids=()
for i in $(seq 0 $((NCARDS-1))); do
  [ -n "${BUCKETS[$i]// }" ] || continue
  (
    export ASCEND_RT_VISIBLE_DEVICES=$i
    pytest ${BUCKETS[$i]} -p no:cacheprovider > "$DATE/ut_part$i.log" 2>&1
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# 合并 6 桶 → result_ut.log（单一真源, 兼容 summarize summaries[-1]）
python - "$DATE" "$NCARDS" <<'PYEOF'
import os, re, sys
date_dir, n = sys.argv[1], int(sys.argv[2])
tot_f = tot_p = tot_s = 0
failed, seen, per_part = [], set(), []
for i in range(n):
    p = os.path.join(date_dir, "ut_part%d.log" % i)
    if not os.path.exists(p):
        continue
    c = open(p, encoding="utf-8", errors="ignore").read()
    for cc in re.findall(r"^FAILED\s+(.*)$", c, re.MULTILINE):
        if cc not in seen:
            seen.add(cc)
            failed.append(cc)
    def lastnum(pat):
        ms = re.findall(pat, c)
        return int(ms[-1]) if ms else 0
    f_ = lastnum(r"(\d+)\s+failed\b")
    p_ = lastnum(r"(\d+)\s+passed\b")
    s_ = lastnum(r"(\d+)\s+skipped\b")
    if f_ or p_ or s_:
        per_part.append("part%d: %d failed, %d passed, %d skipped" % (i, f_, p_, s_))
        tot_f += f_; tot_p += p_; tot_s += s_
out = ["###### UT PARALLEL (%d cards, sinkhorn excluded) ######" % n]
out += per_part
out.append("###### FAILED CASES (merged) ######")
out += ["FAILED " + cc for cc in failed]
out.append("===== TOTAL: %d failed, %d passed, %d skipped (%d parallel parts) =====" % (tot_f, tot_p, tot_s, n))
open(os.path.join(date_dir, "result_ut.log"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("UT_MERGED total_failed=%d total_passed=%d total_skipped=%d" % (tot_f, tot_p, tot_s))
PYEOF

echo "=======finish ut test======="
echo "ut $(( $(date +%s) - T0 ))" >> $DATE/timing.txt
echo UT_PHASE_DONE
