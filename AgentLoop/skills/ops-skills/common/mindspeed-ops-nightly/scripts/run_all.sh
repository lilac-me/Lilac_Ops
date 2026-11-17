#!/usr/bin/env bash
# run_all.sh — skill master 编排：断点续跑（$OUT/.phase），容器随时可能释放
# 用法: MODE=daily|version PKG_DIR=<目录(version 必需)> CONTAINER=<ip> bash run_all.sh [DATE]
#       MODE 默认 daily；CONTAINER 默认 unknown（写入报告元信息）
# 阶段: init → install → ut → acc → perf → summarize；每阶段产物/日志实时落 NFS，
#       换容器后重跑本脚本自动跳过已完成阶段（.phase 标记）。
# 2026-08-29 占用率改造: 每阶段 wait 期间心跳循环写 $OUT/heartbeat（每 30s, 观测低占用时段）;
#       ut 阶段 = nightly_ut.sh（N 卡分桶并行, 排除挂起 sinkhorn）+ ut_remain.sh（sinkhorn 900s 限时补跑）。
# 2026-09-09 双模式（容器/服务器）: 卡资源统一参数化 —
#       容器方式默认 6 卡（NCARDS=6, ASCEND_RT_VISIBLE_DEVICES=0..5）;
#       服务器方式（普通用户+sudo 登录带卡服务器）按实际可用逻辑卡覆盖, 如
#       NCARDS=16 ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 bash run_all.sh [DATE]。
set -u
WS="${WS:-/mnt/share/ops_nightly_agent}"
ENVSH="$WS/env.sh"
MODE="${MODE:-daily}"
CONTAINER="${CONTAINER:-unknown}"
DATE="${1:-$(date +%Y%m%d)}"
export DATE   # 透传给 nightly_*.sh / ut_remain.sh（它们读 $DATE 或 $1）
NCARDS="${NCARDS:-6}"
export NCARDS                       # 透传 nightly_ut/nightly_atk_*/run_retest 分桶与并发池
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
OUT="$WS/$DATE"
LOG="$WS/logs/run_all_${DATE}_$(date +%H%M%S).log"
mkdir -p "$OUT/atk_logs" "$WS/logs"

mark() { echo "$1" > "$OUT/.phase"; }

# --- 续跑语义（2026-09-07 修复）: 按固定阶段序, 只运行 ".phase 已完成阶段"之后的阶段 ---
# 原实现 phase() 只跳过"等于当前值"的阶段, 导致 .phase=acc 时 init/install/ut/acc 全部重跑
# （113→127 换容器续跑实测: 每阶段重跑又耗 3h+）。现按序号: CUR 阶段及更早全部跳过。
PHASE_ORDER="init install ut acc perf summarize pack"
CUR_PHASE=$(cat "$OUT/.phase" 2>/dev/null || echo none)
SKIP_TILL=-1
if [ "$CUR_PHASE" != "none" ]; then
  _i=0
  for _p in $PHASE_ORDER; do
    [ "$_p" = "$CUR_PHASE" ] && SKIP_TILL=$_i
    _i=$((_i + 1))
  done
fi
need_phase() {   # $1=阶段名; 返回 0=需要运行, 1=已完成跳过
  local _i=0 _p
  for _p in $PHASE_ORDER; do
    [ "$_p" = "$1" ] && break
    _i=$((_i + 1))
  done
  [ "$_i" -gt "$SKIP_TILL" ]
}

# 阶段运行 + 心跳: 并行起心跳循环（每 30s 追加时间戳+阶段到 heartbeat），阶段结束 kill 心跳
run_stage() {
  local name="$1" script="$2"; shift 2
  bash "$script" "$@" </dev/null >> "$LOG" 2>&1 &
  local pid=$!
  ( while kill -0 "$pid" 2>/dev/null; do echo "$(date +%s) $name" >> "$OUT/heartbeat"; sleep 30; done ) &
  local hb=$!
  wait "$pid"
  local rc=$?
  kill "$hb" 2>/dev/null || true
  return $rc
}

echo "[run_all] MODE=$MODE CONTAINER=$CONTAINER DATE=$DATE NCARDS=$NCARDS VISIBLE=$ASCEND_RT_VISIBLE_DEVICES LOG=$LOG" | tee -a "$LOG"

# --- init: 容器级初始化（换容器必跑；但已被 .phase 标记完成的轮次跳过——bootstrap 由手工/新容器首跑执行） ---
if need_phase init; then
  echo "== phase: init ==" | tee -a "$LOG"
  bash "$WS/skill/scripts/bootstrap.sh" >> "$LOG" 2>&1 || { echo "bootstrap 失败"; exit 1; }
  mark init
fi

# --- install: 按模式更新 ---
if need_phase install; then
  echo "== phase: install (mode=$MODE) ==" | tee -a "$LOG"
  if [ "$MODE" = "version" ]; then
    PKG_DIR="${PKG_DIR:?version 模式必须提供 PKG_DIR（B 版本配套包目录）}" \
      bash "$WS/skill/scripts/mode_version.sh" >> "$LOG" 2>&1 || { echo "mode_version 失败"; exit 1; }
  else
    bash "$WS/skill/scripts/mode_daily.sh" >> "$LOG" 2>&1 || { echo "mode_daily 失败"; exit 1; }
  fi
  mark install
fi

# --- ut: 6 卡分桶并行主跑（排除挂起 sinkhorn）→ sinkhorn 900s 限时补跑 ---
if need_phase ut; then
  echo "== phase: ut ==" | tee -a "$LOG"
  run_stage ut "$WS/scripts/nightly_ut.sh" "$OUT" || true
  run_stage ut_remain "$WS/scripts/ut_remain.sh" "$OUT" || true
  mark ut
fi

# --- acc: ATK 精度（统一 --max_task 10 -to 1200, 6 卡并行池, 慢 op 先入池） ---
if need_phase acc; then
  echo "== phase: acc ==" | tee -a "$LOG"
  run_stage acc "$WS/scripts/nightly_atk_acc.sh" "$OUT" || true
  mark acc
fi

# --- perf: ATK 性能（生成/对比分支, 6 卡并行池） ---
if need_phase perf; then
  echo "== phase: perf ==" | tee -a "$LOG"
  run_stage perf "$WS/scripts/nightly_atk_perf.sh" "$OUT" || true
  mark perf
fi

# --- summarize: v5 报告 ---
if need_phase summarize; then
  echo "== phase: summarize ==" | tee -a "$LOG"
  # 2026-09-07 修复: run_stage 用 bash 包裹执行, 不能直接跑 ELF 二进制 → 改为 source env.sh 后调 python
  ( source "$ENVSH" >/dev/null 2>&1; exec python "$WS/scripts/summarize.py" "$OUT" "$MODE" "$CONTAINER" ) >> "$LOG" 2>&1 \
    || { echo "summarize 失败" | tee -a "$LOG"; exit 1; }
  mark summarize
fi

# --- pack: 打包本轮 UT/ATK 精度/ATK 性能运行日志 → $OUT/${DATE}-log.zip（成功才标记, 失败下次重跑重试） ---
if need_phase pack; then
  echo "== phase: pack ==" | tee -a "$LOG"
  # 2026-09-07 修复: run_stage 内部已 bash 包裹, 不再嵌套传 bash
  if run_stage pack "$WS/scripts/pack_logs.sh" "$OUT"; then
    mark pack
  else
    echo "[run_all] pack 失败（未标记, 下次重跑会自动重试）" | tee -a "$LOG"
  fi
fi

echo "[run_all] ALL DONE — 报告: $OUT/report.md, report.json；日志包: $OUT/${DATE}-log.zip（总耗时见 report.md 元信息）" | tee -a "$LOG"
