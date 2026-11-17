#!/bin/bash
# pack_logs.sh — 日常/版本看护结束后收集本轮运行日志, 压缩为 {日期}-log.zip 存于产物目录。
# 用法: bash pack_logs.sh [OUT_DIR]   （OUT_DIR 默认 $WS/$(date +%Y%m%d); run_all 自动传 $OUT）
# 包含: UT（ut_part*.log / result_ut.log / ut_remain.log）,
#       ATK 精度（result_atk.log / atk_logs/acc_*.log）,
#       ATK 性能（result_atk_perf.log / atk_logs/perf_*.log）,
#       附 timing.txt / op_timing_{acc,perf}.txt。
# zip 内顶层目录 = {日期}, 解压即得 {日期}/atk_logs/... 结构。
# v0.7（2026-09-01）：run_all 新增 pack 阶段调用本脚本（成功才标记 .phase, 失败下次重跑重试）。
set -u
WS="${WS:-/mnt/share/ops_nightly_agent}"
[ -f "$WS/env.sh" ] && source "$WS/env.sh" >/dev/null 2>&1 || true
OUT="${1:-$WS/$(date +%Y%m%d)}"
DATE_DIR=$(basename "$OUT")
ZIP="$OUT/${DATE_DIR}-log.zip"

[ -d "$OUT" ] || { echo "[pack] 产物目录不存在: $OUT"; exit 1; }

python - "$OUT" "$ZIP" "$DATE_DIR" <<'PYEOF'
import os, sys, zipfile
out_dir, zip_path, top = sys.argv[1], sys.argv[2], sys.argv[3]

def want(rel):
    base = os.path.basename(rel)
    if rel.startswith("atk_logs/") and rel.endswith(".log"):
        return True
    if base in ("result_ut.log", "result_atk.log", "result_atk_perf.log", "ut_remain.log"):
        return True
    if base.startswith("ut_part") and base.endswith(".log"):
        return True
    if base in ("timing.txt", "op_timing_acc.txt", "op_timing_perf.txt"):
        return True
    return False

z = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
n = 0
for dirpath, dirnames, filenames in os.walk(out_dir):
    dirnames[:] = [d for d in dirnames if d != "result"]
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, out_dir).replace("\\", "/")
        if os.path.abspath(full) == os.path.abspath(zip_path):
            continue
        if want(rel):
            z.write(full, os.path.join(top, rel))
            n += 1
z.close()
if n == 0:
    os.remove(zip_path)
    print("[pack] 无日志可打包（0 个文件）")
    sys.exit(2)
print("[pack] %d 个日志文件 -> %s" % (n, zip_path))
PYEOF
rc=$?
[ "$rc" -ne 0 ] && exit "$rc"
echo "[pack] zip 大小: $(ls -lh "$ZIP" | awk '{print $5}')"
echo PACK_LOGS_DONE
