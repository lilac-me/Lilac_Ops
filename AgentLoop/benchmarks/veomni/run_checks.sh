#!/usr/bin/env bash
# VeOmni benchmarks 一键验证链（RFC 第 7 节静态 + Dry-run 层级）
# 用法: bash benchmarks/veomni/run_checks.sh
set -u
cd "$(dirname "$0")/../.."   # 仓库根
fail=0

echo "===== 1/4 安装依赖 ====="
python3 -m pip install -r benchmarks/veomni/requirements.txt || fail=1

echo
echo "===== 2/4 场景静态检查（run_scenarios） ====="
python3 benchmarks/veomni/run_scenarios.py || fail=1

echo
echo "===== 3/4 单元测试（unittest discover） ====="
python3 -m unittest discover -s benchmarks/veomni/tests -v || fail=1
python3 -m unittest discover -s benchmarks/veomni/quick_migration/tests -v || fail=1

echo
echo "===== 4/4 评分器冒烟（Omni 场景门禁链路） ====="
python3 benchmarks/veomni/tests/_smoke_scorer.py || fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo "全部检查通过"
else
  echo "存在失败项，请检查上方输出"
fi
exit "$fail"
