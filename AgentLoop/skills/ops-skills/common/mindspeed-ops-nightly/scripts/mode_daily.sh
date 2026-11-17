#!/usr/bin/env bash
# mode_daily.sh — 日常看护：只更新 mindspeed-ops（卸载旧包 → 拉 master 最新 → 源码安装），
#                 其余环境不变。合并自 daily_update_mindspeed.sh（2026-08-26 全链路验证逻辑）。
set -e
WS="${WS:-/mnt/share/ops_nightly_agent}"
ENVSH="$WS/env.sh"
REPO="${REPO_DIR:-$WS/MindSpeed-Ops}"
BRANCH="${BRANCH:-master}"
SOC="${SOC_VERSION:-ascend910_93}"

echo "[daily] 1/3 卸载当前 mindspeed_ops"
bash -c "source $ENVSH && pip uninstall -y mindspeed_ops 2>/dev/null || true"

echo "[daily] 2/3 拉取 gitcode master 分支最新修改"
# source env.sh：GIT_SSL_NO_VERIFY + proxy（git fetch 必需；2026-08-28 实战：
# 缺 source 时 fetch 报 CAfile 证书校验失败，静默沿用旧 commit 装包）
source "$ENVSH" >/dev/null 2>&1 || true
cd "$REPO" 2>/dev/null || { echo "repo 缺失，先 clone"; mkdir -p "$WS"; cd "$WS" && git clone https://gitcode.com/Ascend/MindSpeed-Ops.git && cd MindSpeed-Ops; }
git remote set-url origin https://gitcode.com/Ascend/MindSpeed-Ops.git 2>/dev/null || true
if ! git fetch origin "$BRANCH" > /tmp/msops_fetch.log 2>&1; then
  tail -3 /tmp/msops_fetch.log; echo "[daily] git fetch 失败"; exit 1
fi
git checkout "$BRANCH" >/dev/null 2>&1 || true
git reset --hard "origin/$BRANCH"
echo "[daily] HEAD=$(git rev-parse --short HEAD) $(git log -1 --format=%s)"

echo "[daily] 3/3 源码安装（其他环境不变）"
# MINDSPEED_BUILD_CATLASS=1 必须在 pip install 前 export（2026-09-01 实测验证）：
# 缺它时 setup.py catlass_enabled=False → catlass 算子（如 chunk_loss）不编译,
# UT 全失败（test_chunk_loss.py 27F+2P）；带上后同一文件 29 passed in 10.96s。
# CI 脚本 ci/MindSpeed_Ops_ut.sh 同款（"must be enabled before installation"）。
bash -c "source $ENVSH && export SOC_VERSION=$SOC MINDSPEED_BUILD_CATLASS=1 && cd $REPO && MAX_JOBS=64 pip install -e . --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn --no-build-isolation --no-deps" 2>&1 | tail -5
bash -c "source $ENVSH && python -c 'import mindspeed_ops; print(\"msops import OK\")'"
echo "[daily] install OK"
