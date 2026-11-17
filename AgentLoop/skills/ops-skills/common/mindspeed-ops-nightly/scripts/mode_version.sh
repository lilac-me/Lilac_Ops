#!/usr/bin/env bash
# mode_version.sh — 版本看护：B 版本配套更新后，按用户显式提供的 PKG_DIR 全量更新环境
# （CANN .run 三件套 → conda 依赖 → torch/torch_npu/triton_ascend/atk whl → atk 运行时依赖
#   → 刷新 env.sh/versions.txt），再拉取 gitcode master 最新 mindspeed-ops 源码安装。
# 用法: PKG_DIR=<目录> bash mode_version.sh      （PKG_DIR 必填；配套包布局与 env-pkg 相同）
# 说明: 合并自 version_update_all.sh（2026-08-26 全链路验证的完整实现）。
set -e
WS="${WS:-/mnt/share/ops_nightly_agent}"
PKG="${PKG_DIR:?PKG_DIR 必须显式提供（B 版本配套包统一存放目录）}"
test -d "$PKG" || { echo "PKG_DIR 不存在: $PKG"; exit 1; }
REPO="${REPO_DIR:-$WS/MindSpeed-Ops}"
SOC="${SOC_VERSION:-ascend910_93}"
export TMPDIR=/tmp/cann_tmp
mkdir -p /tmp/cann_tmp "$WS/logs"

# proxy（安装走外网必需；开源仓库不含内部凭据——内部部署时经环境变量注入：
#   PROXY_IP / PROXY_USER / PROXY_PASS，见 references/environment-setup.md。
#   正常路径下 run_all 已 source env.sh，http_proxy 已导出，本块自动跳过）
if [ -z "${http_proxy:-}" ]; then
  IP="${PROXY_IP:?PROXY_IP 未设置}"
  export IP
  export http_proxy="http://${PROXY_USER:?PROXY_USER 未设置}:${PROXY_PASS:?PROXY_PASS 未设置}@$IP:8091"
  export https_proxy=$http_proxy
  export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
fi

# 2026-09-10 修复（80.5.27.94 实战抓 bug）：原实现 source conda.sh + conda activate +
# export PATH=miniconda3/bin:$PATH —— PATH 前置 base bin 会把刚激活的 env 遮掉，
# 导致后续 python/pip3 落到 miniconda base（python3.14），numpy 等依赖解析失败、
# 且 bash -c "conda activate" 报 CondaError（shell 函数不跨 bash -c）。
# 统一改走 $WS/env.sh（CANN set_env + proxy + conda activate，bootstrap 写好 condarc 后必可用）。
source "$WS/env.sh" >/dev/null 2>&1 || { echo "[version] env.sh source 失败（新容器先跑 bootstrap.sh 写 condarc）"; exit 1; }

echo "[version] PKG_DIR=$PKG"
echo "[version] 目录内容:"; ls -la "$PKG" | head -30

echo "[version] 1/6 卸载旧 npu 栈"
pip3 uninstall -y mindspeed_ops torch_npu triton_ascend atk 2>/dev/null || true

echo "[version] 2/6 更新 CANN 全套（*.run → 隔离目录 $WS/ascend，勿动系统默认 Ascend）"
TOOLKIT=$(ls "$PKG"/Ascend-cann-toolkit_*.run 2>/dev/null | sort | tail -1)
A3OPS=$(ls "$PKG"/Ascend-cann-A3-ops_*.run 2>/dev/null | sort | tail -1)
NNAL=$(ls "$PKG"/Ascend-cann-nnal_*.run 2>/dev/null | sort | tail -1)
if [ -z "$TOOLKIT" ]; then echo "[version] PKG_DIR 无 toolkit .run，仅更新 python 包（CANN 不变）"; else
  rm -rf "$WS/ascend"; mkdir -p "$WS/ascend"
  chmod +x "$TOOLKIT" "$A3OPS" "$NNAL"
  "$TOOLKIT" --full -q --install-path="$WS/ascend" || { echo TOOLKIT_FAIL; exit 100; }
  source "$WS/ascend/ascend-toolkit/set_env.sh"
  [ -n "$A3OPS" ] && "$A3OPS" --install --quiet --install-path="$WS/ascend" || { echo A3OPS_FAIL; exit 101; }
  [ -n "$NNAL" ] && "$NNAL" --install --quiet --install-path="$WS/ascend" || { echo NNAL_FAIL; exit 102; }
  echo "[version] CANN 安装完成 ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
fi

echo "[version] 3/6 conda 基础依赖（构建必需, tuna 镜像, 失败重试）"
DEPS="pip setuptools wheel numpy pyyaml filelock typing-extensions sympy networkx jinja2 fsspec pybind11 pytest"
for i in 1 2 3 4 5; do
  python -m pip install -U $DEPS --default-timeout=300 --retries=10 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn && break
  sleep 25
done

echo "[version] 4/6 更新 python 配套包（*.whl → pip install --no-deps，按文件名版本自动取新）"
TORCH=$(ls "$PKG"/torch-2.10.0-cp312-*.whl 2>/dev/null | head -1)
TORCHNPU=$(ls "$PKG"/torch_npu-*.whl 2>/dev/null | sort | tail -1)
TRITON=$(ls "$PKG"/triton_ascend-*.whl 2>/dev/null | sort | tail -1)
ATKWHL=$(ls "$PKG"/atk-*.whl "$PKG"/ATK-*.whl 2>/dev/null | sort | tail -1)
# 2026-09-10 修复: B050 包名为大写 ATK-*.whl，原 atk-*.whl 小写 glob 在 Linux 大小写敏感下漏包
# （版本自检 import atk 报 ModuleNotFoundError）；显式补一个大写 glob。
[ -z "$TORCH" ] && { echo "[version] 无 torch whl，中止"; exit 103; }
# 2026-09-10 修复: 原 bash -c "conda activate ... && pip ..." 跨 shell 丢 conda 函数 → CondaError；
# 当前 shell 已由 env.sh 激活 ops_nightly_agent，直接 pip 即可。
pip install --no-deps "$TORCH"
[ -n "$TORCHNPU" ] && pip install --no-deps "$TORCHNPU"
[ -n "$TRITON" ] && pip install --no-deps "$TRITON"
[ -n "$ATKWHL" ] && pip install --no-deps "$ATKWHL"

echo "[version] 5/6 atk 运行时依赖补齐 + 刷新 env.sh / versions.txt"
python -m pip install -U requests kombu sqlalchemy celery tqdm opentelemetry-api wcwidth \
  'pillow==12.3.0' 'matplotlib==3.11.0' \
  --default-timeout=300 --retries=10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn || echo ATK_DEPS_WARN
cat > "$WS/env.sh" <<EOF
export http_proxy="${http_proxy:?http_proxy 未设置（先注入 PROXY_IP/PROXY_USER/PROXY_PASS）}"
export https_proxy=\$http_proxy
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
export GIT_SSL_NO_VERIFY=1
source $WS/ascend/ascend-toolkit/set_env.sh
source ${MINICONDA_ROOT:?MINICONDA_ROOT 未设置（共享 conda 根, 见 references/environment-setup.md）}/etc/profile.d/conda.sh
conda activate ops_nightly_agent
export SOC_VERSION=ascend910_93
EOF
cat > "$WS/versions.txt" <<EOF
python: $(python -c "import sys; print(sys.version.split()[0])")
CANN: $(head -1 "$WS/ascend/ascend-toolkit/latest/compiler/version.info" 2>/dev/null | tr -d '\r')
torch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)
torch_npu: $(python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
triton_ascend: $(python -c "import triton; print(triton.__version__)" 2>/dev/null)
atk: $(atk --version 2>/dev/null | head -1)
SOC_VERSION: ascend910_93
EOF
cat "$WS/versions.txt"

echo "[version] 6/6 拉取 gitcode master 最新 mindspeed-ops 并源码安装"
cd "$WS" && rm -rf MindSpeed-Ops
git clone https://gitcode.com/Ascend/MindSpeed-Ops.git
cd MindSpeed-Ops
echo "[version] HEAD=$(git rev-parse --short HEAD) $(git log -1 --format=%s)"
# MINDSPEED_BUILD_CATLASS=1 必须在安装前 export（2026-09-01 实测验证）：
# 缺它时 catlass 算子（chunk_loss 等）不编译, UT 全失败；带上后 29 passed。
bash -c "source $WS/env.sh && export SOC_VERSION=$SOC MINDSPEED_BUILD_CATLASS=1 && cd $WS/MindSpeed-Ops && MAX_JOBS=64 pip install -e . --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn --no-build-isolation --no-deps" 2>&1 | tail -5

echo "[version] 版本自检"
bash -c "source $WS/env.sh && python - <<'PY'
import torch, torch_npu, triton, atk
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__, '| npu count', torch_npu.npu.device_count())
print('triton', getattr(triton, '__version__', '?'))
print('atk', getattr(atk, '__version__', 'ok'))
PY"
echo "[version] 配套包更新完成 → 继续 UT/精度/性能"
