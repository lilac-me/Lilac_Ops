#!/bin/bash
# Path B: 版本看护 —— B 版本配套更新后，把环境全部更新为用户提供的最新配套包：
#         CANN 三件套(.run) + torch/torch_npu/triton_ascend/atk(whl, --no-deps) + atk 运行时依赖，
#         再拉取最新 MindSpeed-Ops 源码安装。
# 要求: 用户把配套包放入 $PKG（默认 /mnt/share/ops_nightly_agent/env-pkg，与旧版共存，
#       按文件名/新旧自动取最高/最新即可；如文件名版本号不递增请手动改 PKG 下的旧包名）。
# 用法: bash version_update_all.sh [PKG_DIR]
set -x
WS=/mnt/share/ops_nightly_agent
PKG=${1:-$WS/env-pkg}
export TMPDIR=/tmp/cann_tmp
mkdir -p /tmp/cann_tmp $WS/logs

# proxy 与共享 conda 根：开源仓库不含内部凭据——经环境变量注入
#   （PROXY_IP / PROXY_USER / PROXY_PASS / MINICONDA_ROOT，见 references/environment-setup.md）。
#   正常路径下 run_all 已 source env.sh，http_proxy 已导出，proxy 块自动跳过。
if [ -z "${http_proxy:-}" ]; then
  IP="${PROXY_IP:?PROXY_IP 未设置}"
  export IP
  export http_proxy="http://${PROXY_USER:?PROXY_USER 未设置}:${PROXY_PASS:?PROXY_PASS 未设置}@$IP:8091"
  export https_proxy=$http_proxy
  export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
fi
source ${MINICONDA_ROOT:?MINICONDA_ROOT 未设置}/etc/profile.d/conda.sh
conda activate ops_nightly_agent
export PATH=${MINICONDA_ROOT}/bin:$PATH

echo "=== 0. package inventory ==="
ls -la "$PKG/"

echo "=== 1. uninstall old npu stack ==="
pip3 uninstall -y mindspeed_ops torch_npu triton_ascend atk 2>/dev/null || true

echo "=== 2. CANN: wipe isolated dir + install three .run into $WS/ascend ==="
rm -rf "$WS/ascend"
mkdir -p "$WS/ascend"
TOOLKIT=$(ls "$PKG"/Ascend-cann-toolkit_*.run 2>/dev/null | sort | tail -1)
A3OPS=$(ls "$PKG"/Ascend-cann-A3-ops_*.run 2>/dev/null | sort | tail -1)
NNAL=$(ls "$PKG"/Ascend-cann-nnal_*.run 2>/dev/null | sort | tail -1)
[ -z "$TOOLKIT" ] && { echo "TOOLKIT_RUN_MISSING"; exit 101; }
chmod +x "$TOOLKIT" "$A3OPS" "$NNAL"
"$TOOLKIT" --full -q --install-path="$WS/ascend" || { echo TOOLKIT_FAIL; exit 102; }
source "$WS/ascend/ascend-toolkit/set_env.sh"
[ -n "$A3OPS" ] && "$A3OPS" --install --quiet --install-path="$WS/ascend" || { echo A3OPS_FAIL; exit 103; }
[ -n "$NNAL" ] && "$NNAL" --install --quiet --install-path="$WS/ascend" || { echo NNAL_FAIL; exit 104; }
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

echo "=== 3. conda 基础依赖（构建必需, tuna 镜像, 失败重试） ==="
DEPS="pip setuptools wheel numpy pyyaml filelock typing-extensions sympy networkx jinja2 fsspec pybind11 pytest"
for i in 1 2 3 4 5; do
  python -m pip install -U $DEPS --default-timeout=300 --retries=10 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn && break
  sleep 25
done

echo "=== 4. install whls from $PKG (--no-deps) ==="
TORCH=$(ls "$PKG"/torch-2.10.0-cp312-*.whl 2>/dev/null | head -1)
TORCHNPU=$(ls "$PKG"/torch_npu-*.whl 2>/dev/null | sort | tail -1)
TRITON=$(ls "$PKG"/triton_ascend-*.whl 2>/dev/null | sort | tail -1)
ATK=$(ls "$PKG"/atk-*.whl 2>/dev/null | sort | tail -1)
[ -z "$TORCH" ] && { echo "TORCH_WHL_MISSING"; exit 105; }
pip install --no-deps "$TORCH"   || { echo TORCH_FAIL; exit 106; }
[ -n "$TORCHNPU" ] && pip install --no-deps "$TORCHNPU" || { echo TORCHNPU_FAIL; exit 107; }
[ -n "$TRITON" ]  && pip install --no-deps "$TRITON"  || { echo TRITON_FAIL; exit 108; }
[ -n "$ATK" ]     && pip install --no-deps "$ATK"     || { echo ATK_FAIL; exit 109; }

echo "=== 5. atk 运行时依赖补齐 ==="
python -m pip install -U requests kombu sqlalchemy celery tqdm opentelemetry-api wcwidth \
  'pillow==12.3.0' 'matplotlib==3.11.0' \
  --default-timeout=300 --retries=10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn || echo ATK_DEPS_WARN

echo "=== 6. 刷新 env.sh + versions.txt ==="
cat > $WS/env.sh <<EOF
export http_proxy="${http_proxy:?http_proxy 未设置（先注入 PROXY_IP/PROXY_USER/PROXY_PASS）}"
export https_proxy=\$http_proxy
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
export GIT_SSL_NO_VERIFY=1
source $WS/ascend/ascend-toolkit/set_env.sh
source ${MINICONDA_ROOT:?MINICONDA_ROOT 未设置}/etc/profile.d/conda.sh
conda activate ops_nightly_agent
export SOC_VERSION=ascend910_93
EOF
cat > $WS/versions.txt <<EOF
python: $(python -c "import sys; print(sys.version.split()[0])")
CANN: $(head -1 "$WS/ascend/ascend-toolkit/latest/compiler/version.info" 2>/dev/null | tr -d '\r')
torch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)
torch_npu: $(python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
triton_ascend: $(python -c "import triton; print(triton.__version__)" 2>/dev/null)
atk: $(atk --version 2>/dev/null | head -1)
SOC_VERSION: ascend910_93
EOF
cat $WS/versions.txt

echo "=== 7. 拉取最新 MindSpeed-Ops 并安装 ==="
export SOC_VERSION=ascend910_93
cd $WS
rm -rf MindSpeed-Ops
git clone https://gitcode.com/Ascend/MindSpeed-Ops.git
cd MindSpeed-Ops
git log -1 --format='MINDSPEED_COMMIT %H %s'
pip3 install -e . --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn --no-build-isolation --no-deps || { echo MSOPS_INSTALL_FAIL; exit 110; }

echo "=== 8. 验证矩阵 ==="
bash -c 'source '$WS'/env.sh && python -c "import torch,torch_npu,triton,atk; print(chr(111)+chr(107), torch.__version__, torch_npu.npu.device_count())"'
echo VERSION_UPDATE_DONE
