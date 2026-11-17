#!/bin/bash
# Path A: 日常看护 —— 只更新 MindSpeed-Ops（卸载旧包 -> 拉取仓最新 master -> 源码重装），
#         其他环境（conda/CANN/torch/torch_npu/triton_ascend/atk）保持不变。
# 用法: bash daily_update_mindspeed.sh
set -x
WS=/mnt/share/ops_nightly_agent
source $WS/env.sh
export SOC_VERSION=ascend910_93

echo "=== 1. uninstall old mindspeed_ops ==="
pip3 uninstall mindspeed_ops -y 2>/dev/null || echo "mindspeed_ops not installed"

echo "=== 2. sync latest master branch ==="
cd $WS/MindSpeed-Ops || { echo "repo missing, clone"; cd $WS && git clone https://gitcode.com/Ascend/MindSpeed-Ops.git && cd MindSpeed-Ops; }
git fetch origin master
git checkout master
git reset --hard origin/master
git log -1 --format='MINDSPEED_COMMIT %H %s'

echo "=== 3. reinstall from source (env unchanged) ==="
pip3 install -e . --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn --no-build-isolation --no-deps

echo "=== 4. smoke ==="
python -c "import mindspeed_ops; print('MINDSPEED_OPS_OK', mindspeed_ops.__file__)" 2>&1 | tail -n 2
echo DAILY_UPDATE_DONE
