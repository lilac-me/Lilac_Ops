#!/usr/bin/env bash
# bootstrap.sh — 每次换容器/服务器后必跑（skill init 阶段）
# 作用: 写 .condarc → source env.sh 验证 → 依赖自检 → mindspeed-ops 可导入 → 设备可用性自检
set -e
WS="${WS:-/mnt/share/ops_nightly_agent}"
ENVSH="$WS/env.sh"
CONDA_ENVS="/mnt/share/vllm/conda_envs"
CONDA_PKGS="/mnt/share/vllm/conda_pkgs"

echo "[bootstrap] 1/5 写 ~/.condarc"
cat > /root/.condarc <<'EOF'
envs_dirs:
  - /mnt/share/vllm/conda_envs
pkgs_dirs:
  - /mnt/share/vllm/conda_pkgs
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
EOF
echo "[bootstrap] condarc OK"

echo "[bootstrap] 2/5 环境验证"
bash -c "source $ENVSH && python - <<'PY'
import torch, torch_npu, atk, PIL, matplotlib  # atk 任务链依赖
print('torch', torch.__version__)
print('npu_devices', torch_npu.npu.device_count())
print('atk', atk.__version__ if hasattr(atk,'__version__') else 'ok')
assert torch_npu.npu.device_count() > 0
PY" || { echo "[bootstrap] env 验证失败"; exit 1; }
echo "[bootstrap] env OK"

echo "[bootstrap] 3/5 atk 依赖自检（缺哪个补哪个，需走 proxy）"
bash -c "source $ENVSH && python - <<'PY'
mods = ['requests','PIL','matplotlib','wcwidth','kombu','sqlalchemy','celery','tqdm','opentelemetry']
import importlib, sys
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('MISSING:', missing); sys.exit(2)
print('atk deps OK')
PY" || { echo "[bootstrap] 依赖缺失，调用补装（依赖红线见 references/environment-setup.md：发行版名≠模块名，opentelemetry-api 装的是 opentelemetry 模块；numba 仅 vllm-ascend 需要，atk 链不需要）"; exit 1; }

echo "[bootstrap] 4/5 mindspeed-ops 冒烟"
bash -c "source $ENVSH && export SOC_VERSION=ascend910_93 && cd $WS/MindSpeed-Ops/tests && python -c 'import mindspeed_ops; print(\"msops import OK\")'"

echo "[bootstrap] 5/5 设备可用性自检（2026-09-09 双模式新增; 暴露物理号/逻辑号错位陷阱）"
# 容器与服务器都以"逻辑设备号 0..N-1"访问（55 容器实测物理 davinci12/13 不可用, 须 0/1）。
# 若此步失败: 检查环境变量 ASCEND_RT_VISIBLE_DEVICES 的取值是否为本机可用逻辑设备。
bash -c "source $ENVSH && python - <<'PY'
import torch, torch_npu
n = torch_npu.npu.device_count()
for i in range(n):
    torch_npu._C._npu_setDevice(i)
print(f'devices {n} OK (逻辑 0..{n-1} 全部可用)')
PY" || { echo "[bootstrap] 设备自检失败（检查 ASCEND_RT_VISIBLE_DEVICES / 逻辑设备号）"; exit 1; }
echo "[bootstrap] 全部 OK"
