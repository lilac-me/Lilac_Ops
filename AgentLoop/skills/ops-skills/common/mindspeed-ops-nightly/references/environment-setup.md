# 环境准备要点（environment-setup）

> 本文件取代历史设计/执行记录文档，沉淀为 skill 自包含的**环境安装与验证要点**。
> 面向场景：首次搭建看护环境（Step2/Step3）或 version 模式下全量更新环境（`mode_version.sh` 的 `PKG_DIR` 布局即本文件所述 env-pkg 布局）。

## 环境事实（部署基线）

| 项 | 值 |
| --- | --- |
| WS（工作根） | `/mnt/share/ops_nightly_agent`（NFS 共享，换容器复用） |
| 看护 conda env | `ops_nightly_agent`（python 3.12；envs 落 `/mnt/share/vllm/conda_envs/`） |
| 共享 conda | `/mnt/share/{user}/miniconda3`（conda 26.x，base py3.14） |
| CANN | 9.2.0 三件套（toolkit + A3-ops + nnal）**隔离安装**到 `$WS/ascend`（不动默认 /usr/local/Ascend） |
| SOC_VERSION | `ascend910_93`（用户指定，必须显式 export；自动探测会得到 9382） |
| 设备 | A3 容器默认 6 卡；卡资源参数化（`NCARDS` / `ASCEND_RT_VISIBLE_DEVICES`，服务器按实际逻辑卡 0..N-1 覆盖） |
| 代理 | `80.*`→`80.5.1.30:8091`，`90.*`→`90.91.191.181`，其它→`141.61.33.201`；`GIT_SSL_NO_VERIFY=1` |
| 内部包目录 | `$PKG_DIR`（如 `$WS/env-pkg`）：CANN `*.run` + torch/torch_npu/triton_ascend/atk `*.whl`（cp312 aarch64） |
| 仓库 | `https://gitcode.com/Ascend/MindSpeed-Ops.git`，分支 master，repo 在 `$WS/MindSpeed-Ops` |

## 一、conda 共享配置（新容器必做）

新容器必须写 `~/.condarc`（含 `envs_dirs`，否则 `conda activate ops_nightly_agent` 报 `EnvironmentNameNotFound`）。

```bash
cat > ~/.condarc <<'EOF'
envs_dirs:
  - /mnt/share/vllm/conda_envs
pkgs_dirs:
  - /mnt/share/vllm/conda_pkgs
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
EOF
source /mnt/share/{user}/miniconda3/etc/profile.d/conda.sh
```text
校验：`conda env list | grep ops_nightly_agent`、`which python` 指向该 env。若需新建：

```bash
conda create -n ops_nightly_agent python=3.12 -y
conda activate ops_nightly_agent
python -m pip install --upgrade pip setuptools wheel
```text
## 二、CANN 9.2.0 隔离安装（三件套）

```bash
ASCEND_PATH=$WS/ascend
mkdir -p $ASCEND_PATH /tmp/cann_tmp && export TMPDIR=/tmp/cann_tmp
for pkg in Ascend-cann-toolkit_9.2.0_linux-aarch64 Ascend-cann-A3-ops_9.2.0_linux-aarch64 Ascend-cann-nnal_9.2.0_linux-aarch64; do
  chmod +x $PKG_DIR/${pkg}.run
done
$PKG_DIR/Ascend-cann-toolkit_9.2.0_linux-aarch64.run --full -q --install-path=$ASCEND_PATH
source $ASCEND_PATH/ascend-toolkit/set_env.sh
$PKG_DIR/Ascend-cann-A3-ops_9.2.0_linux-aarch64.run --install --quiet --install-path=$ASCEND_PATH
$PKG_DIR/Ascend-cann-nnal_9.2.0_linux-aarch64.run --install --quiet --install-path=$ASCEND_PATH
```text
要点：A3-ops 为 A3 芯片必需；安装包合计约 4.4G，运行耗时较长（nohup + 日志）；解压目录默认 /tmp（本地盘）。

## 三、torch / torch_npu / triton-ascend / ATK（内部 whl，--no-deps）

```bash
conda activate ops_nightly_agent
pip install $PKG_DIR/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl
pip install $PKG_DIR/torch_npu-2.10.0.post1.dev*-cp312-cp312-manylinux_2_28_aarch64.whl
pip install numpy pyyaml                       # torch_npu import 必需
pip install $PKG_DIR/triton_ascend-3.6.0+dev*-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
pip install $PKG_DIR/atk-26.7.8.dev*-cp312-cp312-linux_aarch64.whl
atk --help | head -20        # 验证 CLI 可用
```text
- triton_ascend 分发包提供 `triton` 3.6.0 模块，**无顶层 `triton_ascend` 模块属正常**。
- atk 运行时依赖补齐闭环（全部落入 NFS env）：requests、PIL、matplotlib、wcwidth、kombu、sqlalchemy、celery、tqdm、opentelemetry-api。注意 `atk node --help` 通过 ≠ 任务链可用（kombu 等 task 路径才 import），需真实单算子冒烟终验；发行版名≠模块名（`opentelemetry`→`opentelemetry-api`、`attr`→`attrs`）；勿装 `atk.*` 点号子模块（可选注册项）。

## 四、固化 env.sh

```bash
cat > $WS/env.sh <<'EOF'
source $WS/ascend/ascend-toolkit/set_env.sh   # 隔离 CANN 9.2.0
export IP=<proxy_ip>
export http_proxy="http://{user}:<pass>@$IP:8091"
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
export GIT_SSL_NO_VERIFY=1
source /mnt/share/{user}/miniconda3/etc/profile.d/conda.sh
conda activate ops_nightly_agent
EOF
```text
每条远端命令显式 `source $WS/env.sh`（非交互 shell 不读 .bashrc）；不沿用 .bashrc 的 LD_PRELOAD(jemalloc)（避免影响性能对比一致性）。

## 五、安装 MindSpeed-Ops（源码）

```bash
source $WS/env.sh
cd $WS
git clone https://gitcode.com/Ascend/MindSpeed-Ops.git
pip install setuptools wheel pybind11 numpy pyyaml pytest   # --no-build-isolation --no-deps 需要
cd $WS/MindSpeed-Ops
export SOC_VERSION=ascend910_93                            # 用户指定，必须显式
export MINDSPEED_BUILD_CATLASS=1                             # catlass 算子必须编译（chunk_loss 等）
export MAX_JOBS=64                                           # 防编译 OOM
pip3 install -e . \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn \
  --no-build-isolation --no-deps
```text
- 会触发 CMake 编译 `mindspeed_ops_C` + `build_aclnn.sh`（A3 自定义算子，耗时长）→ 后台 + 日志。
- 冒烟：`python -c "import mindspeed_ops; print('mindspeed_ops OK')"` + `pytest unit_tests --collect-only -q | tail -5`。
- 复测/后续容器：env 在 NFS 上直接复用，无需重装（仅 daily 模式重装 mindspeed-ops 源码）。

## 六、验证矩阵（全部通过才进入看护）

| # | 校验（先 source $WS/env.sh） | 期望 |
| --- | --- | --- |
| 1 | `python --version` | 3.12.x |
| 2 | `conda env list \| grep ops_nightly_agent` | 存在 |
| 3 | `npu-smi info` | 6 卡 OK |
| 4 | `import torch; torch.__version__` | 2.10.0 |
| 5 | `import torch_npu; torch_npu.npu.device_count()` | 6 |
| 6 | NPU 冒烟：`torch.ones(4,4,device='npu')+1` | 成功 |
| 7 | `import triton; import triton_ascend` | 成功 |
| 8 | `atk --help` | CLI 可用 |
| 9 | `pip list` 关键版本 → `$WS/versions.txt` | 记录基线 |

## 七、风险与预案（要点）

| # | 风险 | 预案 |
| --- | --- | --- |
| R1 | 隔离 CANN 与默认 /usr/local/Ascend 混用 | env.sh 统一 source 隔离路径；不用全局 .bashrc 的 source |
| R2 | conda 通道 403/慢 | 只留 tsinghua pkgs/main；必要时追加 free 通道/huaweicloud |
| R3 | torch_npu/triton_ascend 与 CANN 组合问题 | import 冒烟逐项定位；内部包组合由用户保证 |
| R4 | gitcode clone/fetch 抖动或证书 | `GIT_SSL_NO_VERIFY=1` + source env.sh 后再 fetch + 校验退出码 |
| R5 | 编译 OOM | `MAX_JOBS`（如 64）；NFS 上编译慢属预期 |

## 八、vllm / vllm-ascend（独立环境）

版本冲突红线：看护环境为 torch_npu 2.10.0.post1.dev + triton_ascend 3.6.0+dev；vllm-ascend 强制 torch-npu==2.10.0 / triton-ascend==3.2.1，**必须装独立环境**（如 `ops_nightly_vllm`），否则看护失效。安装规范详见 `vllm_install_notes.md`。
