# Step 2: 环境安装 (Environment Installation)

## 前提条件

- Step 1 完成，已通过 SSH 连接到服务器
- 已完成平台检测（NPU 或 GPU）
- **以下所有步骤中，如果配置了 `${PROXY_ADDRESS}`，每个命令块都会先设置代理环境变量：**

  ```bash
  export http_proxy=${PROXY_ADDRESS}
  export https_proxy=${PROXY_ADDRESS}
  ```

- **非 root 用户自动切换**: 每个命令块在执行前会检查当前用户，如果不是 root，通过 `sudo su -` 切换到 root 权限执行：

  ```bash
  if [ "$(whoami)" != "root" ]; then
      exec sudo su - -c "..."
  fi
  ```

  为简洁起见，文档中不再在每个命令块中重复 `sudo su`，Skill 执行时会自动处理。

---

## 2.0 安装前预检

在正式开始安装前，执行以下检查避免后续步骤失败。

### 磁盘空间检查

CANN 安装包解压需要临时空间（约 5-10GB），默认解压到 `/tmp` 或 `$HOME`。如果根分区空间不足，需提前清理或设置 `TMPDIR`：

```bash
# 检查根分区可用空间
df -h / | tail -1

# 如果可用空间 < 10GB，清理缓存
pip cache purge 2>/dev/null || true
conda clean --all -y 2>/dev/null || true

# 后续 CANN 安装时将设置 TMPDIR 到工作目录
export TMPDIR=${WORKSPACE}/tmp_cann
mkdir -p ${TMPDIR}
```

### 网络连通性检查

vllm / vllm-ascend 需要从 GitHub 克隆源码，网络不可达时安装将失败。

```bash
# 检查 GitHub 连通性
curl -s --connect-timeout 10 https://github.com -o /dev/null && echo "GitHub: OK" || echo "GitHub: UNREACHABLE — 需配置代理或使用离线包"
```

> **GitHub 间歇性不可达**: 即使单次 curl 成功，git clone 仍可能因连接超时失败。建议在 git clone 命令中内置重试逻辑（参见 2.2.3）。

### 构建工具预装

vllm 从源码编译需要 `setuptools_rust`，需提前安装：

```bash
conda activate ${CONDA_ENV_NAME}
pip install setuptools_rust wheel 2>&1
```

---

## 2.1 创建工作目录

```bash
mkdir -p ${WORKSPACE}
cd ${WORKSPACE}
mkdir -p logs
```

---

## 2.2 NPU 平台环境安装

### 2.2.1 安装 CANN (Ascend Computing Language)

CANN 安装包统一从**服务器上的 `${CANN_LOCAL_PATH}` 目录**下获取（由用户在 example.md 中指定）。安装前会先检测服务器架构（x86_64 / aarch64）并校验安装包架构是否匹配。

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

# ASCEND_PATH 固定为工作目录下的 ascend 文件夹
ASCEND_PATH=${WORKSPACE}/ascend
mkdir -p ${ASCEND_PATH}

# ─── 平台架构检测 ───────────────────────────────────────────
ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)  CANN_ARCH="x86_64" ;;
    aarch64) CANN_ARCH="aarch64" ;;
    *)
        echo "[错误] 不支持的服务器架构: ${ARCH}，CANN 仅支持 x86_64 或 aarch64"
        exit 1
        ;;
esac
echo "服务器架构: ${ARCH} → 期望 CANN 包后缀: linux-${CANN_ARCH}"

# ─── 检查目标路径是否已安装 CANN ───────────────────────────
# 检查 ${ASCEND_PATH} 下是否已有 CANN 环境（通过 set_env.sh 和 install.info 判断）
CANN_INSTALLED=false
if [ -f "${ASCEND_PATH}/ascend-toolkit/set_env.sh" ]; then
    CANN_INSTALLED=true
    EXISTING_CANN_VERSION=$(cat ${ASCEND_PATH}/ascend_toolkit_install.info 2>/dev/null | grep "^version=" | cut -d= -f2)
    if [ -n "${EXISTING_CANN_VERSION}" ]; then
        echo "检测到 ${ASCEND_PATH} 下已安装 CANN，版本: ${EXISTING_CANN_VERSION}"
    else
        echo "检测到 ${ASCEND_PATH} 下已安装 CANN（版本信息未知）"
    fi
fi

# ─── 查找 CANN 安装包 ───────────────────────────────────────
# 从 ${CANN_LOCAL_PATH} 目录下查找 CANN toolkit .run 安装包（通配符匹配任意版本，按架构精确匹配）
CANN_PKG=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "Ascend-cann-toolkit_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)

# ─── 安装决策 ───────────────────────────────────────────────
SKIP_CANN_TOOLKIT_INSTALL=false
if [ "${CANN_INSTALLED}" = "true" ]; then
    if [ -z "${CANN_PKG}" ]; then
        echo "CANN 已安装且未提供新的安装包，跳过 CANN toolkit 安装步骤"
        SKIP_CANN_TOOLKIT_INSTALL=true
    else
        echo "检测到新的 CANN 安装包: $(basename ${CANN_PKG})"
        echo "将覆盖安装到 ${ASCEND_PATH}（建议先手动卸载旧版本）"
    fi
fi

# ─── 架构匹配校验 + 安装 ───────────────────────────────────
if [ "${SKIP_CANN_TOOLKIT_INSTALL}" = "false" ]; then
    if [ -z "${CANN_PKG}" ]; then
        # 未找到匹配架构的包，检查是否存在其他架构的包（给出更明确的错误提示）
        MISMATCH_PKG=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "Ascend-cann-toolkit_*_linux-*.run" 2>/dev/null | head -1)
        if [ -n "${MISMATCH_PKG}" ]; then
            echo "[错误] CANN 包架构不匹配:"
            echo "  服务器架构: ${ARCH} (期望 linux-${CANN_ARCH})"
            echo "  实际安装包: $(basename ${MISMATCH_PKG})"
            echo "  请将架构匹配的 CANN 安装包放置到 ${CANN_LOCAL_PATH}/ 目录"
        else
            echo "[错误] 未在 ${CANN_LOCAL_PATH} 下找到任何 CANN toolkit 安装包"
            echo "  请确认 CANN_LOCAL_PATH 参数指向正确的安装包目录"
        fi
        exit 1
    fi
    echo "CANN 包架构校验通过: $(basename ${CANN_PKG})"

    # ─── 磁盘空间二次确认 ─────────────────────────────────────
    ROOT_AVAIL=$(df -k / | tail -1 | awk '{print $4}')
    if [ "${ROOT_AVAIL}" -lt 5000000 ]; then
        echo "[警告] 根分区可用空间不足 5GB，设置 TMPDIR=${WORKSPACE}/tmp_cann"
        export TMPDIR=${WORKSPACE}/tmp_cann
        mkdir -p ${TMPDIR}
    fi

    # 安装 CANN Toolkits
    chmod +x ${CANN_PKG}
    ${CANN_PKG} --full -q --install-path=${ASCEND_PATH}
fi

# 加载 CANN 环境变量（安装 A3 Ops 和 NNAL 前必须执行）
source ${ASCEND_PATH}/ascend-toolkit/set_env.sh

# 安装 CANN A3 Ops（按架构精确匹配）
A3_PKG=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "Ascend-cann-A3-ops_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)
if [ -f "${A3_PKG}" ]; then
    chmod +x ${A3_PKG}
    ${A3_PKG} --install --quiet --install-path=${ASCEND_PATH}
else
    echo "提示: 未找到 A3 Ops 包（Ascend-cann-A3-ops_*_linux-${CANN_ARCH}.run），跳过"
fi

# 安装 CANN NNAL（按架构精确匹配）
NNAL_PKG=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "Ascend-cann-nnal_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)
if [ -f "${NNAL_PKG}" ]; then
    chmod +x ${NNAL_PKG}
    ${NNAL_PKG} --install --quiet --install-path=${ASCEND_PATH}
else
    echo "提示: 未找到 NNAL 包（Ascend-cann-nnal_*_linux-${CANN_ARCH}.run），跳过"
fi
```

### 2.2.2 创建 Conda 环境并安装 PyTorch

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

# 创建 conda 环境
conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

# ─── 安装 PyTorch + torch_npu ────────────────────────────────
# 优先使用 ${CANN_LOCAL_PATH} 下的本地 whl 文件（离线安装，更快更可靠）
TORCH_WHL=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "torch-*cp${PYTHON_VERSION_NO_DOT}*.whl" 2>/dev/null | head -1)
TORCH_NPU_WHL=$(find ${CANN_LOCAL_PATH} -maxdepth 1 -name "torch_npu-*cp${PYTHON_VERSION_NO_DOT}*.whl" 2>/dev/null | head -1)

if [ -f "${TORCH_WHL}" ]; then
    echo "使用本地 whl: ${TORCH_WHL}"
    pip install ${TORCH_WHL}
else
    pip install torch==${TORCH_VERSION}
fi

if [ -f "${TORCH_NPU_WHL}" ]; then
    echo "使用本地 whl: ${TORCH_NPU_WHL}"
    pip install ${TORCH_NPU_WHL}
fi

# ─── 安装 torch_npu 运行时需要的依赖 ──────────────────────────
# numpy 和 pyyaml 是 torch_npu 导入时所必需的，缺失会导致导入失败
pip install numpy pyyaml

# 验证（需要 source CANN 环境变量）
source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch_npu; print(f'NPU count: {torch_npu.npu.device_count()}')"
```

> **本地 whl 优先**: `${CANN_LOCAL_PATH}` 中通常包含匹配架构的 torch 和 torch_npu 离线包，优先使用它们可以避免网络下载慢或 PyPI 版本不匹配的问题。
>
> **numpy / pyyaml**: torch_npu 导入时需要 `numpy` 和 `yaml` 模块，新创建的 conda 环境中这些尚未安装，缺少时会导致 `ModuleNotFoundError: No module named 'numpy'` 或 `No module named 'yaml'`。
>
> **CANN 环境变量**: 验证 torch_npu 时必须先 `source ${ASCEND_PATH}/ascend-toolkit/set_env.sh`，否则会出现 `ImportError: libhccl.so: cannot open shared object file`。

### 2.2.3 安装 vllm + vllm-ascend

> **关键**: 以下所有步骤必须在 `source ${ASCEND_PATH}/ascend-toolkit/set_env.sh` 之后执行，否则 vllm 构建时 torch 导入 torch_npu 后端会因缺少 CANN 运行库而失败。

```bash
# 加载 CANN 环境变量（必须在激活 conda 环境之前或之后执行）
source ${ASCEND_PATH}/ascend-toolkit/set_env.sh

# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

# 激活环境
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

# ─── 预装构建依赖 ────────────────────────────────────────────
# setuptools_rust 是 vllm pyproject.toml 构建所需的依赖，缺少时会出现:
#   ModuleNotFoundError: No module named 'setuptools_rust'
pip install setuptools_rust wheel 2>&1
```

#### A. 从源码安装 vllm

```bash
cd ${WORKSPACE}

# 克隆 vllm（含重试机制，应对 GitHub 间歇性不可达）
if [ ! -d "vllm" ]; then
    echo "克隆 vllm ${VLLM_VERSION}..."
    for i in 1 2 3 4 5; do
        echo "  尝试 $i/5..."
        git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git 2>&1 && break
        echo "  失败，等待 10s 后重试..."
        sleep 10
    done
    if [ ! -d "vllm/.git" ]; then
        echo "[错误] vllm 克隆失败，请检查 GitHub 网络连通性"
        exit 1
    fi
fi

cd vllm

# 关键: 设置 VLLM_TARGET_DEVICE=empty 跳过 GPU 编译
# 关键: 设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0 防止构建时触发 torch_npu 后端加载
export VLLM_TARGET_DEVICE=empty
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

echo "安装 vllm 通用依赖..."
pip install -r requirements/common.txt 2>&1

echo "安装 vllm（可编辑模式，跳过构建隔离）..."
pip install -e . --no-build-isolation 2>&1

# 恢复环境变量
unset TORCH_DEVICE_BACKEND_AUTOLOAD

# 验证
python -c "import vllm; print(f'vllm {vllm.__version__}')"
```

> **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`**: 构建 vllm 时，其 `setup.py` 会导入 `torch`，而 `torch` 在检测到 NPU 环境时会自动加载 `torch_npu` 后端。如果 CANN 环境变量未设置或 numpy/pyyaml 缺失，此加载会失败并导致 `metadata-generation-failed` 错误。设置此环境变量禁用后端自动加载可避免此问题。

#### B. 从源码安装 vllm-ascend

> **重要**: vllm-ascend 的 `requirements.txt` 和 `pyproject.toml` 中可能包含错误或不可用的包版本（如 `triton-ascend==3.2.1` 在 PyPI 上不存在，仅 `3.2.0` 可用）。安装前必须先修复两处文件。

```bash
cd ${WORKSPACE}

# 克隆 vllm-ascend（含重试机制）
if [ ! -d "vllm-ascend" ]; then
    echo "克隆 vllm-ascend..."
    for i in 1 2 3 4 5; do
        echo "  尝试 $i/5..."
        git clone --depth 1 https://github.com/vllm-project/vllm-ascend.git 2>&1 && break
        echo "  失败，等待 30s 后重试..."
        sleep 30
    done
    if [ ! -d "vllm-ascend/.git" ]; then
        echo "[错误] vllm-ascend 克隆失败"
        exit 1
    fi
fi

cd vllm-ascend

# 切换到指定 commit
git fetch origin ${VLLM_ASCEND_COMMIT} 2>&1
git checkout ${VLLM_ASCEND_COMMIT} 2>&1

# ─── 修复 triton-ascend 版本号 ─────────────────────────────────
# 问题: requirements.txt 和 pyproject.toml 中指定的版本可能在 PyPI 上不存在
# 解决: 先查询可用版本，替换为实际存在的版本
AVAILABLE_TRITON=$(pip index versions triton-ascend 2>/dev/null | grep "Available versions:" | head -1 || echo "")
if [ -n "${AVAILABLE_TRITON}" ]; then
    echo "triton-ascend 可用版本: ${AVAILABLE_TRITON}"
fi

# 修复 requirements.txt
sed -i 's/triton-ascend==3.2.1/triton-ascend==3.2.0/g' requirements.txt
echo "requirements.txt 已修复:"
grep 'triton-ascend' requirements.txt

# 修复 pyproject.toml（构建依赖也包含版本约束）
sed -i 's/triton-ascend==3.2.1/triton-ascend==3.2.0/g' pyproject.toml
echo "pyproject.toml 已修复:"
grep 'triton-ascend' pyproject.toml

# ─── 安装 vllm-ascend ─────────────────────────────────────────
# 使用 --no-build-isolation 跳过构建依赖检查（已经手动修复版本）
echo "安装 vllm-ascend 依赖..."
pip install -r requirements.txt 2>&1

echo "安装 vllm-ascend（可编辑模式）..."
pip install -e . --no-build-isolation 2>&1

# 验证（注意: vllm_ascend 模块可能没有 __version__ 属性）
python -c "
import vllm_ascend
print('vllm_ascend 导入成功')
print('模块路径:', vllm_ascend.__path__)
" 2>&1
```

> **版本号修复的必要性**: pip 在解析 `requirements.txt` 和 `pyproject.toml` 中的 `[build-system] requires` 时都会检查版本。即使 `requirements.txt` 已修复，`pyproject.toml` 中的构建依赖版本仍然不匹配会导致构建失败。两个文件都必须修复。
>
> **`vllm_ascend.__version__`**: 部分版本的 vllm-ascend 模块不对外暴露 `__version__` 属性，调用 `vllm_ascend.__version__` 会得到 `AttributeError`。这是正常的，不代表安装失败。可通过 `pip show vllm_ascend` 确认版本。
>
> **triton-ascend 额外索引**: 如果 PyPI 上找不到 triton-ascend，可尝试使用华为社区镜像：
>
> ```bash
> pip install triton-ascend==3.2.0 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
> ```

### 2.2.4 安装 Triton-Ascend（可选）

> 执行时会询问用户是否安装，输入 `y/Y` 确认后执行。

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

conda activate ${CONDA_ENV_NAME}
pip install ./triton_ascend-${TRITON_ASCEND_VERSION}-cp${PYTHON_VERSION_NO_DOT}-*-manylinux_2_27_aarch64.whl

# 修复 numpy 版本
pip install numpy==${NUMPY_VERSION}
```

---

## 2.3 GPU 平台环境安装

### 2.3.1 检查 GPU 环境

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

# 检查 NVIDIA 驱动和 GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 2.3.2 安装 Conda 环境

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

# 创建 conda 环境
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
conda activate ${CONDA_ENV_NAME}
```

### 2.3.3 安装 vllm

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

conda activate ${CONDA_ENV_NAME}

# 安装 vllm
pip install vllm==${VLLM_VERSION}

# 安装 deepgemm（vllm GPU 性能优化库，可选）
bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)

# 验证
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2.3.4 已知问题

**问题: PyTorch 下载 SSL 证书失败**

```bash
# 解决方案: 使用 whl 文件手动解压安装
python3 -c "
import zipfile, site
whl = '${WORKSPACE}/torch-${TORCH_VERSION}+cu124-cp${PYTHON_VERSION_NO_DOT}-linux_x86_64.whl'
sp = site.getsitepackages()[0]
with zipfile.ZipFile(whl) as z:
    z.extractall(sp)
"
```

**问题: vllm 安装超时**

```bash
# 解决方案: 增加超时, 换源
pip install --default-timeout=300 vllm==${VLLM_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2.4 安装测试工具（可选）

> 执行时会询问用户是否安装，输入 `y/Y` 确认后执行。

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

conda activate ${CONDA_ENV_NAME}

# aisbench（精度测试工具）
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip3 install -e ./ --use-pep517
cd ${WORKSPACE}

# 额外依赖（数学验证等）
pip install math_verify latex2sympy2_extended

# 验证
aisbench --version
```

---

## 2.5 准备模型文件

```bash
# 检查模型目录
ls -la ${MODEL_PATH}/../

# 如果模型不在指定路径，创建软链接
# ln -s /actual/model/path ${MODEL_PATH}
```

---

## 2.6 环境验证

```bash
# 激活环境（NPU 平台必须先 source CANN 环境）
source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
conda activate ${CONDA_ENV_NAME}

# Python 版本
python --version

# 关键包版本
pip list | grep -E "torch|vllm|triton|ascend"

# NPU 验证（必须 source CANN set_env.sh 后执行）
python -c "import torch_npu; print(f'NPU count: {torch_npu.npu.device_count()}')"

# vllm 验证
python -c "import vllm; print(f'vllm: {vllm.__version__}')"

# vllm_ascend 验证（注意: 部分版本无 __version__ 属性，通过导模块路径确认）
python -c "
import vllm_ascend
print('vllm_ascend 导入成功')
print('路径:', vllm_ascend.__path__)
print('版本:', vllm_ascend.__version__ if hasattr(vllm_ascend, '__version__') else '(无 __version__，通过 pip show vllm_ascend 查看)')
" 2>&1

# GPU 验证
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

---

## 2.7 下载精度测试数据集（可选）

> 如果提供了 `${DATASET_NAME}`，执行时会询问用户是否下载，输入 `y/Y` 确认后执行。

```bash
# 配置代理（如需要）
export http_proxy=${PROXY_ADDRESS}
export https_proxy=${PROXY_ADDRESS}

conda activate ${CONDA_ENV_NAME}

# 查找 aisbench 包中的数据集配置 README
# 先从克隆的 benchmark 代码中查找
if [ -d "${WORKSPACE}/benchmark/benchmark/configs/datasets/${DATASET_NAME}" ]; then
    DATASET_README="${WORKSPACE}/benchmark/benchmark/configs/datasets/${DATASET_NAME}/README.md"
elif [ -d "${WORKSPACE}/benchmark" ]; then
    # 也检查 benchmark 子目录下
    DATASET_README=$(find ${WORKSPACE}/benchmark -path "*/datasets/${DATASET_NAME}/README.md" 2>/dev/null | head -1)
fi

if [ -f "${DATASET_README}" ]; then
    echo "找到数据集 ${DATASET_NAME} 的 README: ${DATASET_README}"
    echo "按照 README 说明下载数据集..."

    # 按行读取 README 中的下载说明并执行
    # 支持以下指令格式:
    #   wget <url> -O <path>
    #   mkdir -p <dir>
    #   tar -xf <file> -C <dir>
    #   unzip <file> -d <dir>
    #   mv <src> <dst>
    #   cp <src> <dst>
    #   pip install <pkg>
    #   # 注释行跳过
    #   空行跳过
    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过注释和空行
        case "$line" in
            \#*|"") continue ;;
        esac
        echo "执行: ${line}"
        eval "${line}" || echo "警告: 命令执行失败（可跳过）: ${line}"
    done < "${DATASET_README}"

    echo "数据集 ${DATASET_NAME} 下载完成"
else
    echo "未找到数据集 ${DATASET_NAME} 的 README 配置，跳过数据集下载"
    echo "查找路径: ${DATASET_README}"
fi
```

---

## 产出

- Conda 环境 ${CONDA_ENV_NAME} 已创建
- PyTorch + vllm + 其他依赖已安装
- 环境验证通过

## 完成

环境安装完成后，技能执行结束。
