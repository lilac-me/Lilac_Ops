#!/bin/bash
#
# 深度学习环境自动化安装脚本 (通用版)
#
# 工作目录: ${WORKSPACE}
# Conda环境: ${CONDA_ENV_NAME}
# 平台: 自动检测(NPU/GPU)

set -e
set -o pipefail

# ── 配置 (从外部传入) ──
WORKSPACE="${WORKSPACE}"
CONDA_ENV_NAME="${CONDA_ENV_NAME}"
PYTHON_VERSION="${PYTHON_VERSION}"
PROXY_ADDRESS="${PROXY_ADDRESS}"
CONDA_PATH="${CONDA_PATH}"

# NPU specific
CANN_LOCAL_PATH="${CANN_LOCAL_PATH}"
VLLM_ASCEND_COMMIT="${VLLM_ASCEND_COMMIT}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION}"
NUMPY_VERSION="${NUMPY_VERSION}"

# GPU specific
VLLM_VERSION="${VLLM_VERSION}"

# Dataset
DATASET_NAME="${DATASET_NAME}"

LOG_FILE="${WORKSPACE}/install_$(date +%Y%m%d_%H%M%S).log"

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log_info() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1" | tee -a "${LOG_FILE}"; }
log_success() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') [${GREEN}SUCCESS${NC}] $1" | tee -a "${LOG_FILE}"; }
log_error() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') [${RED}ERROR${NC}] $1" | tee -a "${LOG_FILE}"; }

# ── 平台检测 ──
# 仅通过硬件管理命令检测 NPU/GPU 硬件是否存在，不检测任何 CANN 安装路径
detect_platform() {
    if command -v npu-smi &>/dev/null; then
        echo "NPU"; return
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU"; return
    fi
    echo "UNKNOWN"
}

proxy_on() {
    if [ -n "${PROXY_ADDRESS}" ]; then
        export http_proxy=${PROXY_ADDRESS}
        export https_proxy=${PROXY_ADDRESS}
    fi
}

# ── 可选步骤交互 ──
ask_user() {
    local step_name="$1"
    local prompt="是否安装/执行 ${step_name}？(y/n): "
    echo -e -n "${YELLOW}${prompt}${NC}"
    read -r user_input
    case "${user_input}" in
        y|Y) return 0 ;;
        *) log_info "用户跳过: ${step_name}"; return 1 ;;
    esac
}

main() {
    # 非 root 用户自动切换到 root
    if [ "$(whoami)" != "root" ]; then
        echo "当前用户非 root，切换到 root 权限执行"
        exec sudo su - -c "bash $0" "$@"
    fi

    echo -e "${BLUE}=== 深度学习环境自动化安装 ===${NC}"
    log_info "日志: ${LOG_FILE}"
    log_info "工作目录: ${WORKSPACE}"
    log_info "Conda环境: ${CONDA_ENV_NAME}"

    # Step 0: 检测平台
    PLATFORM=$(detect_platform)
    log_info "检测到平台: ${PLATFORM}"
    if [ "${PLATFORM}" = "UNKNOWN" ]; then
        log_error "无法检测平台类型，请手动指定"
        exit 1
    fi

    # Step 1: Conda环境
    proxy_on
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "创建Conda环境: ${CONDA_ENV_NAME}"
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    else
        log_info "Conda环境 ${CONDA_ENV_NAME} 已存在"
    fi
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    pip install --upgrade pip

    # Step 2: 安装依赖
    if [ "${PLATFORM}" = "NPU" ]; then
        log_info "NPU平台安装..."

        ASCEND_PATH="${WORKSPACE}/ascend"
        mkdir -p "${ASCEND_PATH}"

        # ─── 平台架构检测 ─────────────────────────────────────
        ARCH=$(uname -m)
        case "${ARCH}" in
            x86_64)  CANN_ARCH="x86_64" ;;
            aarch64) CANN_ARCH="aarch64" ;;
            *)
                log_error "不支持的服务器架构: ${ARCH}，CANN 仅支持 x86_64 或 aarch64"
                exit 1
                ;;
        esac
        log_info "服务器架构: ${ARCH} → 期望 CANN 包后缀: linux-${CANN_ARCH}"

        # ─── 检查目标路径是否已安装 CANN ─────────────────────
        CANN_INSTALLED=false
        if [ -f "${ASCEND_PATH}/ascend-toolkit/set_env.sh" ]; then
            CANN_INSTALLED=true
            EXISTING_CANN_VERSION=$(cat "${ASCEND_PATH}/ascend_toolkit_install.info" 2>/dev/null | grep "^version=" | cut -d= -f2)
            if [ -n "${EXISTING_CANN_VERSION}" ]; then
                log_info "检测到 ${ASCEND_PATH} 下已安装 CANN，版本: ${EXISTING_CANN_VERSION}"
            else
                log_info "检测到 ${ASCEND_PATH} 下已安装 CANN（版本信息未知）"
            fi
        fi

        # ─── 查找 CANN 安装包 ─────────────────────────────────
        CANN_PKG=$(find "${CANN_LOCAL_PATH}" -maxdepth 1 -name "Ascend-cann-toolkit_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)

        # ─── 安装决策 ─────────────────────────────────────────
        SKIP_CANN_TOOLKIT_INSTALL=false
        if [ "${CANN_INSTALLED}" = "true" ]; then
            if [ -z "${CANN_PKG}" ]; then
                log_info "CANN 已安装且未提供新的安装包，跳过 CANN toolkit 安装步骤"
                SKIP_CANN_TOOLKIT_INSTALL=true
            else
                log_info "检测到新的 CANN 安装包: $(basename ${CANN_PKG})，将覆盖安装（建议先手动卸载旧版本）"
            fi
        fi

        # ─── 架构匹配校验 + 安装 ─────────────────────────────
        if [ "${SKIP_CANN_TOOLKIT_INSTALL}" = "false" ]; then
            if [ -z "${CANN_PKG}" ]; then
                MISMATCH_PKG=$(find "${CANN_LOCAL_PATH}" -maxdepth 1 -name "Ascend-cann-toolkit_*_linux-*.run" 2>/dev/null | head -1)
                if [ -n "${MISMATCH_PKG}" ]; then
                    log_error "CANN 包架构不匹配: 服务器 ${ARCH} (期望 linux-${CANN_ARCH})，实际 $(basename ${MISMATCH_PKG})"
                else
                    log_error "未在 ${CANN_LOCAL_PATH} 下找到 CANN toolkit 安装包，请确认 CANN_LOCAL_PATH 参数指向正确的安装包目录"
                fi
                exit 1
            fi
            log_info "CANN 包架构校验通过: $(basename ${CANN_PKG})"
            chmod +x "${CANN_PKG}"
            "${CANN_PKG}" --full -q --install-path="${ASCEND_PATH}"
        fi

        # 加载 CANN 环境变量（安装 A3 Ops 和 NNAL 前必须执行）
        source "${ASCEND_PATH}/ascend-toolkit/set_env.sh"

        # 安装 CANN A3 Ops（按架构精确匹配）
        A3_PKG=$(find "${CANN_LOCAL_PATH}" -maxdepth 1 -name "Ascend-cann-A3-ops_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)
        if [ -f "${A3_PKG}" ]; then
            log_info "安装 CANN A3 Ops: ${A3_PKG}"
            chmod +x "${A3_PKG}"
            "${A3_PKG}" --install --quiet --install-path="${ASCEND_PATH}"
        else
            log_info "提示: 未找到 A3 Ops 包（Ascend-cann-A3-ops_*_linux-${CANN_ARCH}.run），跳过"
        fi

        # 安装 CANN NNAL（按架构精确匹配）
        NNAL_PKG=$(find "${CANN_LOCAL_PATH}" -maxdepth 1 -name "Ascend-cann-nnal_*_linux-${CANN_ARCH}.run" 2>/dev/null | head -1)
        if [ -f "${NNAL_PKG}" ]; then
            log_info "安装 CANN NNAL: ${NNAL_PKG}"
            chmod +x "${NNAL_PKG}"
            "${NNAL_PKG}" --install --quiet --install-path="${ASCEND_PATH}"
        else
            log_info "提示: 未找到 NNAL 包（Ascend-cann-nnal_*_linux-${CANN_ARCH}.run），跳过"
        fi

        proxy_on
        pip install torch==${TORCH_VERSION} 2>/dev/null || true
        proxy_on
        if [ -n "${VLLM_VERSION}" ]; then
            log_info "从源码安装 vllm: v${VLLM_VERSION}"
            git clone https://github.com/vllm-project/vllm.git -b v${VLLM_VERSION}
            cd vllm
            export VLLM_TARGET_DEVICE=empty
            pip install -r requirements/common.txt
            pip install -e . --no-build-isolation
            cd "${WORKSPACE}"
        fi
        proxy_on
        pip install vllm-ascend
    elif [ "${PLATFORM}" = "GPU" ]; then
        log_info "GPU平台安装..."
        proxy_on
        pip install vllm==${VLLM_VERSION}
    fi

    # Step 3: 可选 - 安装 Triton-Ascend（仅 NPU 平台）
    if [ "${PLATFORM}" = "NPU" ] && [ -n "${TRITON_ASCEND_VERSION}" ]; then
        if ask_user "Triton-Ascend"; then
            proxy_on
            pip install ./triton_ascend-${TRITON_ASCEND_VERSION}-cp${PYTHON_VERSION_NO_DOT}-*-manylinux_2_27_aarch64.whl
            pip install numpy==${NUMPY_VERSION}
        fi
    fi

    # Step 4: 可选 - 安装测试工具
    if ask_user "测试工具 (aisbench)"; then
        proxy_on
        log_info "安装 aisbench..."
        cd "${WORKSPACE}"
        git clone https://github.com/AISBench/benchmark.git
        cd benchmark
        pip3 install -e ./ --use-pep517
        cd "${WORKSPACE}"
        pip install math_verify latex2sympy2_extended
        log_success "aisbench 安装完成"
    fi

    # Step 5: 验证
    proxy_on
    python --version
    pip list | grep -E "torch|vllm"

    # Step 6: 可选 - 下载数据集
    if [ -n "${DATASET_NAME}" ]; then
        if ask_user "数据集 ${DATASET_NAME}"; then
            proxy_on
            log_info "开始下载数据集: ${DATASET_NAME}"

            # 在克隆的 benchmark 代码中查找数据集 README
            if [ -d "${WORKSPACE}/benchmark/benchmark/configs/datasets/${DATASET_NAME}" ]; then
                DATASET_README="${WORKSPACE}/benchmark/benchmark/configs/datasets/${DATASET_NAME}/README.md"
            elif [ -d "${WORKSPACE}/benchmark" ]; then
                DATASET_README=$(find "${WORKSPACE}/benchmark" -path "*/datasets/${DATASET_NAME}/README.md" 2>/dev/null | head -1)
            fi

            if [ -f "${DATASET_README}" ]; then
                log_info "找到数据集 README: ${DATASET_README}"
                while IFS= read -r line || [ -n "$line" ]; do
                    case "$line" in
                        \#*|"") continue ;;
                    esac
                    log_info "执行: ${line}"
                    eval "${line}" || log_info "警告: 命令执行失败（可跳过）: ${line}"
                done < "${DATASET_README}"
                log_success "数据集 ${DATASET_NAME} 下载完成"
            else
                log_info "未找到数据集 ${DATASET_NAME} 的 README，跳过数据集下载"
            fi
        fi
    fi

    echo -e "${GREEN}=== 安装完成！ ===${NC}"
    log_success "安装完成"
    log_info "激活: conda activate ${CONDA_ENV_NAME}"
}

main "$@"