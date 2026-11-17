---
name: environment-install
description: 深度学习环境自动化安装工具。用户提供 example.md 文档填写服务器信息与软件版本，技能自动解析文档、连接服务器并安装深度学习环境（Conda + 框架包），支持NPU（华为Ascend）和GPU（NVIDIA）两种平台。当用户提到环境安装、依赖安装、Conda环境创建、pip安装、NPU部署、GPU部署时使用此技能。
version: 0.2.0
domain: inference-model
function: deployment
tags:
  - environment
  - conda
  - deployment
---

# 深度学习环境自动化安装 (Environment Install)

通过SSH协议连接远程服务器，自动安装深度学习推理环境（Conda + 框架包）。支持**NPU（华为Ascend）**和**GPU（NVIDIA）**两种平台自动识别和适配。

## 使用前准备

用户需提供一个 **`example.md`** 文档，填写服务器连接信息和软件版本参数。技能启动后将自动读取该文档并提取所有必要参数。

## 执行流程

```text
Step 1: 解析 example.md → Step 2: 服务器连接 → Step 3: 环境安装 → Step 4: 可选步骤提示 → 完成
```

> **非root用户处理**: 优先使用 `echo ${SERVER_PASSWORD} | sudo -S bash -c '...'` 方式切换到 root 权限（无需交互输入密码）。如果用户没有 sudo 权限，则提示用户选择其他连接方式（如改工作目录到有写权限的路径）。
>
> **可选步骤交互**: 所有标记为（可选）的步骤，在核心环境安装完成后，逐个询问用户是否继续执行。用户输入 `y/Y` 则执行，否则跳过。

| 步骤 | 参考文件 | 产出 |
| ------ | ---------- | ------ |
| 0. 文档解析 | `example.md`（用户提供） | 提取所有配置参数 |
| 1. 服务器连接 | `references/step-01-connect.md` | SSH连接验证、平台检测、root检查 |
| 2. 环境安装 | `references/step-02-install.md` | 已安装的深度学习环境 |
| 3. 可选步骤提示 | `references/step-02-install.md` | 根据用户选择安装可选组件 |

## example.md 文档准备

用户请参考仓库中的 [example.md](./example.md) 模板文件填写服务器连接信息和软件版本参数，技能将自动解析该文档中的表格字段。

### 参数映射表

| 参数 | 变量名 | 必填 | 说明 |
| ------ | -------- | ------ | ------ |
| 服务器IP | SERVER_IP | 是 | SSH连接地址 |
| 用户名 | SERVER_USER | 是 | SSH登录用户 |
| 密码 | SERVER_PASSWORD | 是 | SSH登录密码 |
| 工作目录 | WORKSPACE | 是 | 服务器工作目录 |
| Conda环境名 | CONDA_ENV_NAME | 是 | Conda虚拟环境名 |
| 代理地址 | PROXY_ADDRESS | 否 | 网络代理，无代理时留空 |
| Conda路径 | CONDA_PATH | 否 | Conda安装路径，默认自动检测 |
| Python版本 | PYTHON_VERSION | 否 | 默认3.11 |
| 平台类型 | PLATFORM | 是 | GPU 或 NPU |
| PyTorch版本 | TORCH_VERSION | 否 | 从文档读取 |
| vllm版本 | VLLM_VERSION | 是 | 从文档读取 |
| **CANN安装包路径** | **CANN_LOCAL_PATH** | **是** | **NPU平台，服务器上 CANN 等安装包所在目录路径** |
| vllm-ascend commit | VLLM_ASCEND_COMMIT | 是 | NPU平台，从文档读取 |
| Triton-Ascend版本 | TRITON_ASCEND_VERSION | 否 | NPU平台，从文档读取 |
| NumPy版本 | NUMPY_VERSION | 否 | NPU平台，从文档读取 |
| **数据集名称** | **DATASET_NAME** | **否** | **aisbench精度测试用的数据集名** |

## 错误处理

任一步骤检测到不可恢复错误时：

1. 向用户输出失败消息
2. 停止后续步骤执行
3. 工作流终止

| 错误类型 | 处理策略 |
| --------- | --------- |
| example.md 未找到 | 提示用户提供文档 |
| 文档参数缺失 | 向用户确认缺失的参数值 |
| SSH连接失败 | 检查网络、密码、服务器状态。优先使用 paramiko 密码登录 + sudo -S 切换 root |
| SSH长时间命令超时 | 设置 SSH keepalive（`transport.set_keepalive(30)`）；对编译类长时间命令使用 `nohup` 后台运行 + 日志轮询 |
| 磁盘空间不足 | 安装前预检 `df -h /`；空间不足时清理 `pip cache purge && conda clean --all -y`；CANN 安装时设置 `TMPDIR` 到有空间的目录 |
| GitHub 不可达 | git clone 内置 3-5 次重试间隔；提示用户配置代理或使用离线包 |
| Conda创建失败 | 检查Conda路径/磁盘空间 |
| CANN 安装解压失败 | 检查 `TMPDIR` 设置，确保指向有 > 10GB 可用空间的分区 |
| pip安装失败 | 检查代理、网络、或换源 |
| vllm 构建失败 (`setuptools_rust`) | 预装 `pip install setuptools_rust wheel` |
| vllm 构建时 torch_npu 导入失败 | 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 或确保 CANN `set_env.sh` 已 source |
| vllm-ascend 依赖版本不存在 (`triton-ascend`) | 同时修复 `requirements.txt` 和 `pyproject.toml` 中的版本号；使用 `--no-build-isolation` |
| torch_npu 导入失败 (numpy/yaml) | 安装 torch/torch_npu 后补装 `pip install numpy pyyaml` |
| 平台检测失败 | 提示用户手动指定 `--platform npu/gpu` |

## 执行日志

每个步骤完成时，向用户输出执行状态，包括：

- 步骤名称
- 执行结果（成功/失败）
- 关键产物或错误信息
- 耗时

## 参考资源

### 安装指南

- **`references/step-01-connect.md`** — 服务器连接、平台检测、环境检查流程
- **`references/step-02-install.md`** — 深度学习环境安装详细步骤（CANN、Conda、vllm、aisbench、数据集）

### 自动化脚本

- **`scripts/install.py`** — Python 自动化安装脚本（paramiko + sudo -S）
