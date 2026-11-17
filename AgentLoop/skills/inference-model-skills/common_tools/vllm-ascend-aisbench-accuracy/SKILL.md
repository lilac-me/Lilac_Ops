---
name: vllm-ascend-aisbench-accuracy
description: >-
  This skill should be used when the user asks to "测一下精度", "精度测试",
  "统计精度", "跑一下 accuracy", "aisbench 评测", "accuracy test",
  "evaluate accuracy", or wants to run AISBench accuracy evaluation
  on a vLLM Ascend online service. Orchestrates environment setup,
  service startup, multi-round evaluation, and report generation.
version: 1.0.1
domain: common
function: common
tags:
  - vllm-ascend
  - aisbench
  - accuracy
  - evaluation
author: Liccol
hooks:
  PreToolUse:

    # --- JSON 双引号转义 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(compilation_config|additional_config|profiler.config)"
      prompt: |
        ⚠️ JSON 参数中的双引号必须转义：PowerShell 用 `\"`，Git Bash 用 `\\"`。
targets-repo:
  - vllm-ascend
---

# vllm-ascend-aisbench-accuracy

## 功能概述

在当前本地环境、远程容器或 conda 环境上运行 AISBench 精度评测，支持单机与多机（DP 多节点）部署。自动完成网络预检、环境准备（ais-bench-benchmark 安装、数据集下载、配置修改）、服务启动、多轮精度测试（默认 10 轮取平均）、以及测试报告生成。

- 结果输出在 stdout（JSON），包含每轮精度指标和汇总统计
- 本地报告保存在 `.workspace-local/aisbench-accuracy/`
- 资源预检：先检查各节点已安装的 AISBench、评测依赖和本地数据集；全部齐全时跳过外网预检。仅在资源不完整时检查下载端点连通性，不可达时提示用户配置 `http_proxy` / `https_proxy` 后重试

## 使用方式

- 直接调用：`/mindspeed-agent:vllm-ascend-aisbench-accuracy`
- 描述精度测试需求，由 Claude 自动匹配调用

## 操作步骤

### 步骤一：确认参数

需确认的参数包括：

| 参数 | 说明 |
| ------ | ------ |
| 个人工号 | `--employee-id <工号>`（必填） |
| 目标 | `--mode local`（当前环境直调）、`--host <ip> [--mode docker] --container <name>`（Docker exec）、`--host <容器IP> --mode direct --password <pwd>`（直连容器SSH）、`--host <ip> --mode conda --conda-env <name>`（Conda） |
| 连接模式 | `--mode {docker,conda,direct,local}`（默认 `docker`）：统一选择连接拓扑，透传给底层 serving；`local` 不使用 SSH |
| CANN 路径 | `--cann-dir <path>`（默认 `/usr/local/Ascend`），执行前必须确认 |
| 模型 | `--model <remote-path>` |
| TP / DP | `--tp <N> --dp <N>` |
| 设备 | `--extra-env ASCEND_RT_VISIBLE_DEVICES=<ids>`（可选） |
| 轮数 | `--runs <N>`（默认 10） |
| 数据集 | `--dataset <name>`（默认 `math500_gen_0_shot_cot_chat_prompt`）；textvqa 场景用 `--dataset textvqa_gen` |
| 数据集种类 | `--dataset-kind {auto,math,textvqa}`（默认 `auto`，按 `--dataset` 名自动推断：含 `textvqa` 走 textvqa 准备，否则走 math） |
| batch_size | `--batch-size <N>`（默认 1024） |
| temperature | `--temperature <N>`（默认 0） |
| seed | `--seed <N>`（默认 1234） |
| 服务参数 | `--serve-args` 后的参数必须裸参数展开，如 `--serve-args --max-model-len 4096`；禁止用外层引号把整组参数包成一个字符串 |
| HTTP 代理 | `--http-proxy <url>`（可选，远端 AISBench 仓库克隆、pip 安装和数据下载（含 ModelScope CLI）走代理，如 `http://host:port`，可配 `--https-proxy`、`--no-proxy`） |
| 多机 Worker | `--worker-hosts <csv>`（多机部署时必填，逗号分隔 worker IP） |
| 多机参数 | `--worker-password`、`--worker-container`、`--worker-conda-env` 等（可选） |

`--serve-args` 是参数分隔标记，其后的每个 vLLM 参数必须作为独立命令行元素传入。禁止用外层引号包住整组参数，例如禁止 `--serve-args "--max-model-len 4096 --generation-config vllm"`；应写为 `--serve-args --max-model-len 4096 --generation-config vllm`。JSON 或含空格的单个参数值仍可单独使用引号。

### 步骤二：执行精度测试

执行前读取[工作目录与归属](../../workspace-ownership.md)，通过 `--employee-id <工号>` 指定本次任务的使用者。该参数用于服务运行目录和 AISBench 准备日志，不改变 AISBench 客户端、仓库和数据目录。

AISBench 仓库 `/home/vllm/benchmark/`、TextVQA 数据目录及媒体白名单 `/home/vllm/benchmark/ais_bench/datasets/textvqa` 沿用环境内固定、可共享复用的路径约定，不随工号变化。Docker 模式下这些是容器内路径，不代表跨容器自动共享存储，也不作为服务使用者的归属依据；服务归属看 `/home/<工号>/serving/<实例>/`。

```bash
# local 模式 — 在当前服务器或容器环境直接执行，不需要 --host
python3 skills/inference-model-skills/common_tools/vllm-ascend-aisbench-accuracy/scripts/accuracy_run.py --employee-id <工号> \
  --mode local \
  --model <model-path> --tp 4 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
  --runs 1 \
  --serve-args --max-model-len 22528

# docker 模式 — SSH 宿主机, docker exec 进容器（默认）
python3 skills/inference-model-skills/common_tools/vllm-ascend-aisbench-accuracy/scripts/accuracy_run.py --employee-id <工号> \
  --host <ip> --container <name> --key-file <path> \
  --model <remote-path> --tp 2 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  --runs 10 \
  --serve-args --max-model-len 4096

# direct 模式 — SSH 直连容器/虚拟机 IP
python3 skills/inference-model-skills/common_tools/vllm-ascend-aisbench-accuracy/scripts/accuracy_run.py --employee-id <工号> \
  --mode direct --host <容器IP> --password <pwd> \
  --model <remote-path> --tp 2 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  --runs 10 \
  --serve-args --max-model-len 4096

# conda 模式 — SSH 宿主机 + conda 环境
python3 skills/inference-model-skills/common_tools/vllm-ascend-aisbench-accuracy/scripts/accuracy_run.py --employee-id <工号> \
  --mode conda --host <ip> --conda-env <env-name> --key-file <path> \
  --model <remote-path> --tp 2 --runs 10

# 指定数据集和采样参数
python3 .../accuracy_run.py --host <ip> --container <name> --employee-id <工号> \
  --model <remote-path> --tp 4 --runs 5 \
  --dataset math500_gen_0_shot_cot_chat_prompt \
  --batch-size 1024 --temperature 0 --seed 1234

# textvqa 多模态精度测试（数据准备自动：ModelScope 下载 + 绝对路径化与完整性检查）
python3 .../accuracy_run.py --host <ip> --container <name> --employee-id <工号> \
  --model <remote-path> --tp 2 --runs 10 \
  --dataset textvqa_gen \
  --batch-size 1024 --temperature 0

# 远端需通过代理访问外网（AISBench 仓库克隆 / pip 安装 / ModelScope 等数据下载）
python3 .../accuracy_run.py --host <ip> --container <name> --employee-id <工号> \
  --model <remote-path> --tp 2 \
  --http-proxy http://proxy.host:3128 \
  --https-proxy http://proxy.host:3128 \
  --no-proxy localhost,127.0.0.1,<内网网段> \
  --runs 10

# ===== 多机部署 =====
# docker 模式 — 两节点
python3 .../accuracy_run.py --host 192.168.1.100 --container my-container --employee-id <工号> \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B --tp 8 --runs 10

# direct 模式 — 两容器直连
python3 .../accuracy_run.py --employee-id <工号> \
  --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-hosts 10.0.0.6 --worker-password wkpwd \
  --model /data/models/Qwen2-72B --tp 8 --runs 10

# conda 模式 — 两节点
python3 .../accuracy_run.py --employee-id <工号> \
  --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-hosts 192.168.1.101 --worker-conda-env vllm_env \
  --model /data/models/Qwen2-72B --tp 8 --runs 10
```

> **多机精度测试行为**：脚本先**逐节点完成环境准备**（ais-bench 安装、数据集下载、配置修改），再通过 serve_start 启动多机集群，然后对 head 节点执行多轮 `ais_bench` 精度评测，完成后停止整个集群。Worker 参数（`--worker-*`）透传给底层 serving。
>
> - **代理使用范围**：`--http-proxy` / `--https-proxy` / `--no-proxy` 在远端网络预检和环境准备时分别导出为 `http_proxy` / `https_proxy` / `no_proxy` 及其大写形式。AISBench 仓库克隆、pip 安装与 `modelscope download` 子进程继承同一组变量；TextVQA 的下载代码不需要单独增加代理参数。ModelScope 使用 HTTPS，需代理时配置 `--https-proxy`，并避免 `--no-proxy` 将其下载域名排除。执行实际 `ais_bench` 精度测试前，脚本会取消本次参数设置的代理变量（含大写形式），避免评测请求经过准备阶段的代理。
> - **本地模式**：`--mode local` 直接在当前进程环境执行命令，不建立 SSH 连接，不接受 `--container`、`--conda-env`、SSH 认证参数或多机 Worker 参数。
>
> Windows 下用 **PowerShell** + `py -3`。Git Bash 用户需加 `MSYS_NO_PATHCONV=1`。

### 步骤三：脚本自动执行流程

`accuracy_run.py` 按以下顺序自动编排：

0. **资源与网络预检**：先对 head + 所有 worker 节点检查 AISBench 包和目录、评测依赖及本地数据集。全部齐全时跳过外网检查；存在缺失项时才检查 ais-bench git 仓库和数据集端点，不可达时输出 `needs_network_proxy` 并中断
1. **环境准备**（**逐节点执行**：head 及每个 worker）：
   - 检查 `pip list | grep ais-bench-benchmark` 是否已安装
   - 若未安装：`git clone` benchmark 仓库到 `/home/vllm/benchmark/`，`pip3 install -e ./`
   - math-eval 依赖先尝试 import，只有缺失时才执行 pip 安装
   - 通过 `pip show ais-bench-benchmark` 找到安装路径
   - 按 `--dataset-kind` 准备数据集：
     - `math`：下载 OpenCompass math 数据集到 `datasets/`（如不存在）。
     - `textvqa`：使用 `--dataset textvqa_gen`；通过 `modelscope download --dataset vllm-ascend/textvqa_subset --local_dir /home/vllm/benchmark/ais_bench/datasets/textvqa` 下载已整理的 `textvqa_json/` 和 `train_images/`，改写图片绝对路径并检查引用图片完整性。不再依赖 HuggingFace、git-lfs 或 resolve 下载。
   - 用真实服务地址修改配置文件（`path`、`host_port`、`batch_size`、`generation_kwargs`）
   - Conda 模式下自动激活对应节点的 conda 环境（worker 用 `--worker-conda-env`，缺省继承 head）
2. **启动服务**：调用 `serve_start.py` 启动 vLLM 服务（多机时启动整个集群），轮询等待就绪，获取服务地址
3. **精度测试**：在 head 节点循环执行 `ais_bench` 命令（默认 10 次），逐轮记录指标
4. **停止服务**：调用 `serve_stop.py` 停止服务（多机时停止整个集群）
5. **汇总报告**：计算均值/标准差，生成本地 Markdown 报告

**资源预检、网络预检与代理询问流程**：

- 执行开始先检查每个节点的现有 AISBench、评测依赖和数据集；全部齐全时网络状态记录为 `skipped/local_resources_ready`，直接进入环境准备
- 只有资源不完整时才检查外网；任一节点无法访问 git 仓库或数据集时，脚本返回 `status: needs_network_proxy`（附资源缺失项和各节点网络明细），**不会**继续启动服务
- 此时应向用户确认：是否需要为远端配置 `HTTP_PROXY` / `HTTPS_PROXY`。若用户愿意配置，拿到代理地址后带上 `--http-proxy` / `--https-proxy`（及可选的 `--no-proxy`）重新执行，脚本会先把代理导入远端环境再做 AISBench 仓库克隆、pip 安装和 ModelScope 等数据下载
- 若用户确认无需代理且网络不可达，则提示环境准备阶段必然失败，建议先解决网络后再执行

### 步骤四：解读结果

单轮输出示例：

```json
{
  "status": "ok",
  "round": 3,
  "metrics": {
    "accuracy": 0.856,
    "total_samples": 500,
    "correct_samples": 428
  }
}
```

最终汇总输出包含 `aggregated` 字段（均值、标准差）和每轮明细。

### 步骤五：生成本地报告

精度测试完成后，脚本自动将结果整理为报告并保存在本地 `.workspace-local/aisbench-accuracy/` 目录下。

**报告内容**：

1. **执行概要**：执行时间、目标机器、模型、TP/DP 配置、数据集、采样参数
2. **精度结果**：以表格形式列出每轮精度和最终平均精度
3. **统计信息**：均值、标准差、最低/最高精度
4. **结论**：精度是否达标

**报告文件命名**：`<YYYY-MM-DDTHH-mm-ssZ>_<host>_accuracy_report.md`

## 注意事项

- **不要手动构造 SSH 命令或 `ais_bench` 命令**，所有操作通过 `accuracy_run.py` 执行
- 精度测试需要在服务端已启动的情况下进行，脚本自动管理服务生命周期
- `ais_bench` 命令依赖配置文件 `benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`，脚本会自动修改其中的 `path`（模型路径）、`host_port`（服务端口）和 `host_ip=127.0.0.1`；显式使用 IPv4 loopback，避免 `localhost` 仅解析到 `::1` 时连接失败
- 远端需有 `pip` / `pip3` 可用，用于检查 AISBench 安装状态及按需安装依赖；仅在 AISBench 未安装且 `/home/vllm/benchmark/` 仓库不存在、需要克隆仓库时要求 `git`。TextVQA 的 ModelScope CLI 下载不依赖 `git` 或 `git-lfs`；数据缓存不完整且缺少 CLI 时通过 `pip3` 安装 `modelscope`。
- **数据集下载/解压健壮性**：math 下载优先用 `wget`，缺失时回退 `curl`，两者都缺失则报错；`unzip` 缺失时自动回退 `python3 zipfile` 解压（python3 必然存在）。下载后校验 `math.zip` 非空、解压后校验 `math/` 目录存在，任一环节失败都会报明确错误并中断
- **textvqa 场景**：使用 ModelScope 数据集 `vllm-ascend/textvqa_subset`，远端需能访问 ModelScope。缺少 CLI 时自动安装 `modelscope`；下载时以 `--local_dir` 将 `textvqa_json/` 与 `train_images/` 直接放到 AISBench 数据集目录的 `textvqa/` 下。准备阶段先检查现有验证集与引用图片的完整性，不完整则重新调用 ModelScope 下载；随后改写图片绝对路径并再次校验，仍缺失或标注无效则报错。服务自动注入 `--allowed-local-media-path /home/vllm/benchmark/ais_bench/datasets/textvqa`（用户显式指定时不重复）。评测命令为 `ais_bench --models vllm_api_stream_chat --datasets textvqa_gen`。
- 跨平台：macOS/Linux 用 `python3`，Windows 用 **PowerShell** + `py -3`
- **⚠️ CANN 环境**：执行前需确认远端 CANN 安装根目录（默认 `/usr/local/Ascend`）
- **网络预检**：资源不完整时才检查各节点的 AISBench 仓库和所选数据集端点；失败时输出 `needs_network_proxy` 并中断，应询问用户是否需要配置 `http_proxy` / `https_proxy`，确认后代入 `--http-proxy` / `--https-proxy` / `--no-proxy` 重跑即可。预检端点可达不保证后续依赖安装及图片 CDN 下载均成功，下载失败时查看环境准备日志。
- **多机**：环境准备在 head 与每个 worker 上分别执行（各节点需能访问外网下载 ais-bench 与数据集）；精度评测在 head 节点执行，工作负载由多机集群承载
- 远端日志：每个节点的环境准备（ais-bench 安装、数据集下载、配置修改）输出会实时 tee 到该节点容器的/home/<工号>/aisbench/prep_<时间戳>_<节点>.log，排障可 tail -f 该文件；vLLM 服务日志在 /home/<工号>/serving/<实例>/stdout.log 与 stderr.log

## 参考信息

- 依赖 skill：`vllm-ascend-serving`（服务启动/停止）
- 被测工具：[AISBench benchmark](https://github.com/AISBench/benchmark)
- 评测数据集：默认使用 math500（OpenCompass math.zip）；多模态场景支持 textvqa（ModelScope: vllm-ascend/textvqa_subset），`--dataset textvqa_gen`
