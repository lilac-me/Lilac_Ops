# 执行前置与就绪判断

## 环境自检

进入阶段2前，在 head 和每个 worker 上分别确认；存在 HCCL/IPC producer 时也检查其实际运行环境：

```bash
which vllm
python -c "import torch, torch_npu, vllm"
```

若 `torch_npu` 报 `libhccl.so`、`libc_sec.so`、`libdrvdsmi_host.so` 等缺库，先 source 目标 CANN 的 `set_env.sh`，或补齐 driver/toolkit 的 `lib64`；不要静默替换 torch/CANN。

## 环境前置修改清单（含 PR 号）

在 baseline 启动前询问用户：“当前模型或验证特性是否有已知的环境前置修改？如有，请提供对应 PR 号/链接或修改说明。”已有明确记录时复用。同时结合目标模型与特性，判断是否需要源码补丁、PR 合入/回退、依赖版本调整、编译或环境配置修改。读取实际导入模块的路径、包版本、源码 commit、容器/环境身份和已有补丁，不能只检查旁边的 checkout。用户已提供的修复 PR 可作为核查线索，不作为所有模型都必须应用的清单。

向用户提供以下清单，并写入任务 `audit.md` 和最终 README；无修改时明确写“无需前置修改”，无法确认时写“待核实”。

| 仓库/组件 | PR 号与链接 | 修改原因与适用模型/特性 | 目标 commit/版本 | 当前状态 | 操作与涉及节点 | 核实证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 按实际组件填写 | 仓库名、PR 号、完整链接；无对应 PR 时注明原因 | 按实际源码和验证目标说明 | 可复现的版本或补丁身份 | 已包含 / 待应用 / 已应用 / 待回退 / 已回退 / 不适用 / 待核实 | 合入、回退、安装、重编译或配置调整；head/worker/producer | 源码位置、commit、patch 或探测日志 |

- PR 的仓库、内容、依赖及适用版本必须核实；不能只凭“已合并”断言现场已包含，也不能将已有等价修复重复应用。无法联网时以本地源码证据为准，PR 当前状态注明未核实。
- 需要回退时同时记录原 PR 与回退 PR/commit。没有对应 PR 的本地补丁，记录补丁文件与校验和；依赖或配置调整写明修改前后值，不能编造 PR 号。
- 必要修改完成后检查实际安装/导入位置及各节点一致性；需要重新编译或重启的，核实新代码已生效。待核实/未满足的必要前置项阻塞相关验证，不以试跑代替核查。应用修改仍遵循已有任务授权和失败诊断边界。
- 清单在 `manifest.json` 的 `environment_changes` 下记录 `assessment`（`required`/`not_required`/`unknown`）和 `items`；每项保存 `component`、`pr`、`reason`、`target_revision`、`status`、`action`、`nodes`、`evidence`。`pr` 保存可复制的仓库名、PR 号和完整链接，无 PR 时说明原因；状态表达实际执行结果，提出修改建议不等于已应用。

## 询问并固化 baseline 特殊配置

生成 baseline 最终命令前询问用户：

> baseline 拉起是否需要额外的启动参数、环境变量或配置文件？如有，请提供具体内容，后续特性验证会共同继承这些配置。

已有本轮明确答复时复用上下文；用户明确不需要时记录空配置。用户尚未答复则保持待补充，继续独立的只读检查，不提前拉起服务。此处询问的是模型运行所需配置，不要求用户设计 TP/DP/EP 等自动推导项。

将确认后的特殊配置叠加到 agent 生成的基础配置中，记录到 `manifest.json.baseline_special_config`：

- `confirmation`：`confirmed` 或 `pending`；
- `revision`：本轮配置版本，公共配置有变化时递增；
- `cli_args`：按 argv 保存额外参数，保留 JSON 的完整值；
- `environment`：环境变量键值；涉及凭据的值脱敏；
- `config_files`：配置文件路径、校验和、分发位置；
- `constraints`：用户说明的适用范围、不可覆盖项和节点差异。

明确不需要特殊配置时，`confirmation` 仍为 `confirmed`，其余集合为空；不能把字段缺失或 `pending` 当作用户已说明无需配置。完整 baseline 启动命令与环境准备命令照常归档到 `logs/launch_commands.txt`。

## Baseline 通过后交付启动脚本

baseline 的容量预检、服务健康、双接口推理，以及本轮 TP/DP/PP/EP、图模式和其他实际开启的特性全部验证通过后，立即交付可复现该 baseline 的 Bash 启动脚本。此步骤由 Agent 根据实际命令与证据导出；`run_baseline.py` 是 HTTP 验证客户端，不掌握服务端完整配置，不能仅凭其退出码或 `server_url` 推断启动参数。

### 导出来源与内容

- 以成功运行的完整服务启动 argv、launcher 状态、服务日志和 `baseline_special_config` 为来源，核对模型路径/版本、服务名、host/port、设备可见性、TP/PP/DP/EP、图模式、上下文长度、调度/内存配置、特性开关及配置文件。保留 JSON 参数为完整的单个 argv，不重新推导一套默认配置，不加入 baseline 未验证的特性。
- 优先复用 `vllm-ascend-serving/scripts/serve_start.py` 的实际生成命令。其运行目录的 `_serve.sh` 可能依赖外层进程注入的环境；复制前补齐实际 CANN/ATB 加载、conda 激活、工作目录及公共/节点专属环境变量，不能只复制末尾 `vllm serve` 一行。脚本最终使用 `exec vllm serve ...`，或原 baseline 已验证的原生 launcher；参数使用 Bash 数组或逐 argv shell 引号，正确保留空格、单引号与 JSON。
- 单机输出任务目录 `launch/baseline/start-baseline.sh`，说明应在实际宿主机、指定 conda 环境或指定容器内运行。容器内脚本不得标为可直接在控制机执行；附真实的容器进入/执行命令。保留验证时使用的绝对模型/配置路径、设备映射和工作目录，检查依赖文件在目标位置仍可读。
- 多机输出 `launch/baseline/start-head.sh` 和各 `start-worker-<index>.sh`，记录脚本与节点、容器/环境的对应关系和实际启动顺序。DP 保留每节点 local size、start rank、head 地址、RPC 端口和 worker 的 headless 配置；跨节点 PP/Ray/专用 launcher 按实际通过的启动方式导出所需脚本与协调步骤，不用 DP 命令替代。各节点环境和设备映射分别保存，不把 head 配置原样复制给 worker。
- 脚本包含 Bash shebang、必要的环境准备、有效工作目录及完整启动命令；运行命令说明日志重定向位置，避免覆盖 baseline 原始证据。设备、地址和路径填写已核实的实际值，不使用 `<model>`、省略号或待填参数冒充可用脚本。原验证需要外部凭据时从调用环境读取并检查所需变量，不能写入密码/token，也不能把 `<redacted>` 当作真实启动值。

### 检查与交付

1. 用 Bash 执行 `bash -n <script>` 检查语法，对照归档 argv 和实际环境核对全部参数、JSON、并行度、rank、端口、设备及特殊配置。目标 Bash 不可用时记录导出阻塞，不能声称语法检查已通过。生成脚本不自动重新启动服务、停止原服务或申请额外设备；语法/配置核对不代表已经用导出脚本完成实机重启验证。
2. 在下一特性开始前，向使用者展示完整脚本（多机逐节点展示）、目标执行环境、脚本路径和可复制运行命令。默认以前台方式运行，用户可通过明确的 `nohup bash ... > <new-log> 2>&1 &` 命令后台启动；说明复现时需先确认原服务已停止、设备及端口可用，不加入自动 kill 或清理。
3. 写入 `manifest.json.baseline_launch`：`status`（`exported` / `blocked`）、`config_revision`、`baseline_report`（通过证据路径）、`scripts`。每个脚本记录 `node`、`role`、`path`（相对任务目录）、`execution_context`（主机/容器/conda 与工作目录）、`run_command`、`sha256`。只有 baseline 已通过、脚本完整且语法/配置核对通过时才写 `exported`；缺少实际启动配置或必要文件时记录 `blocked` 与 `reason`，不臆造配置。README 生成器根据该字段展示脚本入口与运行方式。
4. 公共环境、并行参数或特殊配置变化后，旧脚本仅作为旧版本证据保留。新 baseline 重新通过后再导出对应配置版本，更新 manifest、README 与用户展示；不将后续权重更新降低内存利用率等特性专属差异回写成原 baseline。

## 后续特性继承规则

后续 routing、sleep/wake、pause/resume、HCCL、IPC、Checkpoint、CP/SP/PP 的服务启动和基线对照均从固化配置派生，包括重启和修复后复验。环境补丁持续保留，公共 CLI 参数、环境变量和配置文件保持一致；功能专用开关在此基础上叠加。仅服务参数传给服务；producer/client 继承适用于它们的运行环境，不直接接收 `vllm serve` 专用参数。

- `--additional-config`、`--compilation-config` 等 JSON 对象按嵌套键合并。新增互不冲突的键，保留原键；同键值相同只保留一份。同键值不同、数组/标量替换或互斥参数发生冲突时，展示冲突并询问用户；不能靠重复参数的解析顺序覆盖，也不能静默删除特殊配置。
- 特性有明确要求的差异（例如 HCCL/IPC 需降低显存利用率），若与用户固化值冲突，先说明原因并询问如何处理；记录该特性差异。公共配置或补丁改变后，停止使用旧 baseline，重新预检并采集对应基线；特性专用对照按实际配置采集，保证比较双方条件明确。
- 多机把公共变量通过 `serve_start.py --extra-env KEY=VALUE` 传入，节点专属变量按其 `--worker-extra-env` 等实际支持的入口传递；CLI 参数走 `--` 后透传，服务端口/设备映射等仍由 launcher 按节点处理。检查参数是否被过滤、环境变量是否真正传至容器/worker，不能以控制机 `export` 代替远端注入。
- 使用只接收 `--baseline-cmd` 的验证脚本时，传入包含公共特殊参数的固化命令；脚本实际不支持的环境或配置传递不能假装已继承，应补齐传递或改用已支持的启动入口，无法实现则记录相关项 `blocked`。本机 subprocess 启动需在其实际运行环境设置变量，不把 shell 赋值或重定向误当作可执行 argv。
- 每轮执行前对照公共配置和实际命令/环境，检查是否遗漏或覆盖；每项 JSON 报告记录 `baseline_config_revision`、`inherited_special_config` 和 `config_differences`，并保存实际生效证据。若脚本没有自动写入这些字段，由 agent 汇总时补录，不能仅凭“沿用 baseline”一句话认定继承成功。

## 就绪判断

服务就绪至少同时观察 `/health`、`/v1/models` 和启动日志。大模型加载与图编译可能很慢，应使用脚本的超时参数；一旦日志出现确定性启动失败，不要无期限轮询。
