---
name: bridge-remote-train-run
description: "This skill should be used when the user asks to run remote training validation for MindSpeed-LLM or MindSpeed-Bridge on a remote NPU server. It acts as a local controller that syncs repositories to the remote host, loads the CANN environment, launches the training script via conda or docker, and determines success from the locally stored validation report and training log. Use it when the user wants to plan, sync, preflight-check, or run a remote training script."
license: MIT
targets-repo:
  - mindspeed-bridge
  - mindspeed-llm
---

# MindSpeed-LLM / MindSpeed-Bridge 远端验证

将这个 skill 作为本地控制器使用。只在远端同步代码、加载环境并启动训练；在本地保存输出和判断脚本是否成功。不要在远端修改 task、阈值、history 或产品代码。

## 核心约定

- 要求用户提前配置 SSH 免密登录；始终使用 `BatchMode=yes`，不要交互式读取密码。
- 将服务器、仓库路径、训练脚本、运行器（Conda/Docker）和 CANN 信息放入私有配置。不要把密码、私钥内容或 token 写入配置和报告。
- 支持多个目录同步：`repositories` 数组中的每一项把 `local_path` 增量同步到 `remote_path`（远程完整路径，例如 `/home/user/testllm/MindSpeed-LLM`），不要把仓库内容错误地平铺到它的父目录。
- 训练脚本执行路径不受多目录同步影响：`execution.script_repo` 指定训练的工作目录仓库（默认第一个仓库），训练始终在该仓库的远端目录（容器模式为容器内映射目录）内执行；`execution.script` 是相对该工作目录的路径，可含 `..`（例如从 Megatron-Bridge 工作目录执行 `../MindSpeed-Bridge/xxx.sh`）。
- 默认使用增量 `rsync` 且不删除远端文件。只有配置明确设置 `sync.delete: true` 时才启用删除，并先确认目标是具体仓库目录。**首次同步建议直接开 `sync.delete: true`**，以本地为唯一源，一次性清掉远端残留，确保远端 == 本地；
- **Windows 换行符**：Windows 仓库 `core.autocrlf=true` 会把工作区 .sh 检出为 CRLF，rsync 到 Linux 后 `bash` 报 `$'\r'` 类错误。
  同步前将仓库 `core.autocrlf` 置为`false`、把将被执行的 .sh 归一化为 LF，并 `git add --renormalize` 使工作区与 index 一致（避免行尾差异污染门禁的 scope 检查）。
- rsync 默认不保留源端属主/属组（追加 `--no-o --no-g`），避免 root 同步后在远端产生不存在的 UID/GID 属主；如需保留，设置 `sync.preserve_owner_group: true`。
- 运行器（`execution.runner`）：
  - `conda`（默认）：要求远端用户已经通过 `conda init bash` 或等效 shell 配置，使交互式 Bash 能直接执行 `conda`；在 `conda activate <conda_env>` 之后 source CANN 脚本。
  - `docker`：通过 `docker exec <container> bash -lc` 在容器内 source CANN 脚本（容器内路径）并执行训练；`execution.docker.mounts` 可选，声明期望的 host→container 映射（支持父级映射，如 `/home`→`/home`），预检用 `docker inspect` 按路径前缀核对容器实际挂载，并据此推导训练仓库在容器内的执行目录；不配置时默认容器路径与远端路径相同。容器内执行目录不可达（`cd` 失败）时预检不通过，不会启动训练。
- 启动训练前必须执行 `npu-smi info`（宿主机）。检测到任何 NPU 进程时立即退出且不启动训练；命令失败或无法确认空闲时同样保守退出。该门禁不可通过配置关闭。
- 使用 `bash -e -o pipefail` 执行训练脚本，确保普通命令或管道中的训练失败能够传递为非零退出码。
- 默认在任务工作目录创建`remote-train-run`文件夹作为工作区，将每次训练尝试写入工作区的 `output/remote-validation-<四位数字>/`（即`model_migration/output/`），例如 `remote-validation-0001`、`remote-validation-0002`，可用 `--output-dir` 覆盖。
  扫描已有目录的最大编号后严格递增；失败和超时也保留对应目录及完整日志，不复用旧编号。
  `plan`/`sync`/`preflight` 等非训练模式写入 `output/.<mode>/`。
- 如果是由`$bridge-migration-workflow`创建的工作区，`remote-train-run`已包含在`$bridge-migration-workflow`的工作区目录中。
- 当前版本只根据远端脚本 exit code 判断运行成功；loss、性能和显存 validator 后续在本地分析层实现。

## 配置

复制 `config/remote_config.example.json` 为以 `.local.json` 结尾的私有文件，再按实际环境修改。

配置字段和约束见 `references/remote_config_schema.json`。要点：

- `repositories`：非空数组，每项 `local_path` + `remote_path`，同步模式会逐个创建远端目录并 rsync。
- `execution`：`plan`/`preflight`/`run` 模式必需；`sync` 模式不需要。
  - `script` 为相对 `script_repo` 工作目录的训练命令参数（可含 `..`，例如 `../MindSpeed-Bridge/.../xxx.sh`）；`script_repo` 可填 local_path 或 remote_path。
  - `runner`：`conda` 或 `docker`。conda 模式需要 `conda_env`；docker 模式需要 `docker.container`，可选 `docker.mounts`（`host` 为远端宿主机路径，`container` 为容器内路径；支持父级映射如 `/home`→`/home`，也支持逐仓库映射；不配置时默认容器路径与远端路径相同）。
  - `cann_env_scripts`：conda 模式为宿主机路径，docker 模式为容器内路径。
- `sync.preserve_owner_group`：默认 `false`（不保留属主/属组）；`true` 时保留。
- `sync.delete`：默认 `false`；`true` 时对每个仓库 rsync 追加 `--delete`（远端删除本地不存在的
  文件），命令行 `--sync-delete` 可覆盖配置。开启时脚本要求每个 `remote_path` 至少三级目录。
- 命令行中的 `--output-dir`（兼容别名 `--artifact-root`）、`--host`、`--user`、`--port`、`--repo-root`（作用于第一个仓库）、`--remote-repo`（作用于第一个仓库）、`--script`、`--conda-env` 和重复的 `--cann-env-script` 会覆盖配置文件。

## 文件职责

- 使用 `config/remote_config.example.json` 作为可复制修改的远端验证配置示例。
- 使用 `references/remote_config_schema.json` 定义输入配置的字段、类型和约束，作为配置接口合同。
- 使用 `references/validation_report_schema.json` 定义验证报告的输出格式，作为总控 Skill 和其他程序稳定解析报告的接口合同。

## 工作流

1. 先运行 `plan`，核对本地路径、远端目标、每个仓库的同步命令和将要执行的训练命令（不执行任何远端操作）。
2. 运行 `sync`，逐个创建远端目录并 rsync 每个仓库（不加载环境、不启动训练），用于验证多目录同步。
   **新任务首次同步**建议在配置（或 `--sync-delete`）打开 `delete`，先确认远端目标目录可安全清空，
   再执行；同步后用 `plan` 核对命令、用远端 `git status` 抽查确认两端一致。
3. 运行 `preflight`，验证 SSH、远端仓库、运行器（Conda 环境 / 容器存在性与挂载）、CANN 脚本和训练脚本，但不启动训练。
4. 运行 `run`，同步代码；脚本会先通过 `npu-smi info` 确认所有 NPU 均无运行进程，再启动训练。
5. 读取本地 `validation_report.json` 和 `training.log`；不要只依赖终端截断输出。
6. 如果报告显示远端训练失败，保留原始日志并交给总控 skill 诊断，不要通过修改验证环境掩盖失败。

## 脚本入口

```bash
python3 scripts/run_validation.py --config config/remote_config.local.json --mode plan

python3 scripts/run_validation.py --config config/remote_config.local.json --mode sync

python3 scripts/run_validation.py --config config/remote_config.local.json --mode preflight

python3 scripts/run_validation.py --config config/remote_config.local.json --mode run

python3 scripts/run_validation.py --config config/remote_config.local.json --mode run --output-dir /path/to/custom/output
```

## 输出与退出码

报告格式见 `references/validation_report_schema.json`。本地 artifact 目录至少包含：

- `validation_report.json`
- `training.log`（进入训练阶段后，保存远端训练的完整 stdout 与 stderr）

退出码：`0` 表示对应模式成功；`2` 表示配置或本地输入非法；`3` 表示 SSH、同步、远端预检或 NPU 状态检查失败；`4` 表示训练脚本失败；`5` 表示 NPU 已被进程占用、训练未启动；`124` 表示训练超时。
