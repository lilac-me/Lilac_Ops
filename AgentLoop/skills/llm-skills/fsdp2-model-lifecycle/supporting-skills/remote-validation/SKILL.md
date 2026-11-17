---
name: mindspeed-llm-remote-validation
description: 将本地 MindSpeed-LLM worktree 和可选验证 runtime 同步到用户指定的免密 SSH 服务器，检查 NPU 空闲状态后在指定 Conda 与 CANN 环境中启动训练或精度验证入口，并回收远端报告、日志和结构化状态。适用于 FSDP2、mcore 等后端的远端训练、logits 精度验证、环境预检、同步执行和结果回收。
---

# MindSpeed-LLM 远端验证

将这个 skill 作为本地控制器使用。只在远端同步代码和只读 runtime、加载环境并启动一个仓库内 launcher；在本地保存输出和判断脚本是否成功。不要在远端修改 task、阈值、history 或产品代码。

公共入口只有 `scripts/run_validation.py`。正常执行不要读取其实现；根据本文件、配置 schema 和 `--help` 使用 `plan`、`preflight`、`run`。远端 precision runtime 的公共入口仍只有 `run_alignment.py`，模型仓库中的 `run_complete_alignment.sh` 只是传递模型专用参数的薄 launcher。

## 核心约定

- 要求用户提前配置 SSH 免密登录；始终使用 `BatchMode=yes`，不要交互式读取密码。
- 要求远端用户已经通过 `conda init bash` 或等效 shell 配置，使交互式 Bash 能直接执行 `conda`；不要在配置中绑定 Conda 安装路径。
- 将服务器、仓库路径、训练脚本、Conda 和 CANN 信息放入私有配置。不要把密码、私钥内容或 token 写入配置和报告。
- 服务器、Conda 和 CANN 等执行环境配置即使已存在于 `*.local.json`，也必须先在对话中取得用户显式确认，再用于同步、预检或启动任务；配置文件存在本身不构成使用授权，配置变更后须重新确认。
- 将 `remote_repo` 视为远端仓库的完整路径，例如 `/home/user/testllm/MindSpeed-LLM`，不要把仓库内容错误地平铺到它的父目录。
- 默认使用增量 `rsync` 且不删除远端文件。只有配置明确设置 `sync.delete: true` 时才启用删除，并先确认目标是具体仓库目录。
- 在 `run` 模式下依次执行：创建远端仓库目录、同步 worktree、同步可选 runtime bundles、环境预检、NPU 空闲门禁、执行 launcher、回收指定产物、报告落盘。
- 启动远端任务前必须执行 `npu-smi info`。检测到任何 NPU 进程时立即退出且不启动任务；命令失败或无法确认空闲时同样保守退出。该门禁不可通过配置关闭。
- 使用 `bash -e -o pipefail` 执行 launcher，确保普通命令或管道中的失败能够传递为非零退出码。
- 静默看门狗：远端进程死亡但 ssh 通道保持打开时输出会永久静默，`stream_training` 在 `execution.silence_timeout_seconds`（默认 900，0 表示关闭）内无任何新输出即判定挂死并终止，报告 `stage: stalled`。不要依赖远端 launcher 的硬超时兜底。
- 精度任务启动前检查远端产物目录：precision-alignment 的 `{run_id}` 产物目录已存在且非空时直接失败（`stage: artifact_dir_conflict`），避免远端 FileExistsError 白跑一轮，更避免读到旧目录的过期数据。训练任务共享日志目录，不做此检查。
- 不同步本地 `.git` 或 worktree gitfile；在报告和远端环境中使用本地计算的 `<base-commit>-dirty` revision，避免把未提交接入代码误报成基线提交。
- `sync.runtime_bundles` 将精度脚本等只读 runtime 同步到远端仓库内的隔离子目录；可用 `include` 只同步公开合同声明的运行文件。默认删除该 runtime 目标中的旧文件，但不删除产品仓库其他内容。
- `execution.environment` 的值可使用 `{remote_repo}` 和 `{run_id}`，用于向薄 launcher 传入远端权重、两个 Python、runtime 和唯一输出目录。
- 配置 `artifacts.remote_path` 时，在执行成功或失败后回收匹配的 JSON、Markdown 和日志；精度配置默认不回收大型 `.pt` logits。
- 默认将每次远端任务尝试写入当前 Skill 集合根目录的 `output/remote-validation-<四位数字>/`，例如 `remote-validation-0001`、`remote-validation-0002`。扫描已有目录的最大编号后严格递增；失败和超时也保留对应目录及完整日志，不复用旧编号。
- 当前版本保留远端 launcher 的原始 exit code 到 `execution_result`；loss、性能和显存 validator 仍由本地分析层实现。

## 配置

训练验证复制 `config/remote_config.example.json`；精度验证复制 `config/precision_remote_config.example.json`。目标文件以 `.local.json` 结尾并按实际环境修改，仓库 `.gitignore` 已忽略 `*.local.json`。

配置字段和约束见 `references/remote_config_schema.json`。使用 `output.local_path` 手动配置本地输出根目录；相对路径以当前 Skill 集合根目录解析。命令行中的 `--output-dir`（兼容别名 `--artifact-root`）、`--host`、`--user`、`--repo-root`、`--remote-repo`、`--script`、`--conda-env` 和重复的 `--cann-env-script` 会覆盖配置文件。

## 文件职责

- 使用 `config/remote_config.example.json` 作为可复制修改的远端验证配置示例。
- 使用 `config/precision_remote_config.example.json` 定义 worktree、精度 runtime、远端环境和产物回收的一体化示例。
- 使用 `references/remote_config_schema.json` 定义输入配置的字段、类型和约束，作为配置接口合同。
- 使用 `references/validation_report_schema.json` 定义验证报告的输出格式，作为总控 Skill 和其他程序稳定解析报告的接口合同。
- 精度任务涉及的 13 个脚本、端到端调用图和失败诊断顺序见 `../precision-alignment/references/script-contracts.md`；只有审计、修改接口或结构化状态不足时读取。

## 工作流

1. 先运行 `plan`，核对本地路径、远端目标和将要执行的命令。
2. 远端已经存在同步内容时可运行 `preflight`，验证 SSH、远端仓库、runtime、Conda、CANN、launcher 和配置声明的依赖路径，但不启动任务。
3. 运行 `run`。该单一命令同步 worktree 和 runtime，执行 preflight，通过 `npu-smi info` 确认 NPU 空闲后启动 launcher，并回收配置声明的产物。
4. 读取本地 `validation_report.json`、`execution.log` 和 `remote_artifacts/`；不要只依赖终端截断输出。
5. 如果报告显示远端任务失败，保留原始日志并交给总控 skill 诊断，不要通过修改验证环境掩盖失败。

## 脚本入口

```bash
python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode plan

python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode preflight

python3 scripts/run_validation.py \
  --config config/precision_remote_config.local.json \
  --mode run

python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode run \
  --output-dir /path/to/custom/output
```

## 输出与退出码

报告格式见 `references/validation_report_schema.json`。本地 artifact 目录至少包含：

- `validation_report.json`
- `execution.log`（保存远端 launcher 的完整 stdout 与 stderr）
- `remote_artifacts/`（配置 artifact 回收时包含远端 JSON、Markdown 和日志）
- 训练任务继续把执行日志实际写为 `training.log`，并在报告中同时作为 `execution_log` 暴露；精度任务写 `execution.log`。

退出码：`0` 表示对应模式成功；`2` 表示配置或本地输入非法；`3` 表示 SSH、同步、远端预检、NPU 状态检查、产物目录冲突或产物回收失败；`4` 表示远端 launcher 失败；`5` 表示 NPU 已被进程占用、任务未启动；`124` 表示远端执行超时。报告 `stage: stalled` 表示静默看门狗判定挂死（`execution_result.exit_code` 为 `125`）。远端 launcher 的原始退出码保存在 `execution_result.exit_code`。
