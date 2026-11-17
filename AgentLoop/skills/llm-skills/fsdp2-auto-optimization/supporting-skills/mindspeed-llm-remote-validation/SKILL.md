---
name: mindspeed-llm-remote-validation
description: 将本地 MindSpeed-LLM 仓库同步到用户指定的免密 SSH 服务器，检查 NPU 空闲状态后在指定 Conda 与 CANN 环境中启动验证脚本，并将远端输出、退出码和结构化报告保存到本地。适用于 FSDP2、mcore 等后端的远端训练启动、环境预检、同步验证和运行结果回收。
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM 远端验证

将这个 skill 作为本地控制器使用。只在远端同步代码、加载环境并启动训练；在本地保存输出和判断脚本是否成功。不要在远端修改 task、阈值、history 或产品代码。

## 核心约定

- 要求用户提前配置 SSH 免密登录；始终使用 `BatchMode=yes`，不要交互式读取密码。
- 要求远端用户已经通过 `conda init bash` 或等效 shell 配置，使交互式 Bash 能直接执行 `conda`；不要在配置中绑定 Conda 安装路径。
- 将服务器、仓库路径、训练脚本、Conda 和 CANN 信息放入私有配置。不要把密码、私钥内容或 token 写入配置和报告。
- 将 `remote_repo` 视为远端仓库的完整路径，例如 `/home/user/testllm/MindSpeed-LLM`，不要把仓库内容错误地平铺到它的父目录。
- 默认使用增量 `rsync` 且不删除远端文件。只有配置明确设置 `sync.delete: true` 时才启用删除，并先确认目标是具体仓库目录。
- 在 `run` 模式下依次执行：创建远端仓库目录、同步、环境预检、NPU 空闲门禁、训练脚本、报告落盘。
- 启动训练前必须执行 `npu-smi info`。检测到任何 NPU 进程时立即退出且不启动训练；命令失败或无法确认空闲时同样保守退出。该门禁不可通过配置关闭。
- 使用 `bash -e -o pipefail` 执行训练脚本，确保普通命令或管道中的训练失败能够传递为非零退出码。
- 默认将每次训练尝试写入当前 Skill 集合根目录的 `output/remote-validation-<四位数字>/`，例如 `remote-validation-0001`、`remote-validation-0002`。扫描已有目录的最大编号后严格递增；失败和超时也保留对应目录及完整日志，不复用旧编号。
- 当前版本只根据远端脚本 exit code 判断运行成功；loss、性能和显存 validator 后续在本地分析层实现。

## 配置

复制 `config/remote_config.example.json` 为以 `.local.json` 结尾的私有文件，再按实际环境修改。仓库 `.gitignore` 已忽略 `*.local.json`。

配置字段和约束见 `references/remote_config_schema.json`。使用 `output.local_path` 手动配置本地输出根目录；相对路径以当前 Skill 集合根目录解析。命令行中的 `--output-dir`（兼容别名 `--artifact-root`）、`--host`、`--user`、`--repo-root`、`--remote-repo`、`--script`、`--conda-env` 和重复的 `--cann-env-script` 会覆盖配置文件。

## 文件职责

- 使用 `config/remote_config.example.json` 作为可复制修改的远端验证配置示例。
- 使用 `references/remote_config_schema.json` 定义输入配置的字段、类型和约束，作为配置接口合同。
- 使用 `references/validation_report_schema.json` 定义验证报告的输出格式，作为总控 Skill 和其他程序稳定解析报告的接口合同。

## 工作流

1. 先运行 `plan`，核对本地路径、远端目标和将要执行的命令。
2. 运行 `preflight`，验证 SSH、远端仓库、Conda 环境、CANN 脚本和训练脚本，但不启动训练。
3. 运行 `run`，同步代码；脚本会先通过 `npu-smi info` 确认所有 NPU 均无运行进程，再启动训练。
4. 读取本地 `validation_report.json` 和 `training.log`；不要只依赖终端截断输出。
5. 如果报告显示远端训练失败，保留原始日志并交给总控 skill 诊断，不要通过修改验证环境掩盖失败。

## 脚本入口

```bash
python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode plan

python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode preflight

python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode run

python3 scripts/run_validation.py \
  --config config/remote_config.local.json \
  --mode run \
  --output-dir /path/to/custom/output
```

## 输出与退出码

报告格式见 `references/validation_report_schema.json`。本地 artifact 目录至少包含：

- `validation_report.json`
- `training.log`（进入训练阶段后，保存远端训练的完整 stdout 与 stderr）

退出码：`0` 表示对应模式成功；`2` 表示配置或本地输入非法；`3` 表示 SSH、同步、远端预检或 NPU 状态检查失败；`4` 表示训练脚本失败；`5` 表示 NPU 已被进程占用、训练未启动；`124` 表示训练超时。
