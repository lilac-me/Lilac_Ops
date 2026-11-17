# MindSpeed-Ops 每日/版本看护 Skill（mindspeed-ops-nightly）

> **给 agent 的入口**：先读 `SKILL.md`（完整流程、参数、坑清单，当前 **v0.8**），再按本文件（`references/README.md`）"接入步骤"执行。
> **部署**：服务器运行目录是 `/mnt/share/ops_nightly_agent/skill/`（= `$WS/skill/`），由 `scripts/deploy_to_nfs.py` 同步；本目录即 skill 包根，自包含、可迁移。

## 这是什么

对昇腾 A3 NPU 容器/服务器上的 [MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops) 仓做看护：
UT 单测 + 算子 ATK 精度/性能，产出报告（失败项 + 负责人），归档到 NFS。

## 目录结构（v0.8 定稿）

```text
mindspeed-ops-nightly/
├─ SKILL.md                     ← 主文档：frontmatter + 输入/双模式/MODE/流程/报告格式/坑清单（v0.8: 容器/服务器双模式 + 复测 + 日志打包）
├─ scripts/
│  ├─ bootstrap.sh              ← 每容器/服务器 init：condarc→env 验证→atk 依赖自检→msops 冒烟→设备自检
│  ├─ run_all.sh                ← daily/version 主编排（.phase 断点续跑 + 心跳 + 卡资源参数化）
│  ├─ mode_daily.sh             ← daily 更新（master pull + 源码重装）
│  ├─ mode_version.sh           ← version 更新（PKG_DIR 全量环境 + 最新源码；PKG_DIR 必填）
│  ├─ nightly_ut.sh             ← UT NCARDS 分桶并行（排除挂起 sinkhorn）
│  ├─ ut_remain.sh              ← sinkhorn 900s 限时补跑
│  ├─ pack_logs.sh              ← 看护结束打包日志 → {日期}-log.zip（v0.7）
│  ├─ nightly_atk_acc.sh        ← ATK 精度 NCARDS 并发池（慢 op 先入池）
│  ├─ nightly_atk_perf.sh       ← ATK 性能 NCARDS 并发池（生成/对比分支 + per-op 断点）
│  ├─ run_retest.sh             ← 复测编排（只重跑失败项 + 稳定复现判定）
│  ├─ summarize.py              ← 看护报告解析 + owner_map 动态加载 + report.md/json（v5）
│  ├─ summarize_retest.py       ← 复测报告解析 + 判定（v0.6）
│  ├─ step456_run.sh            ← 简易串联 runner（手动分步用）
│  ├─ check_skill.py            ← ★完整性自检（下一节）
│  ├─ deploy_to_nfs.py          ← ★部署到 NFS（含 md5 校验）
│  └─ helpers/                  ← ★agent 侧运维助手（参数化, 无硬编码）
│     ├─ ssh_cmd.py             ←   paramiko 执行/上传，容器/服务器双模式（NPC_SUDO=1 走 sudo 通道）
│     ├─ poll_run_all.py        ←   轮询 run_all/.phase 进度
│     ├─ watch_container.py     ←   容器失联自动恢复跟踪
│     ├─ wait_phase.sh          ←   等 .phase 推进
│     ├─ wait_ut_done.sh        ←   等 ut 阶段结束
│     ├─ wait_perf_proc.sh      ←   等 perf 阶段结束
│     ├─ check_report_json.py   ←   核对 report.json 关键字段
│     └─ verify_ut_parse.py     ←   回归验证 UT 合并格式可被 summarize 解析
├─ templates/
│  └─ condarc                   ← 新容器 ~/.condarc 模板（envs_dirs/pkgs_dirs 指向共享盘）
└─ references/
   ├─ README.md                 ← 本文件（快速开始 + agent 接入步骤）
   ├─ config.json               ← 统一参数（ws/仓库 master/env/atk --max_task 10 -to 1200/复测/双模式）
   ├─ owner_map.md              ← owner_map 外置说明（2026-09-08 脱敏：映射在运行环境 $WS/owner_map/，开源不含人名）
   ├─ environment-setup.md      ← 环境准备要点（conda/CANN/torch/triton/atk/msops 安装 + 验证矩阵）
   └─ vllm_install_notes.md     ← vllm 安装规范（独立环境, 勿装看护 env）
```text
## 三种模式（用户定义）

| 模式 | 更新范围 | 入口 |
| --- | --- | --- |
| **daily 日常看护（默认）** | 只更新 mindspeed-ops（卸载→拉 master 最新→源码重装），其他环境不动 → UT+精度+性能 | `bash $WS/skill/scripts/run_all.sh` |
| **version 版本看护** | 按 `PKG_DIR` **全量更新环境**（CANN .run + torch/torch_npu/triton_ascend/atk whl+依赖）→ 最新源码 → UT+精度+性能 | `MODE=version PKG_DIR=<目录> CONTAINER=<ip> bash $WS/skill/scripts/run_all.sh` |
| **retest 复测** | **不更新任何东西**：只重跑最近一轮失败项，判定 稳定复现/未复现/无法判定 | `bash $WS/skill/scripts/run_retest.sh [REF_DIR] [DATE]` |

产物目录按"运行当天"新建：看护 `$WS/{当天}/`，复测 `$WS/{当天}_retest/`（互不覆盖）。

## 两种计算资源提供方式（v0.8：容器 / 服务器双模式）

接口统一、环境变量切换，脚本与助手工具均已兼容：

| 项 | 方式A 容器（默认） | 方式B 服务器（NPC_SUDO=1） |
| --- | --- | --- |
| 登录 | root 直连 | 普通用户 + `sudo -S -i` 切 root |
| 凭据 env | `NPC_IP/NPC_USER=root/NPC_PASS` | `NPC_IP/NPC_USER=<用户>/NPC_PASS` + `NPC_SUDO=1` + `SUDO_PASS`（缺省同 `NPC_PASS`） |
| 卡资源 | 默认 6 卡 | `NCARDS=<N>` `ASCEND_RT_VISIBLE_DEVICES=0,1,...,N-1`（**逻辑号**，勿用 /dev/davinci 物理号） |
| 上传 | sftp 直写 NFS | 先传 `/home/<用户>/` 再 sudo cp（`ssh_cmd.py --sftp-put` / `deploy_to_nfs.py` 已内置） |
| 示例 | 6 卡容器 | 16 卡服务器（普通用户 + sudo） |

- `run_all.sh` / `run_retest.sh` / `nightly_ut.sh` / `nightly_atk_acc.sh` / `nightly_atk_perf.sh` / `ut_remain.sh`：卡资源由 `NCARDS` / `ASCEND_RT_VISIBLE_DEVICES` 控制，默认 6 卡行为不变。
- `ssh_cmd.py` / `poll_run_all.py` / `watch_container.py` / `deploy_to_nfs.py`：识别 `NPC_SUDO=1` 自动走 sudo 通道。
- `bootstrap.sh` 5/5 设备自检逐个 `setDevice` 验证逻辑设备号可用（实测物理号 12/13 报 aclInit 107001，逻辑号 0/1 正常 → 自检可提前暴露）。

## agent 接入步骤（每次运行）

1. **要输入**：计算资源（容器 root 直连，或服务器 普通用户+sudo）；MODE（默认 daily）；version 还要 PKG_DIR；可选 DATE/CONTAINER。
   **凭据规则**：密码只经环境变量 `NPC_IP/NPC_USER/NPC_PASS`（服务器另加 `NPC_SUDO=1` 与 `SUDO_PASS`）传给脚本，**绝不进日志/报告/命令行**。
   服务器/多卡环境另传 `NCARDS` 与 `ASCEND_RT_VISIBLE_DEVICES`（逻辑号，默认 6 卡不变）。
2. **自检（可选但推荐）**：`python scripts/check_skill.py`（语法/结构/owner_map 外置脱敏与动态加载/引用完整性）。
3. **部署（新容器/新服务器或包有改动时）**：
   `$env:NPC_IP=...; $env:NPC_PASS=...; python scripts/deploy_to_nfs.py` → 全部文件 md5 校验一致后开始。
4. **接驳 + bootstrap**：
   - `Test-NetConnection <IP> -Port 22` 通 → `python scripts/helpers/ssh_cmd.py "bash $WS/skill/scripts/bootstrap.sh"`（写 condarc + env 验证 + atk 依赖 + msops 冒烟 + 设备自检）。
   - bootstrap 出"全部 OK"再继续；报 `EnvironmentNameNotFound` = condarc 未生效（重跑 bootstrap）。
5. **启动看护（长任务, 后台, 日志落 NFS）**：

   ```bash
   # daily（.phase 断点续跑, 容器释放后换容器重跑同一命令自动续）
   setsid nohup bash $WS/skill/scripts/run_all.sh </dev/null >> $WS/logs/run_all_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
   # 复测
   setsid nohup bash $WS/skill/scripts/run_retest.sh </dev/null >> $WS/logs/run_retest_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
   ```

   （SSH exec_command 会因后台进程挂住超时，属正常；用 helper 单独轮询。）
6. **轮询进度**：`python scripts/helpers/poll_run_all.py <DATE>`（或容器上 `bash wait_ut_done.sh $OUT` / `wait_perf_proc.sh $OUT`）；
   容器失联 → `watch_container.py <DATE>` 自动重连跟踪；彻底换容器 → 重跑 run_all/run_retest（.phase 续跑）。
7. **取报告**：`$WS/{DATE}/report.md` + `report.json`（看护）或 `$WS/{DATE}_retest/retest_report.md`（复测）；
   看护运行日志由 pack 阶段自动打包为 **`$WS/{DATE}/{DATE}-log.zip`**（UT/ATK 精度/ATK 性能日志，无需手工收集）；
   控制台摘要模板（用户确认）：三个统计行下各跟 `失败用例：<测试文件去.py / 算子> @负责人`（@ 与负责人之间无空格；ATK None 只计数不列失败）。
8. **复测触发**：日常看护交付后，用户说"复测"→ 跑 run_retest.sh（自动参考最近一轮）。

## 运行环境事实（服务器侧, 部署不变）

- WS=`/mnt/share/ops_nightly_agent`（NFS 共享, 换容器复用；repo、env、产物全在其下）
- env=`ops_nightly_agent`（python 3.12, conda envs 在 `/mnt/share/vllm/conda_envs/`）
- `source $WS/env.sh`（CANN set_env + proxy + `conda activate ops_nightly_agent`）——**每条远端命令都要先 source**
- 代理：`80.*`→`80.5.1.30:8091`，`90.*`→`90.91.191.181`，其它→`141.61.33.201`；`GIT_SSL_NO_VERIFY=1`
- A3 容器默认 **6 卡**；服务器/少卡环境用 `NCARDS` / `ASCEND_RT_VISIBLE_DEVICES` 覆盖（**逻辑设备号 0..N-1**），`SOC_VERSION=ascend910_93`
- 仓库 `https://gitcode.com/Ascend/MindSpeed-Ops.git`，分支 **master**；repo 在 `$WS/MindSpeed-Ops`
- 各类脚本 7×24 都从 `$WS/skill/scripts/` 与 `$WS/scripts/` 调用（deploy 已同步两份）

## 状态与变更记录

- v0.3（2026-08-26）：daily/version 双模式 + v4 报告结构，全链路真实跑通。
- v0.5（2026-08-29）：占用率改造——UT 6 卡分桶、ATK 6 卡并发池、慢 op 先入池、阶段心跳；summarize.py v5 修复 timing/versions 解析。20260829 全流程交付（repo fe01feb, 新增 chunk_loss）。
- **v0.6（2026-09-01）**：新增**复测模式**（run_retest.sh + summarize_retest.py，只重跑失败项、判定稳定复现/未复现/无法判定）；**归档自包含化**——helpers/templates/check_skill.py/deploy_to_nfs.py 收进包、全部参数化去硬编码。20260901 复测完成：13/13 失败项稳定复现。
- **v0.7（2026-09-01）**：新增**日志打包**——看护结束后 run_all 自动跑 pack 阶段（pack_logs.sh），把 UT/ATK 精度/ATK 性能运行日志压缩为 `{日期}-log.zip` 存于 `$WS/{日期}/`；deploy 的 RUNTIME_SCRIPTS 加入 pack_logs.sh；报告第五段归档位置补充日志包。
- **v0.8（2026-09-09）**：**容器/服务器双模式**——卡资源全脚本参数化（NCARDS/ASCEND_RT_VISIBLE_DEVICES，默认 6 卡不变）；SSH 通道支持服务器方式（NPC_SUDO=1，普通用户+sudo，含 ssh_cmd/poll/watch/deploy 与 --sftp-put 家目录中转）；bootstrap 新增 5/5 设备可用性自检（暴露物理号/逻辑号错位）；owner_map 外置脱敏（v0.8 附带，详见 references/owner_map.md）。
