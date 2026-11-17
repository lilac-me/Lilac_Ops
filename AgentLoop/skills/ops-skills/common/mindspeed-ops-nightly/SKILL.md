---
name: mindspeed-ops-nightly
description: >-
  This skill should be used when the user asks to run daily or version smoke
  guarding for the MindSpeed-Ops repository on Ascend NPU: installing the
  latest master source, running the unit test suite and operator ATK
  accuracy/performance tasks, retesting previously failed items, or producing a
  guarding report that maps failed operators and test cases to their owners.
  Typical triggers include daily guarding, release guarding, UT/ATK smoke
  checks, and retesting failed items.
version: 0.8.0
domain: ops
function: common
tags:
  - nightly
  - smoke-test
  - ut
  - atk
  - guard
author: mindspeed-agent
metadata:
  zh-description: >-
    当用户请求对 MindSpeed-Ops 仓库执行每日/版本看护（UT 单测 + 算子 ATK 精度/性能冒烟）、
    复测失败项、或生成含失败算子与负责人映射的看护报告时使用本 skill。
    典型触发词：跑每日看护 / 版本看护 / UT ATK 冒烟 / 复测失败项。
---

# Skill: MindSpeed-Ops 每日看护（mindspeed-ops-nightly）

> 版本: v0.8（2026-09-09 **容器/服务器双模式**：同一 skill 既支持 root 直连容器，也支持普通用户+sudo 登录带卡服务器——卡资源全脚本参数化（NCARDS/ASCEND_RT_VISIBLE_DEVICES，默认 6 卡不变）、SSH 助手/部署工具识别 NPC_SUDO=1 自动走 sudo 通道、bootstrap 新增 5/5 设备可用性自检（物理号/逻辑号错位陷阱，55 容器实战）、owner_map 外置脱敏；v0.7 为 2026-09-01 新增**日志打包**：日常/版本看护全部阶段结束后，run_all 自动执行 pack 阶段——把 UT / ATK 精度 / ATK 性能运行日志压缩为 `{日期}-log.zip` 存于 `$WS/{日期}/`；v0.6 为同日新增**复测模式**：run_retest.sh + summarize_retest.py，日常看护跑完后用户下发"复测"即只重跑失败项并判定是否稳定复现；v0.5 为 2026-08-29 占用率改造：UT 6 卡分桶并行 + ATK 6 卡并发池 + 慢 op 排序 + 阶段心跳;summarize.py v5 修复 timing/versions 解析；2026-09-07 修复 run_all.sh **续跑语义**：.phase=acc 时旧逻辑会重跑 init→install→ut→acc，现按阶段序号只跑 .phase 之后阶段；同日 nightly_atk_perf.sh 增 **per-op 断点**：已有扁平基准 `$WS/triton/<op>.xlsx` + perf 日志的算子跳过重跑（容器 1~2h 频繁释放场景的收尾加速，结果提取仍从既有日志 grep 不丢算子））
> 结构: SKILL.md + scripts/ + references/ + templates/（MindSpeed-Agent 风格 skill 目录；README.md 与 config.json 位于 references/）
> 依赖: 目标仓库 [MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops) 的 master 分支；环境准备要点见 `references/environment-setup.md`
> 服务器运行目录: `$WS/skill/`（= `/mnt/share/ops_nightly_agent/skill/`）；**本目录即 skill 包根**（自包含, 可迁移）

## 用途

对 [MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops) 仓做**每日/版本看护**：
UT 单测 + 算子 ATK 精度/性能看护，输出报告并汇总失败算子与负责人。
对应原工具型冒烟流程（smoke_plugin test_acc_nightly / test_performance_nightly），
参数与分发方式对齐用户脚本，唯一差异：**全部算子统一 `--max_task 10 -to 1200`**（用户指定）。

> **与其他 skill 的关系**：本 skill 是"仓库级每日/版本看护编排"，面向 MindSpeed-Ops 全仓的 UT + ATK 精度/性能回归与报告聚合（含复测判定、负责人映射）；
> 而 `ops-test-gen` / `ops-boundary-survey` / `ops-profiling` 聚焦单算子测试用例构造与专项分析，定位不同、不构成功能重复。

## 输入（每次运行时提供）

| 项 | 必填 | 说明 |
| --- | --- | --- |
| 计算资源（IP / 账号 / 密码） | 是 | 容器 root 直连，或服务器普通用户 + sudo（另传 `NPC_SUDO=1` / `SUDO_PASS`）；**每次可能不同**，容器随时可能被释放 |
| MODE | 否 | `daily`(日常看护, 默认) \| `version`(版本看护) \| `retest`(复测, 只重跑失败项) |
| PKG_DIR | version 必需 | B 版本配套包统一存放目录（CANN `*.run` + torch/torch_npu/triton_ascend/atk 等 `*.whl`，布局同 env-pkg） |
| DATE | 否 | 默认今日；**每天新建目录**：日常/版本看护产物归档 `$WS/{DATE}/`，复测产物归档 `$WS/{DATE}_retest/`；`.phase` 断点续跑 |
| REF_DIR | retest 可选 | 复测参考哪一轮看护报告（默认 `$WS/20*/` 下最新含 report.json 者；一般无需指定） |
| CONTAINER | 否 | 计算资源 IP 透传给报告元信息（默认 unknown） |
| NCARDS / ASCEND_RT_VISIBLE_DEVICES | 否 | 卡资源覆盖：默认 6 卡（`NCARDS=6`，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5`）；服务器/多卡按实际**逻辑设备号** 0..N-1 覆盖 |

## 执行方式（v0.8 双模式：容器 / 服务器）

同一 skill 支持两种计算资源提供方式，**接口统一、环境变量切换**：

| 项 | 方式A 容器（默认） | 方式B 服务器（NPC_SUDO=1） |
| --- | --- | --- |
| 登录 | root 直连 | 普通用户 + `sudo -S -i` 切 root |
| SSH env | `NPC_IP/NPC_USER=root/NPC_PASS` | `NPC_IP/NPC_USER=<用户>/NPC_PASS` + `NPC_SUDO=1` + `SUDO_PASS`（缺省同 `NPC_PASS`） |
| 卡资源 | 默认 `NCARDS=6`，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5` | 按实际逻辑卡覆盖：`NCARDS=<N>` `ASCEND_RT_VISIBLE_DEVICES=0,1,...,N-1` |
| 上传 | sftp 直写 NFS | 先传 `/home/<用户>/` 再 sudo cp（`ssh_cmd.py --sftp-put` / `deploy_to_nfs.py` 已内置） |
| 典型示例 | 6 卡容器 | 16 卡服务器（普通用户 + sudo） |

- 所有运行脚本（`run_all.sh`/`run_retest.sh`/`nightly_ut.sh`/`nightly_atk_acc.sh`/`nightly_atk_perf.sh`/`ut_remain.sh`）卡资源均已参数化：`NCARDS` 控制分桶/并发池，`ASCEND_RT_VISIBLE_DEVICES` 控制设备可见性，默认值保持 6 卡容器行为不变。
- 助手工具（`ssh_cmd.py`/`poll_run_all.py`/`watch_container.py`/`deploy_to_nfs.py`）识别 `NPC_SUDO=1` 自动切换 sudo 通道，root 直连模式行为不变。
- **设备号一律用逻辑号 0..N-1**（`bootstrap.sh` 5/5 设备自检会逐个 `setDevice` 验证；失败即环境/设备号配置问题，先查 `ASCEND_RT_VISIBLE_DEVICES`）。

## 两种运行模式（用户定义）

| 模式 | 更新范围 |
| --- | --- |
| **daily 日常看护** | 只更新 mindspeed-ops：卸载当前 mindspeed_ops → 拉取 gitcode **master** 分支最新修改 → 源码安装 `pip install -e . --extra-index-url=triton-ascend... --no-build-isolation --no-deps`；**其他环境（conda/CANN/torch/torch_npu/triton/atk）不动** → UT + 精度 + 性能 |
| **version 版本看护** | B 版本配套更新后，按显式提供的 `PKG_DIR` **全量更新环境**（CANN .run 三件套 → conda 依赖 → 4 个内部 whl(--no-deps) → atk 运行时依赖 → 刷新 env.sh/versions.txt）→ 拉取远端 master 最新 mindspeed-ops 源码安装 → UT + 精度 + 性能 |
| **retest 复测** | **不更新任何东西**：只重跑**最近一轮看护（REF）的失败项**——UT 失败测试文件（NCARDS 桶并行）、ATK 精度失败算子、ATK 性能失败算子（NCARDS 并发池）——判定每项**稳定复现 / 未复现（本次通过）/ 无法判定**；产物 `$WS/{DATE}_retest/` |

## 执行流程（run_all.sh 自动编排、断点续跑）

```text
init → install(按模式) → ut → acc → perf → summarize → pack
```text
每阶段完成写 `$OUT/.phase`（`$OUT=$WS/{DATE}`）；换容器后重跑 run_all.sh 自动跳过已完成阶段
（**容器随时可能释放，产物/日志全部实时落 NFS**）。

```bash
# daily
nohup bash $WS/skill/scripts/run_all.sh </dev/null >> $WS/logs/run_all_$(date +%Y%m%d)_$(date +%H%M%S).log 2>&1 &
# version
MODE=version PKG_DIR=/path/to/bundle CONTAINER=<ip> bash $WS/skill/scripts/run_all.sh
# 服务器（普通用户 + sudo, 16 卡）
NPC_IP=<ip> NPC_USER=<user> NPC_PASS=<pwd> NPC_SUDO=1 SUDO_PASS=<pwd> NCARDS=16 ASCEND_RT_VISIBLE_DEVICES=0,1,...,15 bash $WS/skill/scripts/run_all.sh
```text
1. **init — bootstrap.sh（每容器/服务器必跑，换容器必跑）**：写 `~/.condarc`（envs_dirs/pkgs_dirs 指向共享盘，否则 conda activate 失败）→ source `$WS/env.sh` 验证 torch/torch_npu/atk/PIL/matplotlib 与设备数 → atk 运行时依赖自检（requests/PIL/matplotlib/wcwidth/kombu/sqlalchemy/celery/tqdm/opentelemetry）→ mindspeed_ops import 冒烟 → **5/5 设备可用性自检**（逐个 `setDevice` 验证逻辑设备号 0..N-1，暴露物理号/逻辑号错位陷阱）。
2. **install**：daily → `mode_daily.sh`（master 分支 pull + 源码重装，**安装前 export MINDSPEED_BUILD_CATLASS=1 编译 catlass**）；version → `PKG_DIR=<目录> mode_version.sh`（全量环境更新 + 最新源码安装 + 版本自检，同样带 catlass 编译标志）。
3. **UT — nightly_ut.sh**（2026-08-29 改 NCARDS 并行，默认 6 卡）：`pytest unit_tests/` 单进程 → 排除挂起 `test_sinkhorn.py`（triton/tilelang 两个）后其余 `test_*.py` 字典序轮转分 **NCARDS 桶**，各桶进程绑 `ASCEND_RT_VISIBLE_DEVICES` 对应卡并行；python 合并各桶汇总（failed/passed/skipped 求和）写总汇总行 + FAILED 明细 → `result_ut.log`；`ut_remain.sh` 只补两个 sinkhorn（各 900s 限时，仅追加 SINKHORN 标记行不污染汇总）；写入 `$OUT/timing.txt`。
4. **ATK 精度 — nightly_atk_acc.sh**（2026-08-29 改 NCARDS 并发池）：26 算子按最近一轮 `op_timing_acc.txt` 耗时降序入池，**NCARDS 并发作业池**（`wait -n` + kill -0 过滤，活跃恒 ≤NCARDS），`--devices $((idx%NCARDS))` 绑卡（`ASCEND_RT_VISIBLE_DEVICES`）；`grep acc_pass_result` → result_atk.log；per-op 耗时写 `$OUT/op_timing_acc.txt`。慢 op 先入池 → 尾部无长单，多卡全程满载（原 26 全并发 8 卡 → 尾部只剩慢 op 在少数卡, 占用率 <5% 被容器平台释放）。
5. **ATK 性能 — nightly_atk_perf.sh**：同上 NCARDS 并发池结构 `--task performance_device`（backend npu）；无 xlsx 走生成模式、有 `triton/<op>/<op>.xlsx` 走对比分支 `--is_compare False --bm_file ... -s 0 -e 3`（v0.5 起按 per-op 断点跳过已有基准+日志的算子），统一 `-to 1200`；完成后 excel mv 到 `$WS/triton/<op>.xlsx` 作为下轮基准；`grep is_device_perf_pass` → result_atk_perf.log；per-op 耗时写 `$OUT/op_timing_perf.txt`。
6. **summarize.py（v5）**：`python summarize.py $OUT $MODE $CONTAINER` → `report.json`（机读）+ `report.md`（人读）；控制台摘要：三栏统计 + **失败项与负责人（accuracy/performance 逐算子、UT 逐失败用例，均带负责人）** + 按负责人聚合（负责人映射运行时从 NFS `owner_map.json` 动态加载，开源仓库不含人名）。
7. **pack — pack_logs.sh（v0.7 新增）**：把本轮运行日志压缩为 `{日期}-log.zip` 存于 `$OUT/`——UT（`ut_part*.log` / `result_ut.log` / `ut_remain.log`）、ATK 精度（`result_atk.log` / `atk_logs/acc_*.log`）、ATK 性能（`result_atk_perf.log` / `atk_logs/perf_*.log`），附 `timing.txt` / `op_timing_{acc,perf}.txt`；zip 内顶层目录为 `{日期}`（解压即得 `{日期}/atk_logs/...`）。**阶段成功才写 `.phase`**（失败不标记，下次重跑自动重试）。日常/版本看护每次跑完必有该 zip。

## 复测模式（retest，skill v0.6，2026-09-01 新增）

日常/版本看护跑完后，用户下发"复测"命令 → 启动 `run_retest.sh`（**只重跑失败项，不更新/不重装/不拉取**）：

```bash
# 默认：参考最近一轮看护报告（含 report.json 的最新 $WS/20*/ 目录），产物今天的 $WS/{当天}_retest/
nohup bash $WS/scripts/run_retest.sh </dev/null >> $WS/logs/run_retest_$(date +%Y%m%d)_$(date +%H%M%S).log 2>&1 &
# 显式指定参考轮次:  bash run_retest.sh $WS/20260829 [DATE]
```text
- **复测范围**（从 REF `report.json` 提取）：UT 失败测试文件（NCARDS 桶并行，每桶 `timeout ${RETEST_UT_TIMEOUT:-3600}` 防挂起）+ ATK 精度失败算子 + ATK 性能失败算子（NCARDS 并发池，命令与日常看护完全一致：`--max_task 10 -to 1200`，perf 同走 generate 分支）。**perf 无结果(None)项非失败项，不纳入复测**。
- **判定规则（summarize_retest.py）**：复测再次失败 → **稳定复现**；复测通过（UT 无 FAILED / acc `Pass` / perf `Pass`）→ **未复现（本次通过）**；无结果/未执行/case gen 失败/UT 桶超时或文件缺失 → **无法判定**（逐项给出原因）。
- **阶段**：`init → ut_retest → acc_retest → perf_retest → summarize_retest`（`.phase` 断点续跑，换容器可续；每阶段 30s 心跳）。失败清单落 `$OUT/{ut_failed_files,acc_failed_ops,perf_failed_ops}.txt`。
- **产物**：`$WS/{当天}_retest/`（**每天新目录**，与当日日常看护 `$WS/{当天}/` 隔离互不覆盖）：`retest_report.md`/`retest_report.json`（五段结构，判定+基准原因+复测结果+负责人聚合）+ `result_ut_retest.log`、`result_atk_{acc,perf}_retest.log`、`atk_logs/retest_{acc,perf}_*.log`、`timing.txt`。
- **调用约定**：daily 报告交付后，用户消息含"复测"即执行本模式（可用今天新容器；先 bootstrap.sh 再接环境）。

## 报告格式（用户确认 v4：精简 UT + 元信息/耗时 + 负责人维度）

`report.md` 五段结构：

- **一、结果摘要**：UT（通过/失败/跳过 + 多段运行注记 + **失败负责人分布** + **失败明细只到"文件→条数"**，完整用例留在 report.json/result_ut.log）；ATK 精度/性能（通过/失败计数 + 失败算子@负责人(原因)一行）；失败项与负责人速览。
- **二、元信息**：容器、看护路径（daily/version）、环境版本（versions.txt：python/CANN/torch/torch_npu/triton/atk/SOC）、阶段耗时 + 总耗时（timing.txt）。
- **三、失败项详情**：每个失败扩展小节 `### [accuracy|performance|ut] op — 负责人: xx`，含原因、Total Task/success/failed、fp16/运行期异常日志细节、**每算子耗时**、日志路径。
- **四、按负责人聚合**：每个负责人一行，列出其全部失败项（含 UT 文件、精度、性能）。
- **五、报告与归档位置**：report.md/report.json、三个 result 日志、atk_logs/、性能基准 `$WS/triton/`、日志包 `{日期}-log.zip`（UT/ATK 精度/ATK 性能日志一键打包，v0.7）。
- **ATK 无结果（None/NO_RESULT）**：**不写入报告正文**（2026-09-07 从简；perf 首轮无基准属正常），完整保留在 report.json 的 `noresult_ops`，**不并入失败计数**。
- 失败原因归类：优先 `Total Task: N, success M`；全部失败且含 RuntimeError → "运行期异常"；含 float16 → 追加 "含 fp16 失败"。
- 归档：`/mnt/share/ops_nightly_agent/{日期}/`；性能基准 xlsx 存 `$WS/triton/`（跨天累积，供次日对比）。

report.json 顶层结构：

```json
{ "meta": {"date","guard_type","container","env_versions","timing_seconds"},
  "ut": {"passed","failed","skipped","failed_files[]","failed_cases[]","owner_dist"},
  "atk_accuracy": {"passed","failed","noresult","failed_ops[{op,owner,reason,seconds}]","noresult_ops[]"},
  "atk_performance": {...同上...},
  "by_owner": [{"owner","items[{kind,op,reason,seconds}]"}],
  "report_files": {...} }
```text
## 约束与坑（全部来自实战）

- 容器无预警释放：所有长任务 nohup + 日志落 NFS；轮询前先测 TCP:22；被杀进程按"容器释放"处理，换容器靠 `.phase` 续跑。
- **产物目录按"运行当天"新建**：日常/版本看护 `$WS/{当天}/`，复测 `$WS/{当天}_retest/`；同一天多次跑不互相覆盖（报告文件名不同或目录不同）；换天跑即新目录，不要写回旧日期目录。
- **复测判定依据的 status 字符串**：acc 提取 `acc_pass_result`、perf 提取 `is_device_perf_pass`，result 日志统一为 `[op] status: Pass/Failed/None`；None/NO_RESULT/case gen FAILED → 无法判定；UT 桶无汇总行（timeout 杀死）→ 所在桶各文件判"无法判定（超时/中断）"，FAILED 行仍存在的文件优先判"稳定复现"。
- **复测 perf 只重跑失败算子且不覆盖基准**：命令与 nightly_atk_perf.sh 同一 generate 分支（嵌套 `../../triton/<op>/<op>.xlsx` 永不命中），**跳过 excel mv**，不污染 `$WS/triton/` 跨天基准。
- 非交互 shell 不读 .bashrc：每条命令显式 source env.sh（proxy 配置见 `references/environment-setup.md`；GIT_SSL_NO_VERIFY=1）。
- **git fetch 必须 source env.sh（2026-08-28 实战抓 bug）**：任何 git 远端操作前先 `source $WS/env.sh`，否则 gitcode 证书校验失败（CAfile none）且 `git fetch | tail` 会吞掉失败退出码 → **静默沿用旧 commit 装包**。mode_daily.sh 已修复：fetch 前 source env.sh + `if ! git fetch...; then exit 1`。跑完 install 后务必核对 `git log -1` 的 HEAD 是当天新提交。
- **editable 安装新形态（2026-08-28 实战抓 bug，commit 0d55f1a 改 setup.py 后）**：新版 custom finder 只映射 `mindspeed_ops` 包，**不再把仓库根放进 sys.path** → atk 的 `triton_*.py`（`from tests.unit_tests...`）与**文件模式** pytest（`pytest a.py b.py`，如 ut_remain）会 `ModuleNotFoundError: No module named 'tests'`；目录模式 `pytest unit_tests/` 不受影响（rootdir 自动入 path）。修复：nightly_atk_acc.sh / nightly_atk_perf.sh / ut_remain.sh 均 `export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"`。症状：acc/perf 大量算子 **NO_RESULT**（16-19s 秒退）、ut_remain 在 collection 阶段 1 error。
- **ut_remain.sh 只读 `$DATE` 不读 `$1`（2026-08-28 抓）**：run_all 传的是绝对 `$OUT`，但脚本用 `$DATE`（run_all export 的是相对 `20260828`）→ cd 到 tests 后相对路径不存在 → tee 静默失败。已修为 `DATE="${1:-${DATE:-...}}"`。且 `RC=$?` 取的是管道尾部 tee 的退出码 → 用 `PIPESTATUS[0]`；**PART2(sinkhorn) 的 pytest 数字行不能拼进 result_ut.log**（会污染 summarize 的 summaries[-1]，只拼 PART1 段 + SINKHORN 标记）。
- **新容器 .condarc 必须含 `envs_dirs`（2026-08-29 抓）**：只写 channels 的 condarc 会让 `conda activate ops_nightly_agent` 报 `EnvironmentNameNotFound`（env 在共享盘 conda_envs，不在 conda 默认 envs_dirs）。手工接驳时**先跑 bootstrap.sh**（自动写 `~/.condarc`：envs_dirs + pkgs_dirs 指向共享盘）或直接上传带 envs_dirs 的 condarc，再 source env.sh 验证。症状：`which python` 为空、`atk`/pytest 全 command not found。
- **summarize.py v5 两处解析 bug（2026-08-29 抓并修复）**：① `read_timing` 原本 `wall={}` + `p[0] in wall` **恒 False** → 报告"阶段耗时: 未记录"（须初始化为 `{"ut":0,"acc":0,"perf":0}`）；② `read_env_versions` 只分割 `:`，但 versions.txt 是 `KEY=value`（`CANN=Version=9.2.0`，set -x 追踪行以 `+` 开头）→ 环境版本恒空（须兼容 `=`，并跳过 `+`/`note` 行）。修复后元信息段正常输出环境版本 + 阶段耗时 + 总耗时。
- **容器按占用率释放（2026-08-29 平台规则）**：容器占用率 <5% 持续超 2h 会被平台回收 → 看护策略：①UT/ATK 并行任务分散到多卡（UT NCARDS 桶、ATK NCARDS 并发池），多卡保持满载；②慢 op 按上轮耗时降序先入池（避免尾部只剩慢 op 在 1~2 卡、其余空闲）；③run_all.sh 每阶段 wait 期间每 30s 写 `$OUT/heartbeat` 观测低占用时段；④实在无法保证占用率 → 接受容器释放，靠 `.phase` 断点续跑换下一容器。UT 合并日志格式（TOTAL 行 + SINKHORN 标记行）已验证兼容 summarize.py summaries[-1] 解析。
- `npu-smi` 须先 source 隔离 CANN set_env.sh；pkill 防自匹配（`grep '[n]ame'`）。
- atk 依赖坑：`atk node --help` 通过 ≠ 任务链可用（kombu 等 task 路径才 import）→ 真实单算子冒烟终验；发行版名≠模块名（`opentelemetry`→`opentelemetry-api`、`attr`→`attrs`）；勿装 `atk.*` 可选子模块。
- 性能首轮无基准 → 生成模式（None 属正常）；下轮有 xlsx 自动走对比分支；基准名统一扁平 `$WS/triton/<op>.xlsx`。
- 慢算子：conv1d ~90min、chunk_local_cumsum 60 例 ~377s/例（单跑 28s/例）→ 大用例分段/降并行防容器回收。
- **catlass 算子必须在安装前 `export MINDSPEED_BUILD_CATLASS=1`（2026-09-01 实测验证）**：chunk_loss（`unit_tests/catlass/test_chunk_loss.py`）等 catlass 算子本源码编译默认关闭——setup.py：`catlass_enabled = MINDSPEED_BUILD_CATLASS=="1" and normalized_soc in {"ascend910b","ascend910_93"}`（SOC `ascend910_93` 经 `normalize_soc_version` → `ascend910_93` ✓）。**不 export 直接装 → chunk_loss UT 29 条全失败**（27F+2P，复测也稳定复现）；带 export 重装后同一文件 **29 passed in 10.96s**（容器实测验证）。CI 脚本 `ci/MindSpeed_Ops_ut.sh` 同款用法（注释 "must be enabled before installation"）。mode_daily.sh / mode_version.sh 已内置该 export（2026-09-01）。
- **run_all.sh 续跑语义修复（2026-09-07 实战抓 bug）**：旧 `phase()` 只跳过"当前 .phase 值"那一阶段，`.phase=acc` 时换容器续跑会把 init→install→ut→acc **全部重跑一遍**（换容器实测，重跑 install 重编译+UT+acc 约 3h+ 浪费）。已改为 `need_phase()`：按固定阶段序 `init install ut acc perf summarize pack`，**.phase 已完成阶段及更早全部跳过，只跑其后阶段**；无 .phase（首跑/新一天目录）全跑。注意：换容器续跑时 init（bootstrap）会被跳过 → **新容器必须手动先跑 bootstrap.sh**（condarc 现成）再启 run_all。
- **容器 1~2 小时频繁释放（2026-09-07 实战观察）**：一天内连 4 台被平台释放（含 perf 满 6 卡 35min 也被释放，疑似租期策略不只看占用率）。应对：① 产物/日志实时落 NFS（已是常态）；② run_all.sh 阶段级断点（见上）；③ nightly_atk_perf.sh **per-op 断点**——`$WS/triton/<op>.xlsx` + `perf_<op>.log` 已在则跳过该算子（结果提取仍从既有日志 grep `is_device_perf_pass`，报告不丢算子）；④ 新容器必跑 bootstrap.sh 再启 run_all。
- **perf_wy_fast 稳定复现失败（2026-09-03 定位）：atk+CANN profiling 目录清理竞态，非算子计算错误**：performance_device 45 case 全挂（0901 轮 90 处 Errno39 + 全新容器清残留重试 45/45），`is_device_perf_pass: None`；日志固定 `OSError: [Errno 39] Directory not empty: .../ascend_pt/logs`（重试清理 `[Errno 16] Device or resource busy`）→ `RuntimeError: run opp task failed!`，附带 op_summary csv 'N/A' 解析 WARNING；**profiling 采集本身全部正常**（有耗时数据）。定位实验：单 case 截取 json 稳定复现（三连败）；**同容器同方法对照 op 单 case Pass**；失败后 ascend_pt 目录被 cleanup 清空（logs 存在窗口 <0.2s，抓拍不到内容）→ 判定为 wy_fast 该算子 profiling 数据形态与 atk `parser_profile_result` 的 `shutil.rmtree` 存在确定性竞态（msprof 后台写 logs 目录），属 atk/CANN 工具链缺陷（atk 编译分发无法读码）。**处理口径**：看护/复测判「稳定复现（运行期异常：CANN ascend_pt 清理失败）」，原因归类运行期异常而非性能不达标；报障携带：单 case json + 完整日志 + 对照 op Pass 事实。
- **nnal 包不进看护环境**：env.sh 只 source `$WS/ascend/ascend-toolkit/set_env.sh`；nnal（atb/asdsip）装在 `$WS/ascend/nnal/` 但看护链路（msops UT/ATK）不依赖，仅在需要 atb 库/算子 so 时才 `source $WS/ascend/nnal/{atb,asdsip}/set_env.sh`（2026-09-01 查证）。
- **vllm/vllm-ascend**（若 env 需要）：vllm 先跑 `use_existing_torch.py` 剔除 torch 行再 `pip install -r requirements*.txt`；vllm-ascend 必须 `pip install -r requirements.txt`（pin torch-npu==2.10.0/triton-ascend==3.2.1）→ **独立环境**（如 ops_nightly_vllm），勿装入看护 env。详见 `references/vllm_install_notes.md` / `references/environment-setup.md`。
- **设备物理号 ≠ 可用逻辑号（2026-09-09 实战抓）**：`ls /dev/davinci*` 显示物理号（如 12/13），但 `ASCEND_RT_VISIBLE_DEVICES=<物理号>` 时 atk worker 报 `aclInit error 107001`（`rtGetDevMsg execution failed, context null`）→ 60 case 全败（41s 秒退）；改用**逻辑号 0/1**（`ASCEND_RT_VISIBLE_DEVICES=0`）后单算子 perf 60/60 全过。教训：ASCEND_RT_VISIBLE_DEVICES 的取值须本机实测（bootstrap 5/5 设备自检逐个 setDevice 可提前暴露），**勿按 /dev/davinci 文件名推导**；服务器上 perf 复测 NO_RESULT 亦属同类 NPU 初始化/被杀环境问题，不代表算子结论。

## 文件布局

```text
$WS/skill/（= /mnt/share/ops_nightly_agent/skill/）
├─ SKILL.md                主文档（frontmatter + 输入/双模式/MODE/流程/报告格式/坑清单）
├─ templates/condarc      新容器 ~/.condarc 模板（envs_dirs/pkgs_dirs 指向共享盘）
├─ scripts/
│  ├─ bootstrap.sh      每容器/服务器 init：condarc + 环境/依赖/冒烟自检 + 5/5 设备可用性自检
│  ├─ run_all.sh        master 编排（.phase 断点续跑；参数 MODE/PKG_DIR/CONTAINER/DATE/NCARDS/ASCEND_RT_VISIBLE_DEVICES）
│  ├─ mode_daily.sh     日常看护更新（只更新 mindspeed-ops）
│  ├─ mode_version.sh   版本看护更新（全量环境 + mindspeed-ops；PKG_DIR 必填）
│  ├─ nightly_ut.sh     UT（NCARDS 分桶并行 + 挂起防护 + timing）
│  ├─ ut_remain.sh      sinkhorn 900s 限时补跑（只追加 SINKHORN 标记行）
│  ├─ pack_logs.sh      看护结束后打包 UT/ATK 运行日志 → {日期}-log.zip（v0.7）
│  ├─ nightly_atk_acc.sh 精度（NCARDS 并发池 + 慢 op 先入池 + per-op timing）
│  ├─ nightly_atk_perf.sh 性能（生成/对比分支 + per-op 断点 + timing）
│  ├─ step456_run.sh    简易串联 runner（不带 .phase，可替代 run_all 手动分步）
│  ├─ run_retest.sh     复测编排（v0.6：只重跑失败项 + .phase 断点续跑 + 心跳；REF_DIR/DATE 可选）
│  ├─ summarize_retest.py 复测汇总（v0.6：稳定复现/未复现/无法判定 + retest_report.md/json）
│  ├─ summarize.py      结果解析 + owner_map 动态加载（NFS 外置脱敏）+ report.md/report.json（v5）
│  ├─ check_skill.py    ★完整性自检（跨平台 python3，见下节）
│  ├─ deploy_to_nfs.py  ★整体部署到 NFS（含 md5 校验，见下节）
│  └─ helpers/          ★agent 侧运维助手（全参数化，无硬编码）
│     ├─ ssh_cmd.py               paramiko 执行/上传，容器/服务器双模式（env NPC_IP/NPC_USER/NPC_PASS/SUDO_PASS，NPC_SUDO=1 走 sudo 通道，--sftp-put）
│     ├─ poll_run_all.py          轮询 run_all/.phase（DATE/max/interval 参数）
│     ├─ watch_container.py       容器失联自动恢复跟踪（DATE/max 参数）
│     ├─ wait_phase.sh            等 .phase 推进（<OUT_DIR> <cur_phase>）
│     ├─ wait_ut_done.sh          等 ut 阶段结束（<OUT_DIR>）
│     ├─ wait_perf_proc.sh        等 perf 阶段结束（<OUT_DIR>）
│     ├─ check_report_json.py     核对 report.json（<path>）
│     └─ verify_ut_parse.py       回归验证 UT 合并格式可被 summarize 解析
└─ references/
   ├─ README.md            agent 接入指南（快速开始 + 接入步骤）
   ├─ config.json          统一参数（ws/仓库 master/env/atk --max_task 10 -to 1200/复测/日志打包/双模式）
   ├─ owner_map.md         负责人映射外置说明（NFS 动态加载，开源仓库不含人名）
   ├─ environment-setup.md 环境准备要点（conda/CANN/torch/triton/atk/msops 安装 + 验证矩阵）
   └─ vllm_install_notes.md vllm 安装规范（独立环境, 勿装看护 env）
$WS/scripts/ 与上述运行脚本同步（run_all 直接调用 $WS/scripts 下的 nightly_*.sh / run_retest.sh / summarize*.py / ut_remain.sh）
```text
## 归档、自检与部署（v0.6 可迁移）

- **目录即 skill 包根**（自包含：SKILL.md + scripts + templates + references，
  不依赖工作区/服务器上的散件；供其他 agent 读取后按 references/README.md 的"接入步骤"执行）。
- **自检**：`python scripts/check_skill.py`——结构/版本一致性/bash -n 与 py 语法/CRLF/owner_map 外置脱敏与动态加载/文档引用文件存在/垃圾文件，0=通过。
- **部署**：`python scripts/deploy_to_nfs.py`（env: NPC_IP/NPC_USER/NPC_PASS/NFS_WS，服务器另加 NPC_SUDO=1/SUDO_PASS）→
  整包上传 `$WS/skill/` + 运行入口同步 `$WS/scripts/`，远端 md5 与本地逐一比对。
- **接驳**：容器 IP/账号/密码由每次运行的用户输入提供（随机变化），**不写入 skill 包**；
  密码只经环境变量传递，不进日志/报告/命令行。
