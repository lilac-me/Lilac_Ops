---
name: cann-package-split-diagnose
description: >-
  This skill should be used when the user asks to [cann拆包定位][cann问题组件定位][cann组件二分定位].
   对于 Ascend 昇腾环境下的 vLLM 等服务运行异常，通过拆分 CANN 安装包的 toolkit /nnal/ops 三个子包，再进一步拆解 ops 包的多个组件，逐步二分，最终定位到 具体是哪个子包、哪个组件 出的问题。
version: 1.0.0
domain: common
function: common
tags:
  - vllm-ascend
  - cann
author: wuruiyun
targets-repo:
  - cann
---

# CANN 拆包定位(cann-package-split-diagnose)

## 用途

对于 Ascend 昇腾环境下的 vLLM 等服务运行异常，通过拆分 CANN 安装包的 **toolkit / nnal / ops 三个子包**，再进一步拆解 ops 包的多个组件，逐步二分，最终定位到 **具体是哪个子包、哪个组件** 出的问题。

> 核心思想：先用"好包/坏包"建立基线，再逐层替换，每步只改一个变量，通过**用户定义的判定标准**来判断。

### 判定标准：以用户输入为准，不要臆断

**本技能不预设"正常/异常"的判断标准**。异常现象可能是乱码、报错、性能下降、结果错误等，由**用户提供**，例如：

- 乱码（如本次案例：输出 `"ikasi清建建..."` 之类无意义文本）
- 具体报错（如 `ImportError: libhcomm.so`、算子失败、OOM）
- 性能明显变慢 / 吞吐下降
- 输出结果错误等

开始前**必须先向用户确认**：(1) 异常现象具体是什么（拿来当判定信号）；(2) 如何复现/验证这个信号（命令、接口、日志或指标）。后续每一轮都用同一个信号判断该组合是"正常"还是"异常"。

---

## 目录结构

```plaintext
cann-package-split-diagnose/
├── SKILL.md                  # 本说明文档
└── scripts/
    ├── ssh_utils.py          # SSH 连接与上传工具（基于 paramiko）
    └── split_diagnose.py     # 拆包定位辅助脚本（交互式生成安装/启动/替换脚本）

```

前置依赖：目标机器可 SSH 登录，`paramiko` 已安装，环境为 aarch64 + Ascend NPU。

---

## 关键概念速查

| 概念 | 说明 |
| ------ | ------ |
| **ops 可独立替换的组件** | 只有 **6 个**组件可以单独升级替换：组A = `hccl` + `hixl` + `cv`，组B = `math` + `nn` + `transformer`。这 6 个之外的组件（legacy/opbase/dvpp/ras 等）**不能单独替换**，若问题落在它们头上只能整包换 |
| **tk/nnal 是否拆** | **不能拆着换**。toolkit、nnal 只能作为整包子包来装/判断，无法单独替换其中部分组件。若确认问题在 tk 或 nnal，直接生成报告说明是哪个子包，不再下拆 |
| **安装顺序** | toolkit → nnal → ops；nnal 安装前需 source toolkit 的 set_env.sh，ops 安装前需 source toolkit + nnal 的 set_env.sh |
| **整包安装** | `./Ascend-cann-<xxx>_<ver>_linux-aarch64.run --install --install-path=<path> --quiet` |
| **子组件替换** | 必须用 **`--full --install-path=<path>`**；用 `--install` 或 `--run` 会报 RC=1 + 打印 Usage，替换不生效 |
| **同一目录** | 所有子包、子组件必须装到**同一个** `--install-path=<path>` 目录，才能组合成一套环境 |

---

## 输入信息

| 输入项 | 说明 |
| ------ | ------ |
| 服务器连接 | host / port / user / password（IP/密码可能动态变化） |
| **判定标准/异常现象** | **必填**：用户提供的异常信号 + 验证方法，用于每轮判断正常/异常 |
| good 包目录 | 包含 3 个 .run（tk/nnal/ops）的 good 包 |
| bad 包目录 | 包含 3 个 .run（tk/nnal/ops）的 bad 包 |
| vLLM 源码目录 | 运行 `python -m vllm.entrypoints.openai.api_server` 的目录（需 editable install） |
| **vLLM 启动命令** | **用户提供**（含 tensor-parallel、data-parallel、expert-parallel、端口等全部参数，原样使用） |
| 需要多少张卡 | **由启动命令的并行数决定，不是固定**；向用户确认当前可用的空闲卡 |
| 验证命令 | 复现/验证异常信号的具体命令（接口调用、日志查看、指标读取等） |
| 工作目录 | 装包、日志放到工作目录（installs） |
| 安装根目录 | 用户提供 |

---

## 前置检查

| 检查项 | 方法 |
| ------ | --------- |
| 判定标准 | 与用户确认异常信号与验证命令，明确"正常/异常"怎么判 |
| NPU 空闲数 | `npu-smi info`，确认**空闲卡数 ≥ 启动命令所需卡数**（按需，可能 1 张即可） |
| 是否会误杀他人 | 使用前确认该机器是否有多人共用；若有，所有进程操作必须**按 PID**，禁止全局 pkill |
| vLLM 是否可导入 | `pip list \| grep vllm`，确认 editable 指向你的源码目录 |
| ops 包类型 | 看 .run 文件名是 A3-ops 还是 950-ops |
| CANN 版本 | 看 .run 文件名确认版本（如 9.1.0） |

---

## 标准工作流

### 第 0 步 准备

1. 与用户确认：异常现象、验证命令、vLLM 启动命令、需要的卡数、机器是否多人共用。
2. 登录后设置 PATH（含 python bin），确认 `vllm` / `vllm-ascend` 为 editable 安装。
3. 把 good、bad 两套 3 个 .run 分别上传到 `packages/good`、`packages/bad`。
4. 检查 NPU：`npu-smi info`，确认空闲卡数满足启动命令的 tensor-parallel-size。

### 第 1 步 建立基线

**1a. 全 good 跑通（必须是"判定标准里的正常"）**

- 全新安装到 `installs/good`（tk→nnal→ops 顺序）。
- source CANN 环境，用**用户给的启动命令**启动 vLLM，用**用户的验证命令**验证，期望得到"正常"结果（作为 good 基线）。
- 验证完**务必清理 NPU**（按 PID kill，见坑 1）。

**1b. 全 bad 跑通（必须是"判定标准里的异常"）**

- 全新安装到 `installs/bad`，同样流程，确认**异常**。
- 用 good 对比 bad，确认复现稳定。

> 若 good 也异常说明环境本身不行；若 bad 也正常说明问题不在该包。

### 第 2 步 二分定位子包（tk / nnal / ops）

装 **good tk + good nnal + bad ops** 的组合：

- 独立全新安装到 `installs/combo1`（good tk、good nnal、**bad ops**）。
- **严禁直接用 `cp -a` 复制 good 环境**（见坑 2）。
- 判断：
  - **异常 → 问题必在 ops 子包**（进入第 3 步）。
  - 正常 → 问题在 tk 或 nnal：分别用 bad tk、bad nnal 替换定位是哪个子包，
    **直接生成报告**说明问题在 toolkit 或 nnal 子包，**不再下拆**（tk/nnal 不能拆着换）。

### 第 3 步 拆解 ops，二分到组件（只拆 6 个可独立替换的组件）

在 combo1（good tk + good nnal + bad ops）基础上，只对 **6 个可独立替换的组件**做二分：

- 组A：`hccl` + `hixl` + `cv`
- 组B：`math` + `nn` + `transformer`

用 good 的对应组件 `.run` 以 `--full --install-path=<combo1>` 覆盖，每覆盖后启动验证（替换为"当前认为坏"的组合目录）。

**第一轮：分组二分**

- 把**组A（hccl+hixl+cv）**覆盖为 good，其余保持 bad，验证：
  - **异常 → 问题在组A**（进入第二轮 A 组内逐个定位）
  - **正常 → 问题在组B**（进入第二轮 B 组内逐个定位）

**第二轮：在定位到的组内逐个定位**

- 在命中的组（A 或 B）内，逐个把组件覆盖为 good（每次只换一个、其余保持 bad），验证：
  - 某组件换为 good 后由"异常"变"正常" → 该组件即为问题组件。
  - 逐个都换完仍无变化 → 该组内定位不到，进入"不在 6 组件内"分支。

> 判定说明：每轮"覆盖为 good"都只改变一个变量（其余保持 bad）。

**不在 6 组件内的处理**

- 如果第一轮组A、组B 结果都正常（或第二轮组内逐个定位都定位不到），说明坏点**不在这 6 个可独立替换的组件里**（可能在 legacy/opbase/dvpp/ras 等不可拆分组件，或 ops 整包层面）。
- 此时**直接生成报告**说明：问题在 ops 子包，但不属于可独立替换的 6 个组件，建议整包替换 bad → good。

### 第 4 步 输出报告

汇总：用户判定标准 + 基线结果 + 子包二分 + 组件二分（每轮组合与结果）+ 最终定位结果 + 修复建议。**直接打到屏幕展示**，不需另存文件。

- 问题在 tk/nnal → 报告定位到子包即可，不做组件拆分。
- 问题在 ops 的 6 组件内 → 报告定位到具体组件。
- 问题在 ops 但不在 6 组件内 → 报告说明"不在可独立替换组件内，需整包换"。

---

## 安装 / 替换命令速查

```bash
export PATH=<python_bin_dir>:$PATH     # 以实际 python bin 为准
INST=<install_dir>

# 1) toolkit
./Ascend-cann-toolkit_<ver>_linux-aarch64.run --install --install-path=$INST --quiet

# 2) nnal（先 source toolkit 环境）
source $INST/cann/set_env.sh
./Ascend-cann-nnal_<ver>_linux-aarch64.run --install --install-path=$INST --quiet

# 3) ops（先 source toolkit + nnal 环境）
source $INST/cann/set_env.sh
source $INST/nnal/atb/set_env.sh
./Ascend-cann-<A3-ops|950-ops>_<ver>_linux-aarch64.run --install --install-path=$INST --quiet

# 4) 拆分 ops 大包，拿到各子组件 .run（run_package/）
./Ascend-cann-<ops-type>_<ver>_linux-aarch64.run --noexec --extract=<extract_dir>

# 5) 用 good 的某个组件替换组合目录里的 bad 组件（关键：必须 --full）
source $INST/cann/set_env.sh
source $INST/nnal/atb/set_env.sh
<extract_dir>/run_package/cann-<ops-type>-<component>_<ver>_linux-aarch64.run --full --install-path=$INST --quiet
```

验证：按用户的验证命令执行，用第 0 步确认的判定标准判断正常 / 异常。

## 实战经验与坑

### 1. 清理 NPU 必须按 PID，禁止全局 pkill

- 机器可能有**其他人也在用卡 / 跑 vLLM**。`pkill -9 -f vllm.entrypoints...` 这类按命令名模糊匹配会**误杀别人的进程**，绝对禁止。
- 正确的做法：记录**自己启动的进程 PID**，结束时只 kill 自己的 PID，及其子进程链。

  ```bash
  # 启动时记录 PID
  nohup <你的启动命令> > vllm.log 2>&1 &
  echo "PID=$!"                       # 记录这个 PID

  # 结束时按 PID 精确 kill（含其下挂的 EngineCore / Worker 子进程）
  kill <PID>                          # 先优雅停主进程
  # 再按 PID 逐个清掉子进程，不要用 -f 模式匹配
  ps -ef | grep -E 'vllm|EngineCore|VLLMWorker'   # 仅查看
  kill -9 <具体PID1> <具体PID2> ...               # 只 kill 属于自己会话的
  npu-smi info   # 确认自己占的卡已释放，且没动到别人的卡
  ```

- 多卡场景只 kill 自己进程；不确定归属的进程**不要动**。

### 2. 严禁用 `cp -a` 复制环境做组合

- 复制目录会保留 `cann -> /原good路径` 的**软链**，且 set_env.sh 里 **version_dirpath 硬编码了绝对路径**（如 `.../good/cann-<ver>`）。
- 组合环境会被指回原 good 目录，**污染原 good 的 hccl 库**，导致 libhccl.so 加载到旧 / 错位置。
- 因此每次组合必须**独立全新安装**，不要复制。

### 3. 判定 "装好" 要看产物，不要只看进程

- toolkit 安装完应有 `libhcomm.so`（hccl 相关），缺了会导致 vLLM `ImportError: libhcomm.so` / `libhccl.so`。
- 判定标准：**日志 RC=0 + 出现 ALL_DONE**，且**关键产物文件存在**（如 `libhcomm.so`、`libhccl.so`）。
- **不要**用 nohup 进程是否退出 / 挂起来判断（后台安装进程常被误判为 "还没装完"，实际早装好了）。

### 4. 子组件替换必须用 `--full`

- 子组件 .run **不支持 `--install`**（会打印 Usage + RC=1）。
- 统一用 `--full` `--install-path=<path>` 才能正确覆盖到组合目录。
- 同一套组合目录里，各组件都可以这样逐个 / 成组替换。

### 5. 慢 / 挂起的处理

- 装到 NFS 极慢（每个子包几分钟到半小时），优先装到本地盘（如有路径权限）。
- 用 `nohup bash script > log &` + 轮询 `log` 里的 exec 阶段 /`ALL_DONE` 判断完成，不要干等。

### 6. 组件清单按实际包来，不预设

- 每次 `--noexec --extract` 后，**以 run_package/ 里实际存在的组件为准**来分组二分，不要照搬上次的组件名单或结论。

### 7. 服务器 IP / 密码动态变化

- 容器 IP / 密码会变，连不上时向用户确认最新连接信息再操作。

## 报告模板

> **直接打印到屏幕展示即可，不需另存文件。** 报告应包含：用户判定标准、
环境、基线、子包二分、组件二分、最终结论六部分。组件二分结果是**本次
实际二分**得出的，不要写死成 transformer 之类固定结论。

```text
========================================
        CANN 拆包定位最终报告
========================================
定位时间 : <时间>
服务器   : <host>
CANN版本 : <ver>        Ops类型 : <A3-ops/950-ops>
启动命令 : <用户提供的启动命令，含 tensor-parallel 等参数>

[判定标准 (用户定义)]
异常现象 : <乱码/具体报错/性能问题...>
验证方法 : <如何复现与判断正常/异常>

[基线]
全 good 包 -> <正常>
全 bad 包  -> <异常>
=> 问题在 CANN 包，可复现

[子包二分]
good tk + good nnal + bad ops  -> <异常>
=> 问题在 ops 子包 (Ascend-cann-<A3-ops|950-ops>_<ver>_linux-aarch64.run)

[组件二分] (仅 ops 可独立替换的 6 组件)
第一轮(组A=hccl+hixl+cv 换为good, 组B保持bad)  -> <异常/正常> -> 问题在<组A/组B>
第二轮(组内逐个换为good)  -> 逐行列出结果 -> 定位到组件: <组件名>
(若两轮均定位不到 => 不在6组件内)

[结论] (按实际路径二选一/三选一)
- 问题在 tk/nnal 子包 => 直接报告定位到子包，不再下拆
  坏子包: <Ascend-cann-toolkit_<ver> 或 Ascend-cann-nnal_<ver>>
- 问题在 ops 的 6 组件内 => 定位到具体组件
  坏子包: Ascend-cann-<A3-ops|950-ops>_<ver>_linux-aarch64.run
  坏组件: cann-<ops-type>-<组A/组B内命中的组件>
  修复  : 用 good 的该组件 --full 覆盖后重启 vLLM 验证
- 问题在 ops 但不在 6 组件内 => 直接报告说明，建议整包替换 bad -> good
```
