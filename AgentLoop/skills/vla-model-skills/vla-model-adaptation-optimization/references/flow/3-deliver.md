# 交付：patch + test/ + README + MIGRATION_LOG

## 目录结构

```text
model_examples/<ModelName>/
├── README.md                   # 中文，模板见 ../../assets/README.template.md
├── <model>.patch               # 单个 git-apply patch，对应上游 pinned commit
├── mmcv.patch / ...            # 每个三方库一个（感知类才需要）
├── requirements.txt            # NPU 版依赖
├── MIGRATION_LOG.md            # agent 迁移的改动说明（必需产物）：改了哪些 + 为什么
└── test/
    ├── env_npu.sh
    ├── train_full_8p.sh        # 精度
    ├── train_performance_8p.sh # 性能
    └── train_*_multi_server.sh # 可选，多机
```

新式做法（DiffusionDrive / PanoOcc）：不出 `.patch`，改为 `migrate_to_ascend/` 目录，内含 `train.py` / `patch.py` 包装上游。

命名只是**最常用的一套**，实仓并不统一（67 个模型实测）：`test/` 下精度/性能脚本有 `train_full_8p.sh`+`train_performance_8p.sh`（最多）、`train_8p.sh`+`train_8p_performance.sh`（Pi-0）、`train_8p_full.sh`+`train_8p_performance.sh`（SparseDrive）等多种写法；`env_npu.sh` 是多数但非全部（GR00T-N1.7 就没有）；`requirements.txt` 也常见按 torch 版本拆成两份。**核心是「精度一份、性能一份、卡数写进文件名」，不必强求字面一致。**

`MIGRATION_LOG.md` 是**新要求，不要以历史交付为依据**：现仓 0/67 有它（人工适配时代的产物，改动理由在提交者脑子里）。agent 迁移必须留一份可审、可回溯的改动说明（改了哪些 + 为什么），否则开发只能对着 diff 反推。它与该模型的踩坑记录（`<工作区>/<Model>/pitfalls.md`）互链。

## 生成 patch

```bash
cd <upstream_repo>
git checkout -b npu
# ... 改动 ...
git add -A && git commit -m "npu adaptation"
git diff <默认分支>..npu > <model>.patch   # main / master 按上游实际默认分支填
git apply --check <model>.patch      # 无输出 = 通过
```

确认没把二进制 / LFS 文件卷进去（`.gitattributes` 里 LFS 跟踪的 `*.safetensors` / `*.pt` / `*.png`）。

## 冗余审计（交付前必做，凭据对账制）

迁移时会改出一堆**自动化工具已覆盖的代码**（`transfer_to_npu` 的运行时兜底等），这些改动打进度 patch 是多余噪音。**审计凭据从 A 阶段随改随记**（档案卡 `assets/model-card.template.md` 审计凭据段），C 阶段只做对账回退，不靠回忆：

1. **凭据对账**：打开档案卡审计凭据表，逐条核对：「工具覆盖不到? = 是」的改动进入回退验证；「= 否」的确认 diff 里已无手改残留。**凭据只是线索不是豁免**：每一条仍然要按第 2 步验证，防止当时记错。
2. **逐 hunks 回退复查（分层）**：以 `git diff` 为准逐 hunks：显然类（yaml 字面量、纯新增资产、探针已证失败的内核回退）注明豁免理由即可；非显然类（语义分支、设备后端适配、平台枚举）**必须** `git apply --reverse` 试删单处 + 与迁移阶段同判据的短跑负向（前向 + 反向 + optimizer step 跑通，步数取档案卡口径），输出摘要附审计记录，能跑就说明是多余的；不能因为「当时不加不跑」就留着（很多是当时叠加的依赖效应，不是真需要）。**凭据之外，以 `git diff` 为准逐 hunks 复查**：凭据清单 ≠ 改动全集，漏记的改动靠这步兜住。
3. **查自动化覆盖**：`transfer_to_npu` = **白名单函数 × 4 个固定形参名**（`device` / `device_type` / `map_location` / `device_id`），完整范围与边界见 `1-migrate.md` §3.2：**白名单内的源码改动一律从 diff 移除**。**必须留的**：控制流/定义级字面量（`== "nccl"` / `startswith("cuda")` / 模块常量）、`torch.device(...)` 构造对象的下游使用、f-string 设备串、**非白名单的第三方 / vendored 调用点**（`Foo(x, device="cuda")` 这类工具够不到，见 1-migrate.md §3.3）。
4. **归类化简**：同义改动合并（例如多个 `map_location` 改 npu 可以由工具或统一 helper 处理，就不要散落十几处）；能用库级 patch（../ref-ops.md / mx_driving.patcher）解决的，不留逐处手改。
5. **记录取舍**：每个被保留的改动写一句「为什么不能自动化 / 为什么不能删」，被去掉的写「由 XX 工具覆盖」，开发据此判断是否误删。**最终整理进 `MIGRATION_LOG.md`（交付产物，必需）**；过程记录在档案卡审计凭据段的「C 阶段回退验证结果」列（工作区，随改随记）。两者分工：档案卡是过程对账，MIGRATION_LOG 是交付说明，不重复抄。
6. **复测登记**：缩卡口径交付时，`MIGRATION_LOG.md` 标「待 8 卡统一口径复测 ⏳」（按口径注册重建：开箱基线 + 各优化项 A/B）；档案卡「检查点」同步标一份。

审计完 diff 应显著瘦身：常规模型去掉自动化覆盖后，手改通常只剩几十行量级的一组相关改动（设备字符串、fp64、import 裁剪、图模式开关等）。**标了 `[可选优化]` 的不要当冗余删**：那是真实收益，只是可回退；标记约定见 2-optimize.md 开篇纪律。

## MIGRATION_LOG.md（必需产物）

面向**开发 / 维护者**，回答「agent 改了什么、为什么改」。人审 patch 前先读它，不必对着 diff 反推。最小结构：

| 段 | 内容 |
| --- | --- |
| 环境 | 实测通过的版本配对（torch / torch_npu / CANN / 关键三方库）+ 数据与权重来源 |
| 改动清单 | **逐处一行**：`file:line` · 改了什么 · **为什么改**（控制流字面量 / fp64 / attention 后端 / 环境适配…）· **为什么工具兜不住** · 冗余审计结论（保留 / 已删） |
| 优化项 | 有量化收益的项与数值（**复制自档案卡的「优化记录表」**，列定义见 `../../assets/model-card.template.md`；填写要求见 `2-optimize.md` §9）；标 `[可选优化]` 的也列在这，便于回退 |
| 基线 | 口径（卡数 / batch / num_workers / 步数）+ 关键 loss + FPS |
| 复测标记 | 缩卡口径交付时标「待 8 卡统一口径复测 ⏳」 |

与工作区的踩坑记录互链：**那里是给后续迁移 agent 的踩坑，这里是给维护者的改动账**。

## README

结构模板与「准备数据集 / 准备权重」两节的可执行要求见 `../../assets/README.template.md`。要点：

- 表格全局编号：表 1 昇腾版本 / 表 2 三方库 / 表 3 结果
- 结果表行必须成对：`竞品A` + `Atlas 800T A2`（或 A3 / Atlas 950）；无 GPU 基线时注明「未做 GPU 对比」，不要伪造竞品数据
- `变更` 是 append-only 带日期的，包括后续性能优化里程碑
- 数据集与权重两节必须写得**可执行**（获取命令 + 目录结构树 + 校验），不要写「请准备数据集 / 权重」。权重一节还要提醒嵌套 `config.json` 里的绝对路径（如 `backbone` 字段）要改
- 环境搭建一节必须可复现（安装步骤 + 版本配对），不要「请参考官方文档」了事

## 训练脚本

模板见 `../../assets/train_Np.sh.template`。结构：`source test/env_npu.sh` → 默认参数 + `for para in $*` 解析 `--key=value` → 参数校验（路径存在性，提前失败）→ `torchrun ... 2>&1 | tee ${LOG_FILE}` → 抓日志算 FPS。

FPS 提取：**这套脚本依赖训练循环打印一行 `Step <i>/<N>, Loss <x>, StepTime <t>s`**：实仓 188 个 test 脚本里只有 4 个用 StepTime 口径（其余多为 Epoch 计时或 `TrainingTime` 反算），所以**多数模型需要先往训练循环里补这行 print**（照 `train_Np.sh.template` 的注释格式），否则脚本会以「未找到 StepTime 记录」退出、`ActualFPS` 拿不到值。模型本来就有别的计时口径时，改这一段 grep 去适配它即可。

提取逻辑见 `../../assets/train_Np.sh.template` 的「结果提取」段；这里只规定口径：

- `step_time` 取最后 `perf_steps` 步的**均值**（不是中位数），跳过 warmup，与 2-optimize.md §9 同口径
- `FPS = global_batch_size / step_time`；`global_batch_size` 填错会让 FPS **静默**偏掉一个卡数倍，模板的参数校验段有 warning 自检
- 精度脚本的 `FinalLoss` 取日志末行 `Loss <x>`；缺 StepTime 记录时**直接 exit 1**（不要降级成 `N/A` 继续跑）
- 精度跑与性能跑是两次独立运行，不合并

ModelZoo 标准收尾块参考 `Pi-0/test/train_8p.sh`（注释写明「不需要修改」，输出 `ActualFPS / TrainingTime / ActualLoss / E2ETrainingTime`：**评审工具按名抓取，这四个名字不要改**）。

`TrainingTime` 与 `ActualFPS` 是同一个数的两种表达：语料里的标准写法是 `TrainingTime = batch_size × 1000 / FPS`（即「1000 步耗时」，15+ 个模型这么写），模板按 `max_steps × step_time` 给、在 `max_steps=1000`（默认）时**代数上等价**。**实跑步数 ≠ 1000 时先跟评审口径对齐**，不要让两个字段互相冲突。`E2ETrainingTime` 是另一回事：脚本全程墙钟，含启动/数据准备/收尾。**序列长度主导的模型**（视频 / 扩散 / VLM）样本间长度差异大、`ActualFPS` 跨模型不可比，收益判定看 profiling 的算子时间和自算的 tokens/s，但**上报仍按 ModelZoo 字段**。

已验证的优化项 = 模型默认：**有量化正收益**的项直接写进模型默认路径，能直接固化就不留开关；中性/负收益项回退基线（裁判见 2-optimize.md §9）。

## 交付前检查

**train.py 新增参数必须在 test/*.sh 透传**，否则按 README 使用脚本时这些功能等同于不存在（实践中出现过）：

```bash
# train.py 里所有 "--xxx" 参数（含连字符/大写），逐个查是否出现在 test/*.sh 中
grep -ohE '"--[A-Za-z0-9_-]+"' train.py | tr -d '"' | sort -u \
  | while read -r a; do grep -q -- "$a" test/*.sh || echo "未透传: $a"; done
```

不要用 `add_argument\("--(\w+)"` 这种单行正则，`add_argument(` 后面换行的写法会被漏掉。性能脚本要透传优化开关。**profiling 不必做成命令行参数**，脚本里用开关控制即可（如 `PROFILE=1 bash train_8p.sh` 时启用 profiler），避免交付后仍需用户拼接 `--profile_steps` / `--profile_dir` 三个参数。精度脚本不接优化开关，但数据 / 步数 / seed 参数要齐。卡数要覆盖实际可用规模（只给 16p 脚本落到 8 卡机器就没法复现）：每个卡数一份脚本或让脚本支持 `--num_npus=`。

**交付件里禁止残留调试变量**：`ASCEND_LAUNCH_BLOCKING=1` 只能 debug 阶段临时用（算子同步下发，会使性能大幅下降，并让 `TASK_QUEUE_ENABLE` 整体失效）；`ASCEND_*_LOG_LEVEL=0`（DEBUG）同理。交付前自查：

```bash
grep -rn "ASCEND_LAUNCH_BLOCKING\|ASCEND_GLOBAL_LOG_LEVEL=0\|ASCEND_SLOG_PRINT_TO_STDOUT=1" test/ \
  && echo "↑ 交付件里残留了调试变量，必须删掉" || echo "调试变量检查通过"
```

语料 67 个模型的 test 脚本里 `ASCEND_LAUNCH_BLOCKING` **一次都没出现过，出现就是漏删**（危害机制与官方约束见 `../appendix-env.md` §四）。同理，`test/env_npu.sh` 应当只保留基线项（`../../assets/env_npu.sh` 的那几条 + 按场景开的），**禁止往里面累加变量**：VLA 场景大部分环境变量用不上，写默认值等于不写（判据见 `../appendix-env.md`）。

**`git apply` 前不要对工作区做多余操作**：`.patch` 是对上游 pinned commit 的干净 checkout 生成的，`git apply` 要求目标工作区与基座一致。交付物里任何会覆盖同名文件的操作（拷入交付的 `requirements.txt` / `test/` / 修改过的配置文件）必须先 `git apply` 再覆盖，顺序反了 patch 就 `does not apply` 或打不干净：

```sh
git checkout <commit_id>
git apply --reject <model>.patch     # 1. 先在干净 checkout 上打补丁
cp -rf ../test/ ./ && cp -f ../requirements.txt .   # 2. 再覆盖交付文件
pip install -r requirements.txt
```

## 三份文档的分工

| 文件 | 受众 | 内容 |
| --- | --- | --- |
| `model_examples/<M>/README.md` | 外部用户 | 怎么复现：环境、数据、命令、结果 |
| `model_examples/<M>/MIGRATION_LOG.md` | 开发 / 维护者 | **agent 迁移的必需产物**：每处改动「改了哪些 + 为什么改 + 为什么不能自动化/不能删」，加上完整数据与基线。人审 patch 前先读它 |
| 工作区 `<工作区>/<Model>/pitfalls.md` | 后续迁移的 agent | 踩坑条目 + 一行基线；不写环境细节 / 流水账 / 方法论重述 |

内容有重叠但受众不同，不合并；交付后两边互相引用路径。完成一个新模型迁移后，在 `../ref-models.md` §已完成案例 加一行。
