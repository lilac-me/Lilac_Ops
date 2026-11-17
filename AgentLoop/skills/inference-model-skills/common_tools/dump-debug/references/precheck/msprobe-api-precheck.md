# msProbe API 级精度预检

## 适用条件

- 精度控制矩阵已完成，但请求仍然乱码、漂移或崩溃。
- 双环境的权重、脚本和环境版本已经按顺序比较完。
- 需要把问题从“环境/框架”缩小到具体 PyTorch API 或 NPU 融合算子。

该流程只回答“某个 API 在 NPU 上相对 CPU 高精度标杆是否异常”。它不替代双环境对比，也不能只凭一个 `warning` 断定根因。

## 零、安装或下载 msProbe

使用前必须已安装 CANN，并确保当前 Python 就是 vLLM 实际使用的 Python。官方安装指南：<https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/install_guide/msprobe_install_guide.md>。

### 0.1 判断是否已安装

```bash
which python3
python3 -V
python3 -c 'import importlib.util; print(importlib.util.find_spec("msprobe"))'
python3 -m pip show mindstudio-probe
msprobe --version
```

- 能导入且版本、路径正常：记录信息，继续第 1 节。
- `find_spec` 返回 `None`、`pip show` 报未安装或 CLI 不存在：先按 0.2 或 0.3 安装。
- 能导入但 CLI 不存在：通常是当前 Python 的 scripts 目录不在 `PATH`。运行 `python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'` 找到该环境的 `msprobe` 可执行文件，修复 `PATH` 后重试，不要重复安装。

### 0.2 在线安装

仅需基础采集和 `acc_check` 时：

```bash
python3 -m pip install mindstudio-probe
```

在线包不包含 `aclgraph_dump`、`nan_check`、`xor_checksum` 等可选模块；这些模块不是本预检流程的必需项。

### 0.3 源码下载与安装

需要最新代码、排查 msProbe 自身逻辑，或要新增/修改 CPU 高精度标杆时，必须使用源码：

```bash
cd /root/workspace
git clone https://gitcode.com/Ascend/msprobe.git -b master
cd msprobe
git rev-parse HEAD
python3 build.py
python3 -m pip install artifacts/mindstudio_probe*.whl
```

源码编译环境须满足 msProbe 官方要求。只需要基础探针能力时使用默认的 `python3 build.py`，不要为本次预检额外启用不需要的插件。

已有源码仓库时：

```bash
cd /path/to/msprobe
git rev-parse HEAD
git status --short
```

不要执行 `git reset --hard`、覆盖本地文件或直接切换分支。需要更新时先 `git fetch origin master`，确认本地修改和版本差异后再决定是否切换或合并。若修改了 `bench_functions`，必须同时确认源码路径和实际安装副本都生效。

### 0.4 安装后验证

```bash
python3 -m pip show mindstudio-probe
python3 -c 'import importlib.util; print(importlib.util.find_spec("msprobe").origin)'
msprobe --version
```

安装路径必须与运行 vLLM 的 Python 环境一致。editable install 时还要记录仓库 commit 和 dirty 状态。

## 一、确认实际 msProbe

```bash
source /path/to/cann/set_env.sh
source /path/to/nnal/atb/set_env.sh

which python3
python3 -V
python3 -c 'import importlib.util; print(importlib.util.find_spec("msprobe").origin)'
python3 -m pip show mindstudio-probe
msprobe --version
```

同时记录：

```bash
python3 -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)'
cd /path/to/msprobe-repo && git rev-parse HEAD && git status --short
```

重点确认 CLI 实际加载的是仓库源码、editable install，还是 `site-packages` 安装副本。只修改仓库而没有同步安装副本时，`msprobe acc_check` 可能仍使用旧代码。

可直接运行：

```bash
bash scripts/audit_msprobe_env.sh
```

## 二、采集真实张量数据

必须使用：

- `task: tensor`
- `level: L1`
- 真实输入数据
- 与故障一致的 step 和请求参数

不能使用 `task: statistics` 或 `mix`。该类数据只适合统计值对比，不能用于 API 精度预检。

### 定向采集配置

```json
{
  "task": "tensor",
  "dump_path": "/root/workspace/dump_api_precheck",
  "rank": [],
  "step": [0],
  "level": "L1",
  "tensor": {
    "scope": [],
    "list": ["npu_fused_infer_attention_score"],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

- `list` 只写可疑算子，减少数据量和预检时间。
- 全量扫描时改为 `"list": []`。
- 多 step 时先只采故障首次出现的 step。

### vLLM-Ascend 接入

在 `--additional-config` 中加入：

```json
{
  "dump_config_path": "/root/workspace/msprobe_config.json"
}
```

启动服务后发送一次与故障一致的请求：

```bash
curl -sS http://127.0.0.1:8106/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

### PrecisionDebugger 接入

在模型加载前启动，在请求完成后停止：

```python
from msprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger(config_path="/root/workspace/msprobe_config.json")
debugger.start()
# 加载模型并执行一次请求
debugger.stop()
```

具体导入路径和生命周期以当前 msProbe 版本文档为准。不要对同一进程重复 start。

### 采集完整性检查

```bash
find /root/workspace/dump_api_precheck -name dump.json -printf '%p %s\n' | sort
find /root/workspace/dump_api_precheck -name '*.pt' | wc -l
```

每个 rank 应同时存在：

```text
step0/rankN/dump.json
step0/rankN/dump_tensor_data/*.pt
```

只有 `dump.json`、没有 `.pt`，通常是配置模式不对、采集过早停止或张量落盘失败。

## 三、执行 acc_check

单 rank：

```bash
msprobe acc_check \
  -api_info /root/workspace/dump_api_precheck/step0/rank0/dump.json \
  -o /root/workspace/msprobe_acc_check/rank0 \
  -save_error_data
```

多个 rank 可批量执行：

```bash
bash scripts/run_msprobe_api_precheck.sh \
  --dump-root /root/workspace/dump_api_precheck/step0 \
  --out-dir /root/workspace/msprobe_acc_check
```

数据量大时使用：

```bash
msprobe multi_acc_check \
  -api_info /root/workspace/dump_api_precheck/step0/rank0/dump.json \
  -o /root/workspace/msprobe_acc_check/rank0 \
  -n 8 -d 0 1 -save_error_data
```

## 四、输出与判定

每个 rank 生成：

```text
accuracy_checking_result_<timestamp>.csv
accuracy_checking_details_<timestamp>.csv
```

### API 级结果

先看 `accuracy_checking_result_*.csv`：

| 状态 | 含义 | 动作 |
|---|---|---|
| `pass` | 当前标准下通过 | 记录 |
| `warning` | 指标接近或超过容忍边界 | 看逐输出指标，不直接归因 |
| `error` | 明确超过精度标准 | 列为高优先级嫌疑 |
| `SKIP` | 未实际执行 | 必须查明原因 |

### 逐输出指标

查看 `accuracy_checking_details_*.csv`：

- 余弦相似度
- 最大绝对误差
- 均方根误差
- 双百、双千、双万指标
- 二进制一致错误率
- ULP 最大值和平均值
- `inf/nan`、相对误差、小值域错误占比

经验规则：

- 余弦接近 `1`、误差为 bf16 量级：优先视为数值容差。
- 余弦明显下降、输出翻转、`inf/nan`：优先怀疑真实精度缺陷。
- `softmax_lse` 正常但 `attention_out` 异常：检查 PV 矩阵乘、输出布局、切片和 dtype。
- 只有一个 rank 异常：检查并行切分、rank 输入、通信和本地缓存。
- 同一算子重复 warning：按调用序号定位首次偏离，不要只看最差值。

## 五、SKIP：缺少标杆函数

典型报错：

```text
SKIP  There is not bench function npu_fused_infer_attention_score
```

这不是通过，而是 msProbe 没有该算子的 CPU 高精度实现。处理步骤：

1. 在实际加载的 `msprobe/pytorch/bench_functions/` 下新增实现。
2. 新版本在 `fusion_operator_config.yaml` 注册。
3. 老版本在 `pytorch/dump/function_factory.py` 中导入并加入 `npu_custom_functions`。
4. 同时同步仓库和实际安装副本。
5. 修改前保存 `.orig` 并记录 `sha256sum`。

FIA 标杆可借助 `install_msprobe_fia_bench.py` 安装与注册。**本技能不内置标杆实现**，
需要按下列契约自备 `--asset` 文件（CPU 参考实现，函数名与算子名一致）：

```bash
python3 scripts/install_msprobe_fia_bench.py \
  --asset /path/to/npu_fused_infer_attention_score.py
```

该标杆应覆盖当前 vLLM-Ascend MLA 场景的：

- `input_layout=TND`
- Q/K 为 `nope 128 + rope 64`
- `scale=1/sqrt(192)`
- `atten_mask` 的 `0` 表示保留
- `sparse_mode=3` 的右下因果区域
- `attention_out` 和 `softmax_lse` 两路输出

脚本会备份被覆盖文件（`.orig`）并打印 `sha256`；注册后必须重启 Python 进程，
再用 `msprobe acc_check` 复跑确认 `SKIP` 消失。标杆实现属于数值参考代码，
必须先在真机与已知正确结果对齐后再使用，不要凭经验直接写入。

其他融合算子必须按其真实 schema 实现，不能用名称相近的算子替代。

## 六、NPU 与 GPU 最终比对

完整文档流程：

1. NPU 环境对 dump 执行 `acc_check`。
2. 将同一份 dump 拷贝到 GPU 环境。
3. GPU 环境执行 `acc_check`。
4. 将两份 `accuracy_checking_details_*.csv` 放到同一目录。
5. 执行：

```bash
msprobe api_precision_compare \
  -npu /path/to/npu_accuracy_checking_details.csv \
  -gpu /path/to/gpu_accuracy_checking_details.csv \
  -o /path/to/compare_out
```

没有 GPU 环境时：

- 只能完成 NPU vs CPU 高精度标杆预检。
- 报告中必须写清 GPU 侧 `acc_check` 和 `api_precision_compare` 未执行。
- 不能把 NPU 侧 warning 直接写成最终根因。

## 七、报告模板

```text
环境：
- 服务器/容器：
- msProbe 版本/路径/commit：
- Python/torch/torch-npu/CANN：
- NPU：

采集：
- dump 配置：
- dump 路径：
- rank 数：
- API 列表：
- 请求：

结果：
- rank0：pass/warning/error/SKIP
- rank1：pass/warning/error/SKIP
- 异常 API：
- 最差指标：
- 首次异常调用：

标杆处理：
- 缺失的 bench function：
- 修改文件及备份：
- 是否同步安装副本：

限制：
- 是否完成 GPU acc_check：
- 是否完成 api_precision_compare：

结论：
- 高优先级嫌疑：
- 已排除项：
- 下一步：
```
