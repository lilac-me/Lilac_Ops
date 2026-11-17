# 运行时、门禁与恢复合同

## 目录

1. 环境所有权
2. Shell 入口
3. Transformers 隔离环境
4. 运行前回归
5. 阶段状态
6. 产物失效矩阵

## 1. 环境所有权

MindSpeed 设备环境属于用户。模型 launcher 只能继承当前活动环境：

- 不 source CANN、Conda、venv 或 toolkit 脚本；
- 不安装、升级或卸载基础环境包；
- fresh shell 中验证 `sys.executable` 和必需 import；
- 把缺失依赖报告为环境前置条件；
- `BASE_PYTHON` 不能是 `/absolute/path/...`、相对路径、不可执行文件、
  HF venv Python 或无法导入 MindSpeed/Megatron 的解释器。

记录解释器词法路径、resolved binary、`sys.prefix` 和 `sys.base_prefix`。
不能仅因 `bin/python` resolve 到基础二进制就断定不属于 venv。

## 2. Shell 入口

每个改变目录、变量、依赖或启动分布式任务的 `.sh` 必须：

```bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "请使用 bash script.sh ...，不要 source 此脚本" >&2
  return 2
fi
```

MCore 子进程可临时设置：

```bash
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${BASE_PYTHON}" ...
```

不得在父 shell 执行覆盖式 `export PYTHONPATH="$REPO_ROOT"`。不得调用
`conda activate` 或 `source .../activate`。

launcher 应清晰输出阶段：环境验证、门禁、运行时回归、模型构造、strict load、
forward、保存、比较、最终化。早退时说明最后完成阶段及是否进入模型加载/forward。

## 3. Transformers 隔离环境

依赖冲突时创建模型独占 venv：

- setup 是显式动作；
- 必要时使用 `--system-site-packages` 复用加速版 torch/torch_npu；
- 从所选 Transformers checkout 的依赖表/项目元数据读取精确约束；
- 在 venv 内以 `--upgrade` 安装缺失或不兼容依赖；
- 不安装或替换 torch/torch_npu；
- pip 命令设置 `PIP_REQUIRE_VIRTUALENV=true`，并清理继承的
  `PYTHONPATH`/`PYTHONHOME`；
- 支持离线 wheelhouse；
- tokenize 与 HF forward 使用同一绝对 venv Python。

运行前写出 JSON：

```text
python launcher/resolved binary
sys.prefix/sys.base_prefix
transformers.__file__
所选 checkout
依赖版本与约束
基础环境关键包前后对比
```

`transformers.__file__` 必须落在用户指定 checkout。缺少显式路径时，只能使用
经过验证且无歧义的仓库相对 fallback；不得静默使用基础环境的 Transformers。

## 4. 运行前回归

设备 forward 前要求：

- 输入/hash/schema 检查；
- checkpoint 与层身份静态检查；
- 每个 mask/in-place/fused 边界的行为回归；
- RoPE 使用真实 HF 与 MCore 函数比较非零、跨位置 QK score；
- 融合算子在目标 runtime 执行，并通过生产布局恢复函数；
- 已知必要的 module-output synchronization 默认启用并写入元数据。

源码字符串检查只能作为辅助证据，不能替代真实 tensor 行为。

## 5. 阶段状态

正常通过：

```text
numeric_pass=true
gate_bypassed=false
diagnostic_only=false
sequential_precision_certified=true
```

直接前驱诊断绕过：

```text
numeric_pass=<本阶段比较结果>
gate_bypassed=true
diagnostic_only=true
sequential_precision_certified=false
```

用户跳过阶段：

```text
state=skipped
skipped_by_user=true
skip_reason=<非空>
numeric_comparison_executed=false
passed=false
diagnostic_only=true
sequential_precision_certified=false
```

绕过值只能匹配目标阶段的直接前驱。未知值或更早祖先必须拒绝。下游每次只沿一个
直接前驱继续，且全链保持 diagnostic-only，直到从最早未认证阶段严格重跑。

## 6. 产物失效矩阵

| 变化 | 固定输入 | HF logits | MCore logits | 比较报告 |
| --- | --- | --- | --- | --- |
| prompt/tokenizer/mask | 失效 | 失效 | 失效 | 失效 |
| HF 代码/checkpoint/dtype/同步 | 可复用 | 失效 | 可复用* | 失效 |
| MCore 代码/checkpoint/并行/flag | 可复用 | 可复用* | 失效 | 失效 |
| 比较器/阈值 | 可复用 | 可复用* | 可复用* | 失效 |
| preflight 或文档 | 通常可复用 | 依合同 | 依合同 | 依合同 |

`*` 表示复用前必须通过 producer 合同校验。不得只检查文件存在。

复用验证至少覆盖：

- 输入 hash 和嵌入输入；
- finite FP32 `[B,S,V]`；
- checkpoint/code identity；
- producer/tool 版本；
- 执行 flag、并行布局和同步屏障；
- 原始/保存词表大小。

只修复一侧 producer 时提供窄范围 refresh 模式，保留经验证的另一侧产物。
