# 双环境差异定位

## 适用条件

- 精度控制矩阵已完成，但仍无法定位。
- 用户提供已知正确和已知错误两套环境。
- 两套环境可以在两台服务器上，也可以在同一台服务器的两个路径下。

先确定 `GOOD` 和 `BAD`，再按 **权重 -> 脚本 -> 版本** 顺序比较。

## 证据采集表

| 项目 | GOOD | BAD | 是否一致 |
|---|---|---|---|
| 服务器/容器 |  |  |  |
| 工作目录 |  |  |  |
| 服务端脚本 |  |  |  |
| 客户端脚本 |  |  |  |
| 模型路径 |  |  |  |
| 模型真实路径 |  |  |  |
| Python |  |  |  |
| vLLM 路径/版本/commit |  |  |  |
| vLLM-Ascend 路径/版本/commit |  |  |  |
| CANN |  |  |  |
| torch/torch-npu |  |  |  |
| 启动参数 |  |  |  |
| 请求参数 |  |  |  |

## 第一步：权重对比

### 何时执行

只要不确定两侧是否使用同一份权重，就执行。以下情况必须执行：

- 权重路径不同
- checkpoint 名称不同
- 可能通过软链接、容器挂载或 overlay 看到不同内容
- 一侧曾做过权重转换、合并、量化或裁剪

### 基础对比

```bash
python3 scripts/compare_model_dirs.py \
  --good /path/to/good/model \
  --bad /path/to/bad/model
```

默认比较文件清单、文件大小和小文件哈希。大权重可执行全量哈希：

```bash
python3 scripts/compare_model_dirs.py \
  --good /path/to/good/model \
  --bad /path/to/bad/model \
  --hash-mode all
```

大模型全量哈希耗时较长，至少应对所有 `*.safetensors` 分片执行一次，或使用可信的权重校验工具。

### 额外检查

```bash
readlink -f /path/to/model
find /path/to/model -maxdepth 1 -type f -printf '%f %s\n' | sort
sha256sum /path/to/model/config.json /path/to/model/*.index.json
```

检查：

- `config.json` 中架构、hidden size、层数、vocab size
- tokenizer 文件是否一致
- safetensors index 的分片映射
- 分片数量、每片大小
- 权重生成时间、转换脚本和源 checkpoint

### 判定

| 结果 | 处理 |
|---|---|
| 文件清单、大小、关键哈希全部一致 | 权重不是首要差异变量，进入脚本对比 |
| 清单一致但哈希不同 | 权重是首要嫌疑，先统一权重复跑 |
| 清单或分片不同 | 不能把两侧视为同一权重基线 |
| 同一路径但行为不同 | 检查容器挂载、软链接、overlay 和启动时的实际路径 |

## 第二步：脚本对比

```bash
diff -u /path/to/good/serve.sh /path/to/bad/serve.sh
diff -u /path/to/good/client.sh /path/to/bad/client.sh
```

同时读取脚本展开后的实际命令：

```bash
bash -x /path/to/serve.sh
```

或在不启动服务的情况下打印关键变量：

```bash
bash -c 'set -x; source /path/to/serve.sh' 2>&1 | head -n 200
```

重点字段：

- 模型路径、served model name
- `--enforce-eager`
- tensor/pipeline/data parallel
- dtype、量化、KV cache dtype
- max model len、max num seqs
- `--additional-config`
- `--chat-template` / `--tokenizer` / `--served-model-name` 等影响输入组装的参数
- 自定义环境变量和 patch
- 客户端 temperature、top_p、top_k、seed、max_tokens、logprobs

只忽略端口、日志路径等明确无关项，并在报告中记录每个忽略理由。

## 第三步：环境版本对比

在两侧分别采集：

```bash
bash scripts/audit_vllm_ascend_env.sh > /tmp/vllm_env_GOOD.txt
bash scripts/audit_vllm_ascend_env.sh > /tmp/vllm_env_BAD.txt
diff -u /tmp/vllm_env_GOOD.txt /tmp/vllm_env_BAD.txt
```

如果两侧在不同服务器，把两份报告下载到同一目录后 diff。

额外采集：

```bash
python3 -m pip freeze | sort > /tmp/pip_GOOD.txt
python3 -m pip freeze | sort > /tmp/pip_BAD.txt
diff -u /tmp/pip_GOOD.txt /tmp/pip_BAD.txt
```

必须确认：

- `which python3`
- `python3 -c 'import sys; print("\\n".join(sys.path))'`
- `vllm.__file__` 和 `vllm_ascend.__file__`
- git commit 和工作区是否 dirty
- CANN/NNAL/ATB/driver/firmware
- torch、torch-npu、transformers、triton-ascend
- 自定义算子和已安装 patch

## 因果验证矩阵

不要只做静态 diff。按单变量替换验证：

| 组合 | 权重 | 脚本 | 环境 | 目的 |
|---|---|---|---|---|
| 1 | GOOD | BAD | BAD | 判断权重是否致因 |
| 2 | BAD | GOOD | BAD | 判断脚本是否致因 |
| 3 | BAD | BAD | GOOD | 判断环境版本是否致因 |
| 4 | GOOD | GOOD | BAD | 排除权重和脚本 |
| 5 | BAD | GOOD | GOOD | 单独保留 BAD 权重 |

每次替换只改变一个变量。每个组合用同一请求连续跑 3 次，并保存原始响应、日志和环境信息。

数据集（E3/C 分支）场景补一组**输入口径**验证：同一批 prompt 同时打两侧，按
`references/precheck/prompt-template-alignment.md` 核对输入 token 长度——固定偏移说明模板/配置
未对齐，此时上面的权重/脚本组合结论都不成立；只有输入对齐后，精度差异才可归因。

## 结果记录模板

```text
GOOD 环境：
BAD 环境：

权重：一致 / 不一致
- 不一致项：

脚本：一致 / 不一致
- 关键差异：

版本：一致 / 不一致
- 关键差异：

单变量替换结果：
- 换权重：
- 换脚本：
- 换环境：

结论：
- 首要根因：
- 已排除项：
- 下一步验证：
```
