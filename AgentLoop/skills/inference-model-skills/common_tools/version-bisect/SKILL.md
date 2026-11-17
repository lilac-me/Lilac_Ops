---
name: version-bisect
description: >-
  This skill should be used when the user asks to [通过二分定位问题版本].
   通过二分查找法快速定位vLLM推理服务中引入问题的具体版本/commit。支持三种定位维度：CANN软件栈版本、torch_npu版本、vllm-ascend commit。适用于vLLM推理服务异常排查、版本回归分析等场景。当用户提到二分定位、查找问题版本、版本回归分析时使用此技能。
version: 1.0.0
domain: common
function: common
tags:
  - cann
  - torch_npu
  - vllm-ascend

author: wuruiyun
targets-repo:
  - cann
  - torch-npu
  - vllm-ascend
---

# 版本二分定位技能（version-bisect）

## 目录结构

```text
version-bisect/
├── SKILL.md                      ← 本文件，工作流总入口
├── scripts/
│   ├── ssh_utils.py              ← SSH连接 + Docker/Conda环境执行器
│   ├── bisect_cann.py            ← CANN版本二分核心脚本
│   ├── bisect_torch_npu.py       ← torch_npu版本二分核心脚本
│   └── bisect_vllm_ascend.py     ← vllm-ascend commit二分核心脚本
└── reference/
    ├── cann-bisect.md            ← CANN版本二分详细流程
    ├── torch-npu-bisect.md       ← torch_npu版本二分详细流程
    └── vllm-ascend-bisect.md     ← vllm-ascend commit二分详细流程
```

## 适用场景

| 定位维度 | 二分对象 | 适用场景 | 核心脚本 |
| --------- | --------- | --------- | --------- |
| **CANN** | CANN安装包版本（.run） | CANN版本升级导致异常 | `bisect_cann.py` |
| **torch_npu** | torch_npu `.whl` 包版本 | torch_npu版本升级导致异常 | `bisect_torch_npu.py` |
| **vllm-ascend** | vllm-ascend git commit | vllm-ascend代码变更导致异常 | `bisect_vllm_ascend.py` |

## 使用流程

### 第一步：确认定位维度

通过交互式询问用户确认需要定位的维度（1. CANN / 2. torch_npu / 3. vllm-ascend）。

### 第二步：环境准备

#### 运行环境适配

本技能**假设运行环境已经就绪**，由 **environment-install** 技能完成。在执行二分前应确认当前环境类型：

| 环境类型 | 判断方式 | 操作说明 |
| --------- | --------- | --------- |
| **Docker环境** | 容器内存在 `/.dockerenv` 文件 | 命令通过 `docker exec` 进入容器执行 |
| **Conda环境** | 存在环境变量 `CONDA_DEFAULT_ENV` | 激活Conda环境后执行 |

#### 用户输入

开始工作流前，通过**交互式输入**获取以下**所有**信息。敏感信息禁止硬编码：

| 参数 | 说明 | 所有维度通用 |
| ------ | ------ | :---: |
| 远程服务器信息 | host/port/user/password/ssh_key | ✓ |
| 运行环境路径 | Conda环境名或Docker容器名 | ✓ |
| vLLM拉起脚本内容 | 启动服务的shell脚本内容（由用户提供，自行编写） | ✓ |
| API验证命令 | 测试服务是否正常的命令 | ✓ |
| vLLM服务端口 | 用于启动后精确杀进程和健康检查 | ✓ |
| 验证成功标准 | 判定"正常"的条件（如"response包含choices"） | ✓ |
| 代理配置（可选） | 如需连接外网则配置代理 | ✓ |

各维度特有参数见 `docs/` 下对应的文档。

#### 公共注意事项

1. **JSON引号问题**：包含JSON的参数写入文件后执行，避免shell嵌套引号截断
2. **双连接架构**：验证API时使用单独SSH连接（不配置代理）
3. **NPU资源检查**：启动vLLM前检查NPU卡空闲情况，不kill已有进程
4. **base64避坑**：传输复杂shell脚本时用base64编码后写入
5. **进程管理**：
   - 启动vLLM时记录PID（`nohup ... & echo $!`）
   - 关闭服务时使用 `kill <PID>` 精确杀进程，**禁止使用 `pkill -f`**
   - 如果不知道自己服务的PID，根据vLLM服务端口查找：`lsof -ti:<port>` 或 `fuser <port>/tcp` 获取PID后精确kill
   - **绝对禁止** kill 他人进程，只杀自己启动的服务
6. **用户提供启动脚本**：用户提供vLLM拉起脚本内容（不是服务器上的路径），由执行者将其写入到服务器的临时文件中再执行。脚本中的端口号等参数应与用户提供的服务端口一致
7. **任何时候遇到不确定的情况，停下来询问用户**

#### 安全策略

1. **禁止硬编码**：密码、代理凭据等所有敏感信息严禁写入skill文件
2. **交互式输入**：所有敏感信息通过终端交互获取
3. **日志脱敏**：密码→`****`，代理凭据→仅显示用户名

### 第三步：跳转到对应参考文档

根据确认的定位维度，跳转到 `docs/` 目录下对应的详细流程文档并严格执行。

## 核心脚本使用说明

所有脚本输出 JSON 格式结果，便于程序解析。

### 通用二分算法

```text
Phase 1: 测试首尾版本（确认问题区间）
  - 测试 v1（最旧版本），如果异常 → 最早版本已有问题
  - 测试 vn（最新版本），如果正常 → 所有版本均正常

Phase 2: 二分查找首个问题版本
  low = 0, high = n-1
  while low < high:
    mid = (low + high) // 2
    if is_good(v_mid):
      low = mid + 1    # 问题在更晚版本
    else:
      high = mid       # 问题在该版本或更早
  输出 v_low
```

### 特殊情况处理

| 情况 | 处理 |
| ------ | ------ |
| 所有版本均正常 | 附上最新版本的运行结果 |
| 所有版本均异常（v1） | 报告"请检查基线环境" |
| 上传失败 | 提示检查网络和路径权限，重试 |
| 安装失败 | 检查日志，询问用户是否跳过 |
| 服务拉起失败 | 提供vLLM server日志最后100行，询问是否判定为BAD |

## 公共组件：ssh_utils.py

`scripts/ssh_utils.py` 提供了三个核心类：

| 类 | 说明 |
| --- | ------ |
| `SSHClient` | 基础SSH连接与文件上传 |
| `DockerExecutor` | 在Docker容器内执行命令（通过`docker exec`） |
| `CondaExecutor` | 在Conda环境中执行命令（自动激活环境 + 设置CANN环境变量） |

### 用法示例

```python
from ssh_utils import SSHClient, DockerExecutor, CondaExecutor

ssh = SSHClient(host, port, user, password)
ssh.connect()
ssh.exec("ls -la")

# Docker环境
executor = DockerExecutor(ssh, container_name)
rc, stdout, stderr = executor.exec("python --version")
executor.upload(local_file, "/tmp/remote_file")

# Conda环境
executor = CondaExecutor(ssh, env_name, cann_path)
rc, stdout, stderr = executor.exec("python --version")
executor.upload(local_file, "/tmp/remote_file")
```

## 输出报告格式

```text
========================================
  版本二分定位报告
========================================

[定位维度]: CANN / torch_npu / vllm-ascend
[测试摘要]:
总版本数/commit数: n
测试版本数/commit数: m
测试范围: v1 ~ vn
运行环境: Docker / Conda
验证标准: <用户描述的具体标准>

[测试历史]
v1     → 正常  | <关键信息>
v2     → 正常  | <关键信息>
v_mid  → 异常  | <关键信息>   ← 首个异常
...

[定位结果]
首个问题版本/commit: v_mid

[回归范围]
问题引入于: v_{mid-1} 和 v_{mid} 之间

[所有版本均正常时]
结论: 未发现异常版本，所有版本均通过验证。
```
