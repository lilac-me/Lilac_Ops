---
name: vllm-ascend-bisect (doc)
description: vllm-ascend commit 二分定位的详细流程文档。使用 bisect_vllm_ascend.py 核心脚本。
---

# vllm-ascend 版本二分定位

利用vllm-ascend和vllm仓库的git历史，通过二分查找策略，确定**从哪个commit开始**推理服务出现异常。

## 用户输入（特有参数）

| 参数 | 说明 |
| ------ | ------ |
| vllm-ascend安装路径 | 服务器上vllm-ascend源码目录 |
| vllm安装路径 | 服务器上vllm源码目录 |
| 代理配置 | 如需连接外网clone/pip install则配置代理 |

### 先决条件

- torch、torch_npu、CANN等基础依赖已安装完毕
- vllm与vllm-ascend已通过 `pip install -e .` editable模式安装

## 工作流程

### 1. 环境确认与安装准备

1. 检测 Docker/Conda 环境类型
2. 配置代理（如需外网访问）
3. 检查vllm源码目录，如不存在则clone

**安装方式**：先装vllm，再装vllm-ascend。vllm版本参考vllm-ascend commit message中的配套commit id。

### 2. 获取commit列表

在vllm-ascend目录下通过git log获取所有commit，提取每个commit message中的配套vllm commit id。

```text
commit message 格式:
- vLLM main:
https://github.com/vllm-project/vllm/commit/<40位commit_hash>
```

只保留有配套vllm commit信息的commit（跳过CI更新、文档等纯变更）。

### 3. 执行二分

每个commit测试包含：

1. `git checkout -f` 切换vllm-ascend到目标commit
2. `git checkout -f` 切换vllm到配套commit
3. (如需) 重新编译C++扩展
4. 写入用户提供的vLLM启动脚本到服务器临时文件
5. 拉起vLLM服务，记录PID
6. API验证（将payload写入文件，使用 `-d @file` 方式传递）
7. 根据PID或端口（`lsof -ti:<port>` / `fuser <port>/tcp`）精确杀进程

### 4. 输出报告

```json
{
  "results": [
    {"commit": "a1b2c3d4", "vllm_commit": "e5f6g7h8", "status": "normal", ...},
    {"commit": "i9j0k1l2", "vllm_commit": "m3n4o5p6", "status": "abnormal", ...}
  ],
  "commits": [
    {"ascend": "a1b2c3d4", "vllm": "e5f6g7h8", "msg": "feat: add new feature"},
  ]
}
```

## 关键经验

1. **配套commit提取**：vllm-ascend每个commit依赖特定vllm版本，从commit message提取配套vllm commit id
2. **editable模式**：`pip install -e .`后，`git checkout`切换即可生效，不需重新pip install
3. **`git checkout -f`必须**：vllm源码目录常有uncommitted changes，必须强制切换
4. **JSON截断问题**：JSON在嵌套shell中被截断，必须写入文件执行
5. **代理配置**：`docker exec`每次是新shell，需在每条命令中设置代理
6. **NPU卡选择**：使用 `ASCEND_RT_VISIBLE_DEVICES` 指定可用卡；tp=2时需要2个空闲chip
7. **进程管理**：记录PID后精确杀进程，不知道PID时用 `lsof -ti:<port>` 或 `fuser <port>/tcp` 查端口对应PID。绝对禁止 `pkill -f`，防止误杀他人进程
