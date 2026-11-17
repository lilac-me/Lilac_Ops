---
name: cann-bisect (doc)
description: CANN 版本二分定位的详细流程文档。使用 bisect_cann.py 核心脚本。
---

# CANN 版本二分定位

利用本地预先下载好的 CANN 安装包集合，通过二分查找策略，确定**从哪个版本开始** vLLM 推理服务出现异常。

## 用户输入（特有参数）

| 参数 | 说明 |
| ------ | ------ |
| 本地安装包目录 | 存放CANN安装包的目录 |
| 远程安装路径 | 安装CANN的目标路径（如 `/home/cann_agent/`） |

### 安装包目录要求

```text
<本地安装包目录>/
├── B030/               ← 文件夹名包含数字，数字越大版本越新
│   ├── Ascend-cann-toolkit_<version>_linux-aarch64.run
│   ├── Ascend-cann-nnal_<version>_linux-aarch64.run
│   └── Ascend-cann-A3-ops_<version>_linux-aarch64.run
├── B035/
└── B039/
```

版本排序规则：**以文件夹名称中的数字为准**（B030 < B035 < B039）。

## 工作流程

### 1. 环境确认

1. 检测 Docker/Conda 环境类型
2. 父目录权限检查：`chown root:root <远程安装路径>`

### 2. 扫描版本列表

使用 `bisect_cann.py` 的 `scan_package_dirs()` 扫描本地目录，按文件夹名数字排序。向用户展示确认。

### 3. 执行二分

使用 `bisect_cann.py` 的 `run_bisect()` 函数。每个版本测试包含：

1. 上传三个 `.run` 文件到服务器
2. 安装 toolkit + nnal + A3-ops
3. 激活CANN环境变量
4. 写入用户提供的vLLM启动脚本到服务器临时文件
5. 拉起vLLM服务，记录PID
6. 执行API验证
7. 根据PID或端口（`lsof -ti:<port>` / `fuser <port>/tcp`）精确杀进程
8. 记录结果

### 4. 输出报告

```json
{
  "results": [
    {"version": "B030", "status": "normal", ...},
    {"version": "B035", "status": "abnormal", ...}
  ],
  "versions": ["B030", "B035", "B039"]
}
```

## 关键经验

1. **父目录权限**：CANN安装器要求所有父目录owner为root
2. **JSON引号问题**：`--additional_config`等参数包含嵌套JSON时，写入脚本文件后执行
