---
name: torch-npu-bisect (doc)
description: torch_npu 版本二分定位的详细流程文档。使用 bisect_torch_npu.py 核心脚本。
---

# torch_npu 版本二分定位

利用本地预先下载好的 torch_npu `.whl` 安装包集合，通过二分查找策略，确定**从哪个版本开始**推理服务出现异常。

## 用户输入（特有参数）

| 参数 | 说明 |
| ------ | ------ |
| 本地安装包目录 | 存放torch_npu `.whl`文件的目录 |
| CANN安装路径 | 用于激活CANN环境变量的路径 |

### 安装包目录要求

```text
<本地安装包目录>/
├── 20260630/            ← 文件夹名为日期数字，数字越大版本越新
│   └── torch_npu-2.10.0.post3.dev20260630-cp311-cp311-manylinux_2_28_aarch64.whl
├── 20260701/
└── 20260703/
```

## 工作流程

### 1. 环境确认

1. 检测 Docker/Conda 环境类型
2. 记录当前torch_npu版本：`pip list | grep torch`（后续恢复用）

### 2. 扫描版本列表

使用 `bisect_torch_npu.py` 扫描本地目录，按文件夹名数字排序。

### 3. 执行二分

每个版本测试包含：

1. 上传 `.whl` 文件到服务器
2. 卸载旧版本：`pip uninstall torch-npu -y`
3. 安装新版本：`pip install <whl>`，验证版本号一致
4. 激活CANN环境变量
5. 写入用户提供的vLLM启动脚本到服务器临时文件
6. 拉起vLLM服务，记录PID
7. 执行API验证
8. 根据PID或端口（`lsof -ti:<port>` / `fuser <port>/tcp`）精确杀进程
9. 记录结果

### 4. 输出报告

```json
{
  "results": [
    {"version": "20260630", "status": "normal", ...},
    {"version": "20260701", "status": "abnormal", ...}
  ],
  "versions": ["20260630", "20260701", "20260703"]
}
```

## 关键经验

1. **pip版本验证**：安装后务必通过 `pip list | grep torch` 确认版本日期与whl文件名一致
2. **强制重装**：如果已安装同版本号但内容不同的whl，需用 `--force-reinstall` 覆盖
3. **JSON引号问题**：包含JSON的参数写入脚本文件后执行
