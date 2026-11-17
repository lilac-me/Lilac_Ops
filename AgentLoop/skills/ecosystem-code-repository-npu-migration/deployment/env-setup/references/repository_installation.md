# 仓库安装

遵循目标 checkout 维护的工作流进行安装，同时保留选定的 NPU runtime。

## 发现安装契约

按以下顺序阅读适用文件：

1. `README*`、`INSTALL*`、`CONTRIBUTING*` 和安装文档。
2. `pyproject.toml`、`setup.py`、`setup.cfg` 和包管理器配置。
3. requirements、依赖组/extras、conda YAML 和引导脚本。
4. `uv.lock`、`poetry.lock` 或其他锁文件。
5. 展示受维护安装路径的 CI 或容器配方。

根据仓库实际文档选择安装方式；文件存在本身不是安装契约。

## 选择并保护软件栈

- pip 和由解释器驱动的安装器都使用选定的绝对 Python。
- 对于 uv 或 poetry，要证明软件包进入选定的 conda 环境，而不是隐式的嵌套环境。
- 只安装请求 workload 所需的 extras。排除 CUDA-only 可选组，并说明这种偏离。
- 保持明确的 CANN/torch/torch_npu/Python 约束为最高优先级。
- `"<selected-python>" -m pip install --no-deps -e .` 是有条件的安全选项，不是通用默认值。只有 editable 安装受支持且普通解析会替换已验证 NPU 软件栈时才使用它。

安装前后都检查 torch 和 torch_npu 的版本/导入路径。安装后验证目标导入来自预期 checkout，并重跑 mandatory base gate。修复未经批准的漂移；要求无法共存时返回 `BLOCKED_TARGET_INSTALL`。
