# 可选能力安装

只有请求的 workload 需要时，才安装可选软件包。

1. 记录选定的 Python 以及 torch 和 torch_npu 的版本/导入路径。
2. 检查当前官方兼容性和该能力维护的安装说明。
3. 使用最小必要 extras 安装到选定环境。
4. 重新检查版本/导入路径，并重跑 `"<selected-python>" -c 'import torch; torch.npu.set_device(0)'`。
5. 继续之前先修复受保护软件栈的漂移；无关的基础 workload 仍健康时，只阻塞该能力。

不要创建能力产物或静默安装 CUDA-only 替代方案。向调用方返回精确命令和当前 smoke 证据。
