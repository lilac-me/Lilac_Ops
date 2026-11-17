# 迁移进度

<!-- 仅为说明性示例。请将每个值替换为当前任务事实。 -->

## 目标

让仓库的标准 SFT 和 DPO 训练路径在昇腾 NPU 上运行。

## 已完成

- 环境 gate：PASS
  - 选定 Python：`/absolute/path/to/python`
  - 命令：`"/absolute/path/to/python" -c 'import torch; torch.npu.set_device(0)'`
  - 证据：`logs/env_gate.log`
- SFT smoke：UNVERIFIED
  - 命令：`<repository-native SFT command>`
  - 证据：`<日志路径或当前任务输出>`

## 当前

- DPO 代表项：调查第一个 NPU runtime 失败。
  - blocker：`<简短的最深层异常>`
  - 下一步：`<一个具体的下一步行动>`

## 剩余工作

- 修复后运行 DPO。
- 运行剩余的可行 capability check。
- 协调最终兼容性报告。

## 重要决定

- 选定 Python：`/absolute/path/to/python`
- 外部知识根目录：`<绝对路径或 none>`
- 支持结论需要当前源码、精确命令、日志和真实 NPU 行为。
- 不要为了强行通过而修改测试。

## 最后更新

- `<UTC timestamp>`
