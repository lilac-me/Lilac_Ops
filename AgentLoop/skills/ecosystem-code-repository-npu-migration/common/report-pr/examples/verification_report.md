# 验证报告

## 范围

- 最终状态：`PARTIAL`
- 证据：当前命令、退出码、日志、导入路径和 diff

| 目标 | 结果 | 证据 |
| --- | --- | --- |
| 基础环境 | 通过 | 选定 Python 的设备选择命令 |
| `examples/train_one_step.py` | 阻塞 | 当前 stderr 显示所需数据集不可用 |

不要把 config/import/static 检查升级为仓库 NPU 支持结论。
