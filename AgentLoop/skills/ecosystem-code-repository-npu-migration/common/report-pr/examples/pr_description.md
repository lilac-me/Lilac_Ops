# NPU 迁移 PR

## 摘要

增加有边界的 NPU 设备选择适配，并保留现有 CPU/GPU 行为。

## 验证

- `"<selected-python>" -c 'import torch; torch.npu.set_device(0)'`：NPU 基础环境 gate 通过。
- `python -m pytest tests/test_device.py`：使用报告记录的精确解释器通过。
- `examples/train_one_step.py`：因等待所需本地数据集而阻塞。

## 限制

这不代表多卡、checkpoint/resume 或完整仓库支持。
