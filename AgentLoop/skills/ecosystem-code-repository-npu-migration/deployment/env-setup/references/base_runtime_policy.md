# 基础 Runtime 策略

加载请求的 CANN 环境后，使用选定的非 base 绝对 Python。

## 必需门禁

```bash
"<selected-python>" -c 'import torch; torch.npu.set_device(0)'
```

退出码为 0 是环境 PASS 标准。它证明 torch 已加载、NPU backend 已注册且可选择逻辑设备 0；不能证明 tensor 计算、forward/backward、分布式通信、训练、推理或长时间稳定性。

分别检查 torch 和 torch_npu 的版本及导入路径，避免成功命令掩盖软件包漂移。安装目标仓库或能力组件后重新运行 gate。

## 可选诊断

- `npu_smoke.py`：在用户请求或 base gate 不足以定位问题时，进行更深入的 tensor/forward/backward 诊断。
- `python -m pip check`：依赖诊断；它不是通用检查，单独运行也绝不是 PASS 要求。
- `hccl_smoke.py`：仅用于分布式范围，通过选定 Python 的 `-m torch.distributed.run` 启动。
- `ray_npu_smoke.py`：仅用于明确请求的 Ray 范围。

直接返回命令输出和范围。不要写入环境产物。
