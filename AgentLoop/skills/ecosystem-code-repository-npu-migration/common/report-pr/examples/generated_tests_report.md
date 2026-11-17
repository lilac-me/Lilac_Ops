# 生成测试报告

| 测试 | 位置 | 原因 | 官方要求 | 命令 | 需要真实 NPU | CI 候选 |
| --- | --- | --- | --- | --- | --- | --- |
| `test_selected_device_is_npu` | `tests/generated/test_selected_device.py` | 覆盖两个范围内训练 entry 使用的共享设备 helper | `torch_npu` 设备选择必须明确，并保留 CPU fallback | `selected-python -m pytest -q tests/generated/test_selected_device.py` | yes | review |
| `test_cpu_reference_matches_npu_shape` | `tests/generated/test_cpu_reference.py` | 检查有界 operator 替代方案的 shape/dtype 对齐 | 本地迁移策略要求 CPU/参考控制 | `selected-python -m pytest -q tests/generated/test_cpu_reference.py` | no | yes |

## 备注

- 生成测试是补充证据，不能替代仓库现有示例或 unit test。
- 第一个测试需要真实 NPU；NPU 不可见时必须保持 `NOT_RUN`。第二个测试是参考检查，单独不能建立 NPU 支持结论。
- 来源：`.npu-agent/feature_model.json`、`.npu-agent/coverage_plan.json` 和当前源码审查。
