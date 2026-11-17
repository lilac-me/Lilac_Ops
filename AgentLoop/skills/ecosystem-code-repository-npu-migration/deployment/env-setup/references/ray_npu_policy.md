# Ray NPU 策略

仅当请求的 workload 实际使用 Ray 时使用。

- 直接使用当前任务选定的绝对 Python。
- 验证 Ray 由该解释器导入，并报告预期的 NPU 资源。
- 使用同一解释器运行 `scripts/ray_npu_smoke.py`。
- 将成功发现资源视为 Ray placement 证据，而不是仓库训练证明。
- 安装或修改 Ray 相关软件包后，重跑 mandatory torch NPU base gate。

返回直接命令证据；不要写入能力产物。
