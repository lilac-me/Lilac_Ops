# 官方文档快照

## 快照元数据

- 检查时间：`2026-01-15T10:30:00Z`
- runtime 目标：`昇腾 NPU / torch_npu`
- CANN：`9.0.0`
- torch：`2.10.0`
- torch_npu：`2.10.0`
- 网络状态：`PARTIAL`
- cache 状态：`validated-cache-used`
- 配置 hash：`replace-with-current-config-hash`
- 仓库 commit：`replace-with-current-commit`

## 来源

| URL | 访问状态 | 观察到的版本或标题 | 获取时间 | 用途 |
| --- | --- | --- | --- | --- |
| `https://example.invalid/torch-npu/install` | `OK_HTTP_200` | `torch_npu 安装指南` | `2026-01-15T10:31:00Z` | 选定 Python/CANN 设置 |
| `https://example.invalid/torch-npu/compatibility` | `CACHE_VALIDATED` | `兼容矩阵` | `2026-01-15T10:32:00Z` | torch/torch_npu 版本组合 |

## 使用的迁移规则

| 规则 | 来源 URL | 应用对象 | 补丁或决定 |
| --- | --- | --- | --- |
| 导入 `torch_npu` 前先 `source` CANN 环境 | `https://example.invalid/torch-npu/install` | 环境 bootstrap | 选定的非 base Python 和 CANN 激活 |
| 增加 NPU 选择时保留 CPU/GPU 分支 | `https://example.invalid/torch-npu/compatibility` | `src/device.py` | preflight 审查接受窄分支 |

## 访问失败或 fallback

| 来源 | 失败 | 回退路径 | 风险 |
| --- | --- | --- | --- |
| `https://example.invalid/torch-npu/releases` | 网络超时 | 未作出当前发布版本结论 | 精确 wheel 可用性仍未验证 |

## 门禁决定

- 可以继续 preflight：`yes`
- 原因：两个影响决策的来源要么是在线来源，要么是经过校验的 cache 条目；未引入不受支持的硬件常量。
- 该快照是兼容性依据，不是本地 NPU 执行证据。
