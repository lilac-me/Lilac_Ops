# 昇腾 NPU 支持

## 文档目标

优先使用已有的文档位置和命名约定。如果没有合适位置，可提议
`docs/npu.md`、`docs/ascend-npu.md`、`docs/source/ascend-npu.md` 或
`NPU-support.md`；在已有文档树中创建新的顶层页面前应先询问用户。

## 支持的环境

| 组件 | 版本 | 证据 |
| --- | --- | --- |
| CANN | | |
| PyTorch | | |
| torch_npu | | |
| Python | | |
| Hardware | | |

## 功能支持

| 功能 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| | | | |

状态值：`Supported and verified`、`Partial support`、`Blocked`、`Skipped by scope`、`Not applicable`、`Not verified`。

## 必需章节

当页面是通用 NPU 支持页面时，只覆盖已有对应证据的章节：支持的
CANN/torch/torch_npu/Python/硬件版本；不覆盖系统 CANN 或 Python 的安装说明；
功能支持；验证命令及其证明范围；跳过或排除的范围；CPU fallback；核心缺口
闭环；共享代码审查和 backend 保留；Triton/自定义 kernel 状态；分布式/HCCL
状态；仓库原生多卡状态；已知限制；以及最小 NPU smoke。

对于不在范围内的证据可以省略章节，但应写明 `Not verified` 或明确标记为阻塞，
不要推断为正向状态。

## 验证

| 命令 | 结果 | 证明内容 |
| --- | --- | --- |
| | | |

## 核心缺口闭环

| 缺口 | 决策 | 证据 | 剩余风险 |
| --- | --- | --- | --- |
| | | | |

## 仓库原生多卡训练

| 机制 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| | | | |

## 已知限制

- 量化：
- TensorFlow/TFLite：
- CPU fallback：
- Triton/自定义 kernel：
- 分布式：
- 多卡训练：
- 长时间稳定性：

## 最小 NPU Smoke

```bash

```

## 证据与来源完整性

- 不要编造对应的 `.npu-agent` 证据中没有的结论。
- 关键 operator 运行在 CPU fallback 上属于部分支持，不是已验证的 NPU 支持；
  应写出准确的 fallback 路径。
- 单卡、通用 HCCL 或自动生成的 DDP smoke 不能证明仓库原生多卡训练。
- 保持命令可复现，并遵循仓库写作风格。
- 除非证据明确需要复现，否则不要复制私有机器路径。
- 如果修改了共享 GPU/NPU 代码，应说明 GPU 路径是已测试还是仅完成代码审查。
