---
name: local-assets
description: This skill should be used when the user asks to discover, validate and query bounded local model or dataset assets for an NPU migration without downloading new content. Use when repository examples need existing Hugging Face, ModelScope or filesystem assets and the agent must produce a reproducible inventory or select a path from user-authorized roots.
version: 0.16.0
domain: ecosystem-code-repository-npu-migration
function: common
tags:
  - ascend
  - npu
  - migration
  - pytorch
author: addsubmuldiv
---

# 本地资产清单

## 角色

在用户授权的根目录下构建或查询可用本地模型和数据集的确定性清单。发现资产不等于目标命令实际使用了该资产。

## 使用场景

当迁移命令需要已有模型或数据集路径、元数据或完整性证据时使用。资产获取和下载不属于此 skill。

## 工作流程

1. 明确候选根目录和大小/范围限制。默认不要扫描 home、系统目录或无关存储。
2. 当存在候选根目录或计划中的示例需要模型/数据集时，运行 `scripts/local_asset_inventory.py build`，并将唯一的持久化清单写入 `.npu-agent/local_asset_inventory.json`。
3. 使用前先校验清单。读取 `models`、`datasets`、绝对 `path`、完整性 `evidence`、扫描限制、`truncated` 和 `errors`；不要把不完整或被截断的扫描当作资产不存在的证据。
4. 针对仓库中的每个模型或数据集引用运行脚本的 `query` 命令。仅当存在一个完整匹配时返回 `FOUND`，多个匹配时返回 `AMBIGUOUS`，没有匹配时返回 `MISSING`。
5. 返回精确路径、证据和限制。调用方必须在真实示例命令中绑定选定路径，并验证启动器确实使用了它。

## 边界

不要下载、修复、转换或删除资产；不要安装环境、执行训练或作出支持结论。不要另建 entry-to-asset 绑定产物、workflow 状态或 Python 集合作为第二套事实来源，也不要在未重新校验的情况下使用过期清单进行选择。

## 完成条件

返回经过校验的有界清单，或返回包含已搜索根目录和排除原因的明确无匹配结果。

## 资源

- `scripts/local_asset_inventory.py`：构建、校验和查询清单。
- [local_asset_contract.md](references/local_asset_contract.md)：完整性和字段规则。
- `examples/`：仅提供脱敏结构。
