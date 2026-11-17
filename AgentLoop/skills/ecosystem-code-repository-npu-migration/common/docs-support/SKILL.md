---
name: docs-support
description: This skill should be used when the user asks to create or update repository-facing Ascend NPU support documentation from accepted current migration evidence. Use when users need installation, launch, validation, limitation or troubleshooting sections added in the target repository's existing documentation style; do not use to implement or validate NPU support.
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

# NPU 文档支持

## 角色

在不增强技术结论的前提下，将已接受的当前证据转换为面向仓库用户的文档。

## 使用场景

当用户在实现和验证后要求补充目标仓库文档时使用。

## 工作流程

1. 检查仓库的文档结构、写作风格和已有安装方式。
2. 阅读调用方提供的当前 diff、精确命令/日志、导入来源和明确限制。
3. 使用 [npu_support_doc_template.md](references/npu_support_doc_template.md) 作为章节清单，并适配仓库约定。
4. 只纳入可复现的环境、启动和验证命令；明确标记不支持、阻塞或未验证的路径。
5. 检查文档 diff，确认没有机器专属路径、密钥、过期版本或真实执行不支持的结论。

## 边界

不要修改迁移源码、安装软件包、执行测试或编造支持证据。文档不能把基础环境门禁或导入检查提升为训练支持。

## 完成条件

返回文档改动、对应证据来源，以及因缺少证明而有意省略的章节。

## 资源

- [npu_support_doc_template.md](references/npu_support_doc_template.md)：可适配的内容清单。
- `examples/npu_support.md`：仅用于参考措辞的脱敏示例。
