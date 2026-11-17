# Patchgen 主流程与 generated 文件管理

## Patchgen 主流程

1. **定位 patch 目录**：VeOmni 中 patch 配置按模型族组织（如 `patches/<model_family>/`），每个模型区分 GPU 与 NPU 配置文件。
2. **新建 NPU patch 配置**：
   - 复制最接近参考模型的 GPU 配置作为起点
   - 常见 NPU 差异点：attention 后端选择（torch_npu npu_fusion_attention）、rope 实现、dtype 默认值、flash attention 开关
3. **声明生成目标**：patch 配置中列出将要生成的 modeling 文件路径（generated 文件）。
4. **执行 Patchgen**：按 VeOmni 文档执行 patchgen 命令，生成 modeling 文件到模型目录。
5. **校验生成结果**：对比生成文件清单与 patch_manifest.json 声明一致。

## generated 文件管理（硬约束）

- generated 文件**只能由 Patchgen 生成**，任何手工修改都被视为违规。
- 需要修改行为时：修改 patch 配置（或上游 modeling 模板）→ 重新运行 Patchgen。
- 测试与评审时通过 `patch_manifest.json` 中的文件清单 + 内容哈希核对：重新运行 Patchgen 后哈希不变才算通过。
- 违规场景（测试设计第 7 节"generated 文件被修改"）必须被检测并报告，不得静默通过。

## source revision 记录

patch_manifest.json 必须记录：

- VeOmni 仓库 commit hash（`git rev-parse HEAD`）
- Patchgen 配置文件路径
- 生成文件相对路径列表与内容哈希

## 权重转换要点

- 优先直接加载；仅当命名/分片不兼容时编写转换脚本
- 映射表（原名称 → 目标名称）单独成文件，脚本读表转换，避免硬编码
- 转换前后各自打印 key 数量与总参数量，确认无损
- VLM/Omni 模型注意视觉/音频 tower 权重前缀
