# Transformers v5 迁移注意点（VeOmni 适配视角）

来源：VeOmni 仓库 `veomni-migrate-transformers-v5` skill 的拆分整理，主流程保留在 `SKILL.md`，本文件只承载迁移检查点。

用途：当接入的模型基于 Transformers v5 代码结构时，在**创建 Patchgen 配置之前**先过一遍本清单，避免 NPU patch 写在过时的接口假设上。具体 API 签名以所用 Transformers 版本的官方文档与源码为准。

## 迁移检查点清单

### 1. 配置与注册

- [ ] `config.json` 字段名是否随 v5 重命名（如归一化层类型、attention 实现开关）
- [ ] config 类继承路径是否变化；auto_map / remote code 注册方式是否调整
- [ ] 模型注册表中该模型的入口类名是否更新

### 2. 模型结构与 attention

- [ ] v5 若统一了 attention 接口，NPU patch 中的 attention 后端替换点需要重新定位
- [ ] RoPE / 位置编码实现是否重构（影响 NPU patch 中的 rope 替换）
- [ ] 归一化（RMSNorm/LayerNorm）调用位置是否有 fp32 上溢处理约定

### 3. Tokenizer / Processor

- [ ] tokenizer 与 processor 的加载入口是否变化（影响 VLM/Omni 数据侧接入）
- [ ] 特殊 token 与 padding 方向约定是否变化（影响精度对齐的输入固化）

### 4. 权重与分片

- [ ] v5 权重命名是否调整（影响权重转换映射表）
- [ ] 分片索引命名规则是否变化
- [ ] tied embedding / lm_head 权重绑定行为是否一致

### 5. 对 VeOmni 适配产物的连锁影响

| 变化点 | 影响的适配产物 |
| -------- | --------------- |
| config / 注册变化 | registry 改动、auto_map 条目 |
| attention / rope 重构 | NPU patchgen 配置 |
| tokenizer 变化 | 数据预处理、toy config |
| 权重命名变化 | 权重转换脚本与映射表 |
| padding 约定变化 | golden 固化输入、MFU 有效 token 统计 |

## 使用原则

- 本清单只定位"哪里要改"，不代替官方迁移指南；执行时以 VeOmni 所依赖的 Transformers 版本为准。
- 每确认一项，把结论写进 `adaptation_plan.md` 与 `patch_manifest.json`（如涉及权重转换则更新映射表），保持产物可追溯。
- 迁移引出的差异最终要靠精度对齐闭环验证（见 `veomni-precision-alignment`），不要凭"应该兼容"跳过回归。
