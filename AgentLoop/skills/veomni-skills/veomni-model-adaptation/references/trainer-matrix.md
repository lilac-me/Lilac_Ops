# Trainer 接入差异矩阵

| 维度 | Text | VLM | DiT | Omni | RL |
| ------ | ------ | ----- | ----- | ------ | ----- |
| 数据输入 | text token | 图文对 + processor | 视频/图像 + latent | 多流（文本/音频/图像） | rollout 轨迹 + reward |
| Collator/预处理 | tokenize + padding | 图像预处理 + metadata | 视频帧抽取 + VAE latent | 多模态字段对齐 | 轨迹采样 + advantage 计算 |
| Trainer 入口 | Text Trainer | VLM Trainer | DiT Trainer | Omni Trainer | RL Trainer |
| 特殊 metadata | 无 | 图像 grid / position id | 视频时长/分辨率 | 各模态采样率 | episode/step 边界 |
| Checkpoint 注意 | 常规 DCP | 常规 DCP | 常规 DCP | 常规 DCP | actor/critic 分开保存 |
| toy config 要点 | 小 vocab + 短序列 | 小分辨率 + 1 图 | 低分辨率短视频 | 各模态各 1 条样本 | 小 rollout batch |

## 接入检查清单

1. Trainer 配置 dataclass 中模型类型字段已支持新模型
2. collator 产出的字段名与模型 forward 签名一致
3. 多模态 metadata 在分布式切分（SP/数据并行）后仍正确对齐
4. toy config 能在单卡完成 1-2 step（功能验证入口）
5. RL 场景仅接入训练侧；完整训推编排交 RL Agent（能力边界）
