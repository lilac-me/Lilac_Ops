# Flux DiT SFT 训练前置改动

当用户明确迁移 Flux SFT，且 VeOmni 工作副本包含下列路径时，按本清单实施并验证。这里的文件路径是目标 VeOmni 工作副本中的路径，不在 MindSpeed-Agent 仓库中直接创建同名文件。

## 配置与模型声明

| 目标文件 | 必需改动 | 验证点 |
| --- | --- | --- |
| `configs/dit/flux_sft.yaml` | 增加 `dp_shard_size: 16`、`global_batch_size: 16`、`vae_subfolder: vae` | 启动前校验实际 world size、数据并行划分和显存能支持 16；确认权重目录存在 `vae/`。不可在不满足拓扑约束时强行使用 16。 |
| `configs/model_configs/flux/flux.json` | 将 `architectures` 从字符串 `"FluxModel"` 改为数组 `["FluxModel"]` | 通过模型配置加载和 Auto class/注册测试，确认消费者接受数组类型。 |

保持既有 YAML 缩进和键所在配置节；不要因为本清单而移动或重命名无关配置。训练命令、最终生效配置和 Git diff 必须写入过程记录。

## 本地权重加载

在 `veomni/models/module_utils.py` 的 `_load_state_dict` 起始处增加本地文件存在性分支：

1. 先区分本地 checkpoint 文件与远端 Hub 标识；本地文件存在时直接走本地加载路径。
2. 仅当本地文件不存在且输入确实是远端标识时，才调用 `cached_file`。
3. 本地路径既不存在又不允许远端解析时，抛出包含原始路径的明确错误；不得吞掉错误或静默返回空 state dict。
4. 保持 map location、dtype、分片和安全加载的既有语义。

必须覆盖“本地文件存在”“本地文件不存在”“远端标识”三种测试，确保不会再把本地路径误送入 `cached_file`。

## 批量前向包装

在 `veomni/models/transformers/flux/modeling_flux.py`：

1. 将当前单样本 `forward` 实现移动或重命名为 `_forward_single`，其原有单样本行为、签名语义、梯度和返回类型不得改变。
2. 新增 `forward` wrapper，按 VeOmni 中 `wan_t2v` 的既有实现选择和遍历 list 输入；不要自行发明不同的批处理协议。
3. list 分支必须严格校验所有并行输入的长度；每个元素调用 `_forward_single`，输出顺序与输入一致。非 list 输入必须直接委派给 `_forward_single`。
4. 明确 kwargs、condition、mask、device/dtype 和返回对象在单样本与 list 分支中的传播方式，避免 wrapper 递归调用 `forward`。

如果该文件由 Patchgen 生成，不得直接保留手改：应将等价改动落实到 Patchgen 配置或上游模板并重新生成，然后更新 `patch_manifest.json` 的哈希。

## 图像条件尺寸对齐

在 `veomni/models/transformers/flux/flux_condition/modeling_flux_condition.py` 的 `encode_image`：

1. 在 VAE 编码前将图像高宽调整到 32 的倍数，并保持现有图像类型、device、dtype 与归一化顺序。
2. 采用项目既有的插值方式和长宽策略；当代码中可获得 VAE 下采样倍数时，断言 `32 == 2 * vae_scale_factor`，而不是硬编码一个与 VAE 不一致的常量。
3. 编码后校验 latent 的高、宽均为偶数；不满足时给出输入尺寸、resize 后尺寸和 VAE scale 的可诊断错误。
4. 对非 32 倍数图片以及本来对齐图片分别测试，确认不会改变 batch 对齐或条件图顺序。

## 最小验证与交付证据

在用户授权的环境中，至少执行：配置加载、模型 JSON 加载、本地权重加载三分支、单样本与 list `forward`、非 32 倍数图像的 `encode_image`、以及一至两个训练 step。随后执行既定精度和性能流程。

最终 `delivery_coverage_report.md` 至少应列出 `config`、`model`、`dataloader`/图像条件预处理、权重加载和训练验证；`migration_process_record.md` 必须记录每个文件的 diff、命令、日志和未执行原因。
