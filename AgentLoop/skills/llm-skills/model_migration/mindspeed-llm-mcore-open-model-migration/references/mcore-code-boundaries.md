# MCore 模型迁移代码边界

## 目录

1. 修改位置决策
2. 示例目录
3. Model Spec
4. 模型专用模块
5. Checkpoint 转换
6. 共享框架代码
7. 数据与文档

## 1. 修改位置决策

按照以下顺序选择最窄边界：

```text
仅参数/任务变化？
  -> examples/mcore/<slug>/
同系列已有 Spec 能表达且保持旧行为？
  -> 扩展 <series>_spec.py
结构属于新系列或不兼容变体？
  -> 新建系列/变体 Spec
需要目标模型独有 forward、层或状态字典逻辑？
  -> 模型专用模块
行为真正跨模型通用？
  -> 获得共享范围授权后修改框架，并加跨模型回归
```

不得为了省文件，把新模型永久指向另一个系列的 Spec。

## 2. 示例目录

`examples/mcore/<slug>/` 拥有：

- 预训练、微调、评估、推理 launcher；
- HF→MCore、MCore→HF 和在线转换脚本；
- 模型专用数据转换、最小 ST 和操作说明；
- 路径、端口、保存/日志默认值。

示例负责声明模型参数和调用已有实现，不应包含新的大型运行时框架、通用 converter
逻辑或共享 transformer 实现。

复制相近示例前记录模板目录 baseline。迁移完成后要求模板目录零意外变化。

## 3. Model Spec

`mindspeed_llm/tasks/models/spec/<series>_spec.py` 拥有：

- attention、MLP/MoE、norm 和 transformer layer 的模块选择；
- TE/local 组件选择；
- residual/norm 位置；
- sharded state-dict key map；
- 系列级且由 args 驱动的层模式。

Spec 应只读取必要参数。系列决策留在系列 Spec，通用算子行为放在共享组件。

以下变化通常需要新系列或变体 Spec：

- dense 与 MoE 互换；
- 标准 attention 与 MLA/DSA 互换；
- 插入 SSM/混合 attention；
- norm/residual 位置变化；
- MTP 所有权变化；
- 不兼容的 sharded key map。

## 4. 模型专用模块

当现有组件无法表达目标行为时，新建模型专用模块，典型包括：

- 条件层/特殊模块注册；
- 模型独有 forward 或 loss；
- layer-pattern、NoPE、DSA indexer；
- 目标 checkpoint 要求的 state-dict 变换；
- 模型专用参数兼容层。

模块必须使用与 TransformerBlock 和 converter 一致的全局层身份。不要在只持有
`args` 的模块中盲用期望 `TransformerConfig` 的 offset helper。

## 5. Checkpoint 转换

转换边界拥有：

- HF/MCore 键名称和 packing 映射；
- TP/PP/EP/ETP shard 拆分与合并；
- 词表 padding、tied embedding、专家和 MTP 键；
- 模型 slug、架构别名和配置注册；
- 缺失/多余键的显式策略。

转换器不得决定运行时不一致的参数注册。新增模型类型时同时追踪：

1. 外层转换 CLI；
2. 构造 MCore 模型时再次进入的 parser；
3. checkpoint `model_cfg.json`；
4. 从 HF `architectures` 推断 slug 的别名；
5. 两个转换方向。

结构参数必须在正反转换、训练和推理入口一致。

## 6. 共享框架代码

只有满足以下条件才修改共享 transformer、attention、mask、parallel state 或
通用转换框架：

- 问题不是单模型策略；
- 模型开关无法在窄边界正确表达；
- 已搜索全部调用者；
- 默认行为保持兼容；
- 添加至少一个旧模型和目标模型回归；
- 在最终报告中单独说明影响。

生产缺陷应在真实位置修复，不能通过模型独占 harness 绕过。反之，单模型差异
不得伪装成共享修复。

## 7. 数据与文档

- 通用数据格式使用已有 handler；
- 模型专用 chat template/tokenizer 行为放入最小专用配置；
- 只根据 tokenizer 文件或参考代码增加特殊行为；
- 示例说明必须区分从头训练、转换后加载和结构缩减 ST；
- 无运行环境时记录静态验证和用户下一条命令，不虚构已运行结果。

## 边界审查清单

```text
[ ] 目标 slug 与目录归属正确
[ ] 最终 --spec 属于目标系列
[ ] 模板目录无意外修改
[ ] 层身份在 runtime/转换/导出一致
[ ] 正反转换结构参数一致
[ ] 共享修改已搜索调用者并有兼容回归
[ ] 模型专用行为未泄漏到通用工具
[ ] 数据、tokenizer 和文档与真实资产一致
```
