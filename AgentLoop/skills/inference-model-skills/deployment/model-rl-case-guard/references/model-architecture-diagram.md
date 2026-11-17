# 模型结构图交付

在读取模型资产、配置与实际建模实现后生成结构图，再进入服务功能验证。
图用于说明本次开箱的模型，不将论文示意图或模型名称当作配置事实。

## 1. 建立参数依据

在 `model_openbox/<model-slug>/architecture/` 保存 `facts.json`：

- 资产身份：模型 revision、配置路径与 SHA256、建模代码仓/commit；有未提交修改时记录相应文件哈希。
- 结构事实：每项包含 `name`、`value`、来源文件与配置键或源码符号；不能确认的值置空并说明原因。
- 实现差异：配置声明与部署实现的冲突、尚未确认的分支和实际运行限制，不能静默选一个值。

只读解析配置和源码即可，不为画图加载完整权重或执行不可信的 remote code。
沿用已完成的资产审计，避免重复下载模型。源码行号可以辅助定位，文件版本或哈希才是可复核身份。

按模型实际结构核对以下信息；没有的模块不画：

| 部分 | 应核对的信息 |
| --- | --- |
| 输入和主干 | tokenizer/模态入口、词嵌入、hidden size、主干层数、重复层类型及顺序、残差/归一化、输出 head |
| 注意力 | MHA/GQA/MLA/稀疏等真实实现，Q/KV heads、各 head/latent 维度、RoPE、Indexer 计算或复用方式 |
| FFN/MoE | Dense/MoE 层分布、中间维度、routed/shared expert 数和 Top-K；不能把 shared expert 计入 routed Top-K |
| 多模态 | 编码器层数/维度、patch/merge 规则、adapter/projector、归一化类型及接入语言主干的位置 |
| MTP/投机 | 有无结构层、结构层数与执行路径；运行时 speculative token 数单列，不能当作模型层数 |
| 并行与缓存 | 维度默认画逻辑尺寸；分片尺寸标注 TP/EP/PP；明确 KV 逻辑结构与实际 cache 实现的区别 |

模型含逐层异构结构时，核对各类型的层数之和及覆盖范围；不要只展开第一层后把所有层画成同一种类型。
使用减层 checkpoint 验证时，保留正式模型结构依据并另列减层差异，不能把减层层数标成正式模型层数。

## 2. 绘图与导出

1. 依据 `facts.json` 绘制总览：输入 → 编码/嵌入 → 主干 → 输出；用重复框标注层数，侧边展开注意力、MoE 或其他特殊模块。
2. 使用 Graphviz、SVG/HTML 或项目已有绘图工具生成图，保留可编辑源码和实际渲染命令。不要以需要人工重画的文字草稿代替图。
3. 输出 `model-architecture.png` 和可编辑图源（SVG、DOT、HTML 等实际采用的格式）。如采用 DOT/HTML，保留渲染方式，不强制新增另一套工具链。
4. 附上简短阅读说明：数据流方向、维度是否为逻辑尺寸、层号起点、共享/复用关系及没有展开的分支。参考他人图片时注明只借鉴的布局或内容来源。

### 绘图示例：Kimi K3 布局参考

参考 [CalvinXKY/InfraTech 的 Kimi K3 模型结构图](https://github.com/CalvinXKY/InfraTech/blob/main/models/kimi_k_3/kimi_k_3_architecture.jpg)：

![Kimi K3 模型结构图：主干与侧边模块展开的布局示例](https://raw.githubusercontent.com/CalvinXKY/InfraTech/main/models/kimi_k_3/kimi_k_3_architecture.jpg)

Agent 生成当前模型结构图时，可按这个示例组织画面：

- 中央画输入到输出的主干，用箭头明确方向，以重复框和倍数标注各类层。
- 两侧用虚线框展开 Attention、MoE、FFN 等模块，并连回主干中的对应位置。
- 有多模态分支时画出编码器、adapter/merge 与语言主干的接入关系。
- 在关键连线上标注张量维度，模块旁标注 heads、experts、Top-K 等已确认参数。

例如接到“为当前模型生成开箱交付结构图”时，先完成本文的参数依据，再参考上述图的主干、模块展开和维度标注布局，生成当前模型的 PNG 与可编辑图源。
**只借鉴布局与表达方式**；模块、层数、维度、连接和配色可按当前模型调整。不能照搬 Kimi 的结构或参数，也不要求当前模型具有示例中的 KDA、MoE 或视觉模块。
参考图不可访问时记录该限制，仍按已核实的当前模型事实绘图，不虚构参考图内容。

### 成品示例：HY4 开箱交付结构图

[HY4 正式版模型结构图](https://github.com/iwantknow1/hy4-xyh/blob/main/docs/architecture/hy4-full-architecture.png) 是开箱过程中使用 Codex 根据实际配置与实现生成的成品，可作为清晰度、信息组织和模块展开程度的参考：

![HY4 正式版模型结构图：开箱成品效果示例](https://raw.githubusercontent.com/iwantknow1/hy4-xyh/main/docs/architecture/hy4-full-architecture.png)

这个示例展示了中央语言主干、侧边注意力/MoE 展开、视觉与 MTP 分支，以及关键维度和层数标注。
阅读说明、可编辑 SVG 和参数依据见 [HY4 结构图交付目录](https://github.com/iwantknow1/hy4-xyh/tree/main/docs/architecture)。

Agent 应对照成品检查自己的图是否清晰、连线是否完整、参数是否有依据；当前模型的模块、层数、维度和连接须重新核对，不能沿用 HY4 的值。
Kimi 图用于参考布局，HY4 图用于参考已交付成品效果；示例不替代当前模型的 `facts.json` 和源码审计，也不作为必须逐像素复刻的模板。

工具的选择服从已有环境。缺少渲染器时先保存图源和错误，将 PNG 导出记为未完成；不能改扩展名冒充 PNG，或擅自安装系统组件。
允许用户指定输出目录和文件名；上述名称仅是未指定时的默认值。

## 3. 交付检查

- PNG 能实际打开，尺寸与清晰度足够阅读；检查文字裁切、中文字体、箭头方向和连接遗漏。
- 图中层数、head/hidden/expert 等数值逐项对应参数依据；核对异构层计数与输入/输出维度。
- 归一化类型按各分支源码分别确认，不能把语言主干的 RMSNorm 套到视觉分支。
- “配置声明窗口”“已测窗口”“可选路径”分别说明；画出某模块不等于该模块已在 NPU 验证通过。
- 在 `audit.md` 记录结构来源与冲突，在最终 README 链接 PNG、图源和 `facts.json`。没有证据的连接或参数不能写成已确认。

画图不改变功能用例的通过标准，也不能替代服务启动与精度验证。
