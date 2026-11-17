---
name: validate-mindspeed-training-scenarios
description: >-
  This skill should be used when the user asks to design, generate, review,
  run, compare, or diagnose MindSpeed-LLM training scenario matrices for
  fixed-length or packed data, recomputation, TP/SP, PP/num-layer-list,
  EP/ETP, DSA/TND, automatic HF-to-MCore checkpoint conversion and reuse,
  paired feature-toggle launchers, OOM prevention, traceable logs, mean-loss
  alignment, convergence, or distributed runtime failures.
license: MIT
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM 训练场景验证

构建受控训练实验，而不是一组“能启动”的 shell 文件。变更一个声明的功能组合时，
保持架构、输入、checkpoint、优化器和测量合同不变。

## 读取按需参考

- 确认训练后端、入口、在线转换、数据流或测试位置前，阅读
  [MCore 训练流与验证入口](references/mcore-training-flow.md)；
- 生成或审查 launcher、checkpoint cache、runner 和比较逻辑前，阅读
  [场景合同](references/scenario-contracts.md)；
- 启动失败、OOM、重复转换、checkpoint 键异常、Pack/DSA/TND 或 loss 异常时，
  阅读[故障手册](references/failure-playbook.md)。

本 skill 直接适用于 MindSpeed-LLM MCore 路径。先从入口脚本、`--spec` 和示例
目录确认后端，不要为 MCore 场景修改 FSDP2 或 MindSpore 代码。

## 建立实验合同

从请求与仓库解析：

- baseline launcher、模型和精确 Spec；
- world size、设备、TP/PP/CP/EP/ETP、MBS/GBS 和层数；
- 数据、tokenizer、固定长度/Pack、序列长度和随机种子；
- HF 源 checkpoint、MCore cache 和转换复用要求；
- MoE+TP/SP、非均匀 PP、DSA/TND、共享模块和 MTP 等架构约束；
- 终止 iteration、加载 iteration、日志间隔和验收阈值；
- 可用显存及 recompute 测试所需的统一内存拓扑。

目录名称相似不能证明 checkpoint 等价。记录来源、模型身份和切分合同。

## 先设计场景矩阵

- 固定长度与 Pack 是独立功能/收敛场景，不比较逐步 loss；
- recompute on/off 除重计算 flag 外保持全部参数相同；
- 模型要求 TP 同时启用 SP 时，将 TP+SP 声明为一个 toggle bundle；
- PP split 优先使用用户显式输入；未提供时使用运行时均匀切分；
- EP on/off 使用同一专家架构、权重、数据、优化器和 GBS；
- recompute 配对的内存节省并行拓扑必须相同。

为每一对场景写出 intentional-delta 表。启动前拒绝任何未列出的差异。

## 采用公共 launcher

优先创建：

1. 一个包含全部不变量的公共 launcher；
2. 每个场景一个仅导出场景变量的薄 wrapper；
3. 一个串行 runner，记录退出码并在上一进程退出后运行下一场景；
4. 一个可独立调用的比较 runner。

每个场景名称和日志路径必须稳定且唯一。日志头写入场景名、有效拓扑、数据模式、
checkpoint 来源/cache、终止 iteration 和 toggle bundle。

不要在多个 wrapper 中复制数百个公共参数。

## 执行 preflight

分布式初始化前验证：

```text
world_size % (TP * PP * CP) == 0
world_size % (ETP * EP * PP) == 0
num_experts % EP == 0
GBS % (MBS * dense_DP) == 0
SP => TP > 1
PP == 1 => 不使用 num-layer-list
显式 PP 层列表总和 == decoder 层数
未指定层列表 => 使用运行时均匀切分
```

还应检查每个 PP stage 的 microbatch 是否足够，以及 producer/consumer、
full/shared 模块是否保持合法 stage 所有权。执行 shell 语法检查和只打印有效配置
的 dry-run 后再占用加速卡。

## 安全转换并复用权重

用户提供 HF 权重或要求加载预训练权重时，公共 launcher 必须传入仓库等价参数：

```text
--load <hf-source>
--enable-hf2mg-convert
--model-type-hf <model-type>
--mg-save-dir <converted-cache>
--save <training-output>
```

同时确认在线转换 dispatcher 和 converter 都支持模型类型。cache key 至少包含
模型身份、拓扑、层切分、架构 flag 和 converter 版本。

复用前检查：

- `latest_checkpointed_iteration.txt` 可读；
- 引用的 `release` 或 `iter_*` 目录存在；
- 所有预期 rank shard 存在且非空；
- 全部 rank 对 cache 完整性结论一致。

检查失败时重新生成。共享转换身份的场景转换一次后直接加载 MCore cache，
不得每个场景重复转换。

## 保持 Pack 与 DSA/TND 语义

不得为了让 Pack 运行而关闭用户要求的模型特性。追踪 packed metadata 到每个
DSA、稀疏注意力、loss 和 recompute 路径。

验证：

- cumulative query/key sequence lengths 到达所有消费者；
- 算子要求的实际长度是正确设备上的 contiguous `int32`；
- 算子若只接收结束位置，应移除开头累计零；
- TND tensor 与稀疏索引使用算子要求的 flattened-token shape；
- mask 隔离不同文档并保持因果语义；
- 不支持组合在启动前给出具体错误。

每个修复的 shape/dtype/value 合同都添加小型可执行回归。

## 串行执行

默认顺序：

```text
1. 全部场景 preflight/dry-run
2. 固定长度功能场景
3. Pack 功能场景
4. recompute on/off
5. TP(+SP) on/off
6. PP(+layer split) on/off
7. EP on/off
8. 收敛与配对比较
```

共享设备或转换资源时必须串行。单场景失败后可继续，但保留失败码；只有依赖有效
日志的比较才跳过。

## 应用 loss 验收合同

解析日志中真实出现的 iteration，不从 shell 变量推测更新数。每个配对必须满足：

```text
起始 iteration 相同
结束 iteration 相同
两侧 iteration 连续
两侧 loss 有限
abs(mean(candidate) - mean(reference))
  / abs(mean(reference)) * 100 < 1%
两侧分别下降并收敛
```

默认收敛窗口为 5 个 step（运行足够长时）。要求末窗口均值低于首窗口均值，且
完整序列线性斜率小于 0。

逐 step 相对差和最大值默认仅是诊断项；若主合同是全程均值对齐，不因单 step
超过 1% 直接失败。除非用户明确要求，不以 grad norm 为门禁。严格使用 `<`，
恰好 1% 也失败。

固定长度与 Pack 各自只检查有限、连续、正常下降，不相互比较。

## 从最早违反的合同诊断

依次检查：

1. 进程退出及第一个 rank 的主要异常；
2. 输入、数据和 iteration 对齐；
3. checkpoint identity、完整性和局部/全局键所有权；
4. 配对的有效参数差异；
5. 拓扑、DP、microbatch 和 collective；
6. 失败算子的 tensor layout/dtype/value；
7. 收敛和精度指标。

在定位前，不关闭 strict load、DSA，不更换数据或放宽阈值。后续多 rank
NPU/HCCL 错误可能只是首个 Python/算子异常的级联结果。

## 验证变更

至少执行：

- 比较每个 wrapper 导出的变量；
- shell 语法和全部 dry-run/preflight 分支；
- HF 加载/转换参数和在线模型类型支持断言；
- 完整与不完整 cache 检测；
- 日志解析器编译及合成日志边界测试；
- 影响 PP 所有权时检查一个非均匀映射；
- 修改 DSA/TND 时测试精确 packed tensor 合同。

环境可用时运行实际训练；否则明确说明验证仅为静态/合成，并给出下一条命令。

## 输出要求

报告：

- 场景矩阵和每对 intentional delta；
- checkpoint 转换/cache identity 与复用行为；
- PP split 来源和全局/局部层所有权；
- 唯一日志映射及串行执行结果；
- 真实对齐 iteration 范围；
- mean loss 差异及两侧收敛结论；
- 问题、根因、修复、回归和残余风险；
- 已运行命令与尚未验证内容。
