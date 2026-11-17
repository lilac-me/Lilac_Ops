# Checkpoint 与 Launcher 模式

## Checkpoint 与 resume 契约

训练 checkpoint 可能包括 model shard、optimizer state、scheduler、scaler、RNG、dataloader/sampler 位置、global step 和 topology metadata。推理导出是不同的契约。

识别信号包括 shard 缺失、state-dict key 或 shape 不匹配、optimizer tensor 位于错误设备、不完整的完成标记、按 rank 命名的文件，以及 restore 期间 topology 变化。

检查仓库声称保存的内容、哪些 rank 写入每一部分、何时发布完成标记、如何处理 `map_location` 或设备转换，以及官方是否支持 topology 变化。保留现有 CUDA 和 CPU 加载路径。

验证文件完整性、相同 topology 的 save/load 以及真实的加载后 NPU forward。声称 exact resume 时，与不间断运行比较下一个 batch/step，并包含 optimizer、RNG 和 dataloader state。仅加载模型权重不能证明精确训练 resume。

## Shell wrapper 与真实命令

可执行 shell 文件不自动等于完整命令。检查必需的位置参数、加载的环境、默认配置、导出变量、工作目录和最终执行的 Python 命令。

当仓库原生 wrapper 编码了必需设置时保留它。通过支持的参数或配置绑定选定的绝对 Python 和本地资产路径，然后验证 wrapper 确实转发了它们。shell 进程在到达 Python 前退出是 launcher 失败，不是 NPU runtime 失败。

## Hydra、YAML 与分层配置

识别信号包括 launcher 要求 config name、配置搜索路径缺失、override 被解析为位置参数、插值失败，或仓库默认值静默选择 CUDA。

跟踪 shell 参数经 Hydra/argparse 到最终 runtime 值的过程。区分配置组选择和 key override，保留仓库的组合顺序。不要通过编辑生成输出或删除必需 key 绕过配置校验。

先验证配置组合或执行 dry parser check，然后在真实启动中打印或以其他方式检查解析出的 device/model/data 值。parser 成功不能证明模型、数据、NPU backend 或训练路径已经运行。

## 命令证据

对每个修复的 launcher，保留绝对解释器、cwd、精确命令、退出码、有界 stdout/stderr，以及已到达预期 Python entry 和解析参数的证据。只有重跑原始语义目标或明确仍被阻塞时，环境或命令修复才算完成。
