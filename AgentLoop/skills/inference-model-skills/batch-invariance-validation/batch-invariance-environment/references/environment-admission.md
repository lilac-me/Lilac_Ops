# 环境准入执行说明

命令以本 Skill 的根目录（与 `SKILL.md` 同级）为工作目录。先验证用户已确认的 TP 计划，再执行完整预检：

```bash
python scripts/check_batch_invariant_env.py \
  --run-config <运行目录>/run_config.json \
  --report-json <运行目录>/environment/preflight.json
```

保留命令、标准输出、标准错误和退出码，连同报告写入阶段产物。芯片原名和规范分类以报告为准；不得因为构建成功就跳过动态库加载、Torch 注册或设备检查。

OPP 路径优先采用显式 `--opp-path`，其次是非空 `ASCEND_OPP_PATH`，最后采用兼容默认值。已存在的 `vendors/batch_invariant/op_api/lib/libcust_opapi.so` 仍须实际加载并验证必需算子。

安装能力和安装状态是不同结论：`SUPPORTED` 表示匹配源码中存在构建入口，`READY` 表示包、动态库和必需注册检查通过。缺失或损坏时保留首次检测结果，自动构建并复检。A2 对应 `ascend910b`，A3 对应 `ascend910_93`，A5 对应 `ascend950`。预编译 wheel 不包含构建脚本时，必须使用与已安装版本匹配的源码树。

安装能力依据 vLLM Ascend 官方 [Batch Invariance 文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/batch_invariance.md)。执行时记录目标源码 commit 和构建脚本 SHA-256，以目标已安装版本对应源码为准，不直接用持续变化的主线覆盖当前环境。

芯片探测实现位于 `scripts/chip_detection.py`，由本 Skill 的预检脚本维护；其他阶段通过相邻的环境 Skill 复用，不复制实现。
