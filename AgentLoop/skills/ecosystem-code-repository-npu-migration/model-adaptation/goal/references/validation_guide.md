# NPU 迁移验证指南

运行能够证伪结论的最窄当前检查。直接命令就是证据；不需要 workflow 状态产物。

## 环境

使用选定的绝对 Python：

```bash
"<selected-python>" -c 'import torch; torch.npu.set_device(0)'
```

退出码为 0 只证明 torch NPU backend 已注册且可以选择逻辑设备 0。单独检查软件包版本和导入路径。安装目标仓库或其他加速器软件包后，重新运行相同命令。

## 可选语义校验器

生成本地资产清单后进行校验：

```bash
python <plugin-root>/skills/local-assets/scripts/local_asset_inventory.py validate \
  --repo <repo> --inventory <repo>/.npu-agent/local_asset_inventory.json --json
```

生成两个 feature-model 文件后进行校验：

```bash
python <plugin-root>/skills/feature-model/scripts/validate_feature_model_artifacts.py \
  --repo <repo> --json
```

该检查证明 v2 功能地图和代表项计划在内部完整且交叉引用正确。它不证明 runtime 支持，也不证明所选 equivalence group 在语义上足够。

## 证据阶梯

1. 静态/源码：语法、lint、生成文件一致性。
2. Import/config：选定软件包来源和 launcher 解析。
3. Runtime 组件：真实 NPU tensor/operator/forward/backward/collective。
4. 仓库原生 smoke：真实 entrypoint、数据流、loss/generation/update 和请求的输出。
5. 代表性覆盖：不同 task family、topology、backend 和高风险语义已执行或明确限定。

只能声称实际达到的最高级别。CPU fallback、`--help`、配置解析和初始化日志都不是端到端 NPU 证明。
