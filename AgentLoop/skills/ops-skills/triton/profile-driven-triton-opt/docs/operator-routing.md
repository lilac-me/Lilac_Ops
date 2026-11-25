# 算子实现语言路由与优化原则

## 两个独立维度

优化目标由算子特性决定，代码实现语言由真实训练 dispatch 决定，不能把 Vector 等同 Triton、Cube 等同 Ascend C。

| 原始实现 | 单算子优化 | 同语言融合生成 |
| --- | --- | --- |
| 已确认 Triton | triton-op-generator 内 verifier/latency/simulator/MFU skills | triton-op-generator 生成融合 Triton kernel |
| 已确认 Ascend C | tilelang2ascendc-ops-generator；真实源码可优化，复杂算子可走 TileLang 设计再转译 | 同插件生成 Ascend C 融合 kernel |
| CANN 原生（如 aclnn） | 默认目标 Ascend C；拿不到原 kernel 源则生成等价替代，禁止冒称修改原内建实现 | 默认 Ascend C 融合生成 |
| Triton + Ascend C 混合 | 分别优化 | 默认切分同语言子序列；跨语言融合需明确批准、可行性/收益/精度验证 |
| 未知或冲突 | 先回查实际运行 dispatch | 不自动生成或替换 |

CANN 后端没有显式 Triton 时优先考虑 Ascend C，但不能把任何 NPU kernel 都认成 Ascend C 源码：通信、runtime、AI CPU、图编译或其他 DSL 可能存在。Ascend C 是 native CANN 路径的目标语言，不是对其内部源码语言的断言。Core Type、Kernel 名是辅助证据，优先用 trace 的 Host/Device correlation、dispatch、实际源码及 JIT/注册信息。source_map 可填写 backend 和 evidence（实际调用链、源码位置或运行时证据），不得随意填写。通用 kernel 名不能直接证明 Triton；无真实源码映射不进入旧 Triton 优化链。

示例：
```json
{
  "source_map": {
    "causal_conv1d_fwd_kernel": {
      "source": "fsdp/ops/gdn/triton/convolution.py",
      "mode": "B",
      "backend": "triton",
      "evidence": "实际训练调用栈对应的 @triton.jit kernel，需替换为真实证据"
    }
  }
}
```

## 优化目标

- Vector：先识别同 step、同 stream 连续小算子，再校验真实张量依赖、输入输出语义、stride、别名/in-place、外部消费者、通信、forward/backward、autograd saved tensors。对同语言合法子图生成融合 kernel，减少中间 HBM 读写、launch 和同步；以原始完整序列实测 latency 为性能分母，不以其中一个 kernel 或 CPU reference 顶替。
- MatMul/GEMM/GMM：保持原语言，优先 MFU。固定真实 M/N/K、分组和 batch、dtype、布局、FLOPs、峰值口径及 latency。Ascend C 路径使用对应插件的性能分析，不能调用 Triton MFU 工具假装测 Ascend C；独立依据真实 FLOPs / latency / 峰值核算。不得将 Cube pipe ratio 等同 MFU。优化 tiling、数据搬运、流水、核间负载及适用的库模板。
- 其他热点：保持原实现语言，优先 latency，验证实际 shape 和精度。不为追求融合而融合，不默认跨语言转换。
- 新生成算子须有真实原训练 reference、可独立运行的输入及前反向验证；不能因源码不可见伪造 baseline。整网回归仍统一用原 checkpoint、数据及并行配置，验证 step time、loss 和真实吞吐。

## CANNBot 上游入口

- Triton：[cannbot/plugins-official/triton-op-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/triton-op-generator)。
- Ascend C：[cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/quickstart.md](https://gitcode.com/cann/cannbot-skills/blob/master/plugins-community/tilelang2ascendc-ops-generator/quickstart.md)。公开 quickstart 记录简单算子直接开发、复杂算子 TileLang 设计表达再转译 Ascend C；生成语言为 Ascend C，不是 TileLang。
- 两个插件均需实际安装到目标 Agent 环境，社区插件的支持工具、CANN 与 TileLang 版本遵从当前上游 quickstart。Claude Code 可按该 quickstart 安装 tilelang2ascendc-ops-generator@cannbot。不要将 Triton vendor 安装成功等同于 Ascend C 插件安装成功。

## 本版执行边界

分析输出 analysis/operator_routes.json；报告同时展示目标、插件及状态。Ascend C 单算子和同语言融合生成独立 generation_requests/task_*/request.json + prompt.txt，由已安装对应插件的 Agent 按提示执行。默认需人工核定 reference/数据流，不自动执行这些生成任务，不写模型仓，不复用原 Triton 单文件补丁契约。已有已确认 Triton 单算子继续原 tune/finish 闭环。旧配置保留旧的 strong+source_map 路径以兼容历史用例；推荐统一 training 配置，以启用更严格的来源判定。Ascend C 多文件工程、融合图级替换以及前反向注册的独立集成门禁仍需实现，现阶段不得将 generation_requests 当成已集成结果。
