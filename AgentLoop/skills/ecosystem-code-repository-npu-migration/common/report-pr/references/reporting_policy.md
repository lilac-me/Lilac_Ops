# 报告策略

## 证据优先级

1. 当前仓库原生命令、解释器、退出码和日志。
2. 当前组件/runtime 命令和观察到的行为。
3. 当前源码 diff 和导入来源。
4. 匹配的可选语义输出，例如 feature model 或资产清单。
5. 静态检查和官方兼容性研究。

不要把较低级别的证据提升为较高级别的支持结论。

## 必须区分的内容

- 基础环境门禁与 tensor/operator/training 行为；
- 静态/import/config 与 runtime 执行；
- CPU fallback 与真实 NPU 执行；
- 单设备与分布式；
- 单个 task family/model topology 与仓库级覆盖；
- 外部 blocker 与代码失败；
- 当前证据与过期或不匹配的结果；
- 任务处置与迁移支持结果。

每一条 PASS 都要说明运行了什么以及范围是什么。每条被阻塞或未验证的请求路径都必须可见。

`COMPLETE` 表示在当前约束下没有剩余的安全且可执行工作，不表示 `SUPPORTED`；当请求功能缺少必要的真实 NPU 证据时，应报告 `PARTIAL` 或 `BLOCKED`。
