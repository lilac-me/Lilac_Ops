# torch_npu Runtime 模型

## torch_npu 的作用

`torch_npu` 是 PyTorch 的昇腾扩展。它注册 NPU 设备/backend，将 PyTorch 操作连接到昇腾 runtime，并提供 NPU 专用 API 和适配；它不是简单改写应用源码，也不应被描述为用任意打补丁代码替换原始 torch wheel。

可用 runtime 是耦合的软件栈：

```text
target repository
  -> PyTorch Python 和 dispatcher API
  -> torch_npu 设备/backend 与 operator 集成
  -> CANN 库以及编译器/runtime
  -> 昇腾驱动和硬件
```

某一层导入成功，不能证明其下层也可用。

## 版本匹配

接受版本组合前，比较 torch 和 torch_npu 的 `major.minor.patch` 基础发布版本线。例如：

- `torch==2.10.0`
- `torch_npu==2.10.0.post4`

两者都属于 2.10.0 基础发布线。`.post4` 是打包/发布修订后缀，不表示 torch_npu 面向不同的 PyTorch 发布线。

`major.minor.patch` 三段版本号一致是必要条件，但不是充分条件。精确的 CANN、Python、架构、wheel 构建版本和 torch_npu 后缀仍以当前官方兼容矩阵为准。不要假设任意 `.postN`、release candidate、本地构建或相同前缀的 wheel 都受支持。

## 依赖安装如何破坏软件栈

目标仓库和可选加速器软件包可能声明 torch 约束。依赖解析器可能卸载选定 wheel、只替换 torch 而不替换 torch_npu、安装 CUDA/CPU 构建，或改变相关二进制软件包。结果可能仍满足普通 Python 元数据，但 NPU backend 已无法加载。

每次安装目标或能力组件前后：

1. 检查已安装 torch 和 torch_npu 的版本及导入路径；
2. 比较它们的基础发布线和当前官方版本组合；
3. 运行 `"<selected-python>" -c 'import torch; torch.npu.set_device(0)'`；
4. 发生漂移时，按文档顺序重新安装固定的官方匹配组合，并重跑该命令；
5. 目标安装的硬性要求无法与请求的 NPU 软件栈共存时，阻塞目标安装。

安装后命令成功退出前，不要认为软件栈已经修复。
