# 兼容性研究

在修改环境前，使用当前官方证据选择候选组合。安装结果必须由本地执行证明，不能只依赖文档。

## 来源顺序

1. 与版本对应的昇腾 PyTorch 安装和兼容性页面。
2. 昇腾 PyTorch 扩展仓库、发布版本、tag 和 wheel 元数据。
3. 选定安装线对应的 CANN 发布说明。
4. 目标仓库需要的官方 Python 和软件包元数据。

有用的官方来源类别包括：

- `https://ascend.github.io/docs/sources/pytorch/install.html`
- `https://github.com/Ascend/pytorch`
- `hiascend.com` 上按版本提供的 PyTorch 迁移和 API 支持文档

搜索与请求发布版本匹配的页面，不要把示例版本 URL 当作永久有效的来源。

## 解析规则

- 将明确的 CANN、torch、torch_npu 和 Python 版本视为硬约束。
- 只根据当前官方证据解析 `torch_npu=auto`。
- 先比较 torch 与 torch_npu 的 `major.minor.patch` 基础发布版本线，再根据官方矩阵核对精确后缀和 wheel。
- 没有受支持的 wheel 满足该组合时应阻塞，除非用户授权更换版本或源码构建。
- 将本地观察到的软件包与文档声明的兼容性分开记录。

在当前任务中返回精确 URL、访问日期、使用的结论、选定的安装顺序和不确定性；不要写入环境产物。
