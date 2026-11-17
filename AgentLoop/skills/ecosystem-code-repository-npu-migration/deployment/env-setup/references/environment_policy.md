# Conda 环境策略

绝不修改系统 Python，也不安装到 conda base。

## 选择

1. 现场校验通过后，复用用户明确提供的非 base 绝对 Python。
2. 只有重新运行 base gate 后，才能复用本次任务中之前创建的同一 Python。
3. 其他情况使用 mamba 或 conda 创建专用环境。

不要接管任意已有环境，也不要依赖跨任务记忆。

## 命名与冲突

将仓库目录规范化为小写 slug。计算绝对仓库路径的 SHA-256 十六进制摘要前 8 位，并使用：

```text
<repo-slug>-npu-py<major><minor>-<repo-path-hash>
```

如果名称已存在且不是本任务中创建的环境，追加 UTC 时间戳。绝不删除、覆盖或静默复用它。

优先使用 mamba，否则使用 conda。解析新环境的绝对 Python，并将其用于所有命令。两种管理器都不存在时返回 `BLOCKED_ENVIRONMENT`；不要退回 venv、系统 Python、裸 `python` 或 base。

## CANN

只校验并加载用户选择的 CANN 路径。不要安装、替换或静默选择系统 CANN 目录。
