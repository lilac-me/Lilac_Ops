# 多仓运行

多仓配置的 `repositories` 数组中，每项包含 `name: owner/repo`，并可设置 `mode`、
`max_issues`、`workspace` 和额外 `maintainers`。默认模式为 `dry-run`。

不同仓库的 SQLite、报告、代码缓存和发布账本按仓库目录隔离。批量运行前统一检查
只读 Token；若未配置，不应逐仓执行后再返回大批身份未知结果。

定时执行默认不得发布。即使配置项为 `template-auto`，仍需命令行显式请求发布并满足
写 Token、精确目标仓锁和发布总开关。
