# 发布安全

真实评论发布必须同时满足：

1. 用户已审阅目标 Issue 和最终草稿，并明确授权本次发布；
2. 模式为 `template-auto` 且命令包含 `--allow-post`；
3. 发布 Token 已通过本地 Observatory 设置页保存到 Windows 凭据管理器；
4. 目标 `owner/repo` 在可信仓库列表中，且“交互发布”已开启；
5. 用户在网页审阅完整模板草稿后，为该仓库、Issue、正文哈希签发 5 分钟一次性授权；
6. 发布前重新读取 Issue、评论和成员身份；
7. 发布后回读远端评论确认。

授权只对本次明确目标有效。超时或回读失败时记录 `pending_delivery`，不得自动重复 POST。
定时无人值守发布只适用于预设模板，还要求目标仓库单独开启“定时发布”，并在
`schedule --allow-post` 中显式请求；不能用交互发布授权代替。旧的
`GITCODE_POST_TARGET` 与 `GITCODE_ALLOW_REAL_POST` 不再构成发布授权，新流程无需设置。
