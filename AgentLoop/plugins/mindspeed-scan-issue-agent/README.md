# mindspeed-scan-issue-agent 0.4.1

面向任意 GitCode `owner/repo` 的本地 Issue 分析与安全回复插件。支持多仓配置、全量分页、评论与成员读取、代码检索型 shadow 草稿、受保护的模板回复以及本地 Observatory。

## 快速试用

以下命令在本插件目录执行。需要 Python 3.10+，运行时只依赖标准库。

```powershell
python scripts/issue_agent.py check
python scripts/issue_agent.py check
```

批量扫描需要只读 Token，以可靠判断维护者身份。首次使用建议直接启动下面的网页，
在“Agent 设置中心”输入 Token；它会保存在 Windows 凭据管理器，而不是聊天、
浏览器存储、SQLite 或仓库文件。旧环境变量仍作为兼容方式：

`scan` 和多仓配置的 `max_issues` 默认值为 `0`，即扫描所有开放 Issue。以下命令显式限制为 5 条，供首次预览使用。

```powershell
$env:GITCODE_READ_TOKEN = '<只读 token>'
python scripts/issue_agent.py scan --repo owner/repo --mode dry-run --max-issues 5
```

兼容已有环境变量 `GITCODE_TOKEN`。如果用户明确接受公开 API 的不完整预览，可使用：

```powershell
python scripts/issue_agent.py scan --repo owner/repo --mode dry-run --max-issues 5 --allow-public-degraded
```

该降级结果不能可靠判断成员身份，不得用于自动发布或声称得到了完整待回复清单。

`dry-run` 严格零本地写入。保存代码证据和草稿使用 shadow：

```powershell
python scripts/issue_agent.py scan --repo owner/repo --mode shadow --issue 123
```

也可指定已有本地代码仓，避免重复 clone：

```powershell
python scripts/issue_agent.py scan --repo owner/repo --mode shadow --issue 123 --workspace D:\workspace\repo
```

## 多仓运行

复制并修改 `config.example.json`：

```powershell
python scripts/issue_agent.py scan-all --config config.example.json
python scripts/issue_agent.py schedule --config config.example.json --interval-seconds 3600
```

配置中的每个仓库独立保存 SQLite、报告、代码缓存和发布账本。

插件自有代码缓存按 `--default-branch` 指定的分支更新；未指定时使用远端默认分支。每次分析前获取并检出所选分支的最新提交。缓存存在本地改动、分支不存在或获取失败时，会报告代码准备失败，不再使用旧代码继续分析。通过本地路径指定的用户工作区保持原样。

共享同一 SQLite 账本的发布任务通过事务原子抢占事件，只有成功占位的任务可以发送评论；交互发布的一次性审批也在同一事务内消费。处于 `posting` 或 `pending_delivery` 的 Issue 会阻止新的发布占位，需先核对远端结果。
定时真实模板回复还必须显式增加 `schedule --allow-post`，并满足下述全部环境门禁；
默认调度只生成本地结果。

## 完整网页版 Observatory

用户已经指定仓库后，可直接启动独立的完整
[`mindspeed-issue-observatory`](https://gitcode.com/ye_qm/mindspeed-issue-observatory) 网站：

```powershell
python scripts/issue_agent.py visualize --repo Ascend/MindSpeed-MM --port 5173
```

首次运行会在数据目录的 `web-observatory/app` 下克隆 0.4.0 对应的网站分支、执行 `npm ci` 和生产构建，
后续启动复用已有安装。要求 Git、Node.js 24+ 和 npm。启动器只监听 `127.0.0.1`，完成
健康检查后自动打开浏览器；未配置只读 Token 时直接打开 `/agent-settings`。可使用以下命令查看状态或停止服务：

```powershell
python scripts/issue_agent.py visualize-status
python scripts/issue_agent.py visualize-stop
```

无 Token 也允许启动设置页，但其采集数据仅代表公开 API 可见范围；配置只读 Token 后
重启 Observatory，才会按凭据范围重新采集。网站设置页只接受同源本机请求，不能通过
反向代理开放到公网。

原有 Ascend 下的 MindSpeed-MM、MindSpeed-LLM、MindSpeed、FSDPTurbo 和
MindSpeed-Bridge 保持原有历史数据键。新增 GitCode `owner/repo` 可从设置页加入可信列表，
并从设置页打开该仓库看板；其他仓库从零开始积累独立快照。

## 轻量诊断页

旧的 `serve` 只用于读取插件自身 SQLite 的轻量诊断，不是完整网页版。先执行一次非
dry-run 扫描，然后运行：

```powershell
python scripts/issue_agent.py serve --data-dir <data-dir> --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。服务拒绝绑定非 loopback 地址。
只查看单仓时可改用 `--db <data-dir>\repos\<repo-slug>\state.sqlite3`。

## 真实模板回复

默认不会发布。推荐流程：

1. 在 `/agent-settings` 保存发布 Token，并将准确的 `owner/repo` 加入可信列表；
2. 为该仓库打开“交互发布”，点击“保存仓库配置”；
3. 运行 `scan --mode template-auto --repo owner/repo --issue N`，只生成模板草稿；
4. 在网页的“单次发布审批”中查看完整草稿并确认：授权仅限仓库、Issue、正文哈希，5 分钟内有效且只能使用一次；
5. 运行相同的 `scan ... --allow-post`。发布前重新读取 Issue、评论和成员，发布后回读远端评论确认。

```powershell
python scripts/issue_agent.py scan --repo owner/repo --mode template-auto --issue 123
# 在本机网页检查并授权此草稿后：
python scripts/issue_agent.py scan --repo owner/repo --mode template-auto --issue 123 --allow-post
```

先在测试仓验证。若 POST 结果不明确，账本记录 `pending_delivery`，程序不会自动重复 POST。
无人值守的模板发布另外需要在设置页为目标仓库开启“定时发布”，并在调度命令中传
`--allow-post`；默认关闭。旧的 `GITCODE_WRITE_TOKEN` 仍可作为凭据来源；
`GITCODE_POST_TARGET` 与 `GITCODE_ALLOW_REAL_POST` 不再绕过仓库策略或单次授权。

## 数据目录

0.4.0 默认使用 Windows `%LOCALAPPDATA%\9-17-scan-issue-agent\`，避免切换对话或工作目录后丢失配置；旧版当前目录下的 `.gitcode-issue-agent/` 不会被自动移动。可通过 `GITCODE_ISSUE_AGENT_DATA` 或 `--data-dir` 显式使用旧目录。每仓目录包含：

```text
repos/<host--owner--repo>/
├── state.sqlite3
├── reports/latest.json
└── worktrees/<host--owner--repo>/
```

## 测试

### 插件打包

唯一正本位于仓库根的 `skills/common/common/gitcode-issue-triage/`；只修改和提交该目录。
插件内的 `skills/` 是本地生成的软链接或 Windows junction，已被 Git 忽略。
从仓库根目录执行：

```powershell
python tools/install-unit.py link mindspeed-scan-issue-agent
python plugins/mindspeed-scan-issue-agent/scripts/package_plugin.py --output .tmp/issue-plugin-package
```

打包输出目录必须尚不存在。打包器直接从正本复制全部入口、引用资源和桥接脚本，
生成不依赖仓库或链接的独立插件目录；该目录可作为发布包的插件源。
不再使用 `sync_skills.py`，也不提交生成的链接或打包文件。

安装前先按仓库说明注册本地 marketplace，再使用以下入口（或 `install.bat` / `install.sh`）：

```powershell
python tools/install-unit.py codex mindspeed-scan-issue-agent --run
python tools/install-unit.py claude mindspeed-scan-issue-agent --run
```

安装工具先创建链接，在临时目录中构建完整包，再临时将插件的技能视图换成真实文件，
供 marketplace 安装器复制；成功或失败后均恢复开发链接。不要并发安装同一插件。
若进程被强制终止且遗留 `.skills-install-backup`，应先检查并恢复备份，再重新安装。
直接从未经打包的 Git 插件子目录安装不会自动生成技能，应使用上述入口或物化后的发布包。
旧版真实副本目录若仍存在，链接工具会报错；先备份其中的本地修改并移除副本，再创建链接。
DSH 继续使用 `python tools/install-unit.py dsh mindspeed-scan-issue-agent` 投影正本。

### 回归测试

```powershell
python -m unittest discover -s tests -v
```

插件已更名为 `mindspeed-scan-issue-agent`。为兼容已有配置、凭据和发布账本，内部数据目录及凭据键继续沿用旧名称，无需重新设置 Token。
