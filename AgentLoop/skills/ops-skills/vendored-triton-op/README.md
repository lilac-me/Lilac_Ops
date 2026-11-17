# vendored-triton-op

本目录是 **cannbot 性能调优模块** 的版本固定快照（vendor 副本），由
`scripts/vendor-triton-op-generator.sh` 生成，只保留调优相关 skill（verifier /
latency-optimizer / simulator-optimizer / precision-debug / npu-arch / mfu-analyzer）+ 会话
归档渲染器 `render_session.py`。**不重新生成算子**——extractor/designer/coding 不入列。

## 来源与生成方式

上游只有一个仓：`cannbot`。它的 `triton-op-generator` 自带统一安装器
（`plugins-official/triton-op-generator/init.sh` → `script/bin/cannbot.js`），由安装器负责：

- 拉取共享 skill 仓 `cannbot-skills`（cannbot 的 `vendor/cannbot-skills` submodule，
  可用 `--skills <SHA|本地路径>` 覆盖）；
- 克隆插件依赖仓 `cannbot-knowledge` 到 `<安装目录>/.cannbot/dependencies/`；
- 把 skill 软链装到 `<安装目录>/.claude/skills/`，插件资产装到
  `<安装目录>/.cannbot/plugins/triton-op-generator/`。

本目录的脚本只是：拉 `cannbot` → 跑它的安装器（装到 `.vendor_src/cannbot_install`）→
从安装产物取上述 skill 与渲染器快照。**不直接 clone cannbot-skills，也不改写 vendor 内容。**

- 运行时与上游完全解耦：S4 只读取本目录副本。
- 升级：重跑 `scripts/vendor-triton-op-generator.sh`（可设 `CANN_BOT_TAG` 固定 cannbot 版本、
  `CANN_BOT_SKILLS` 固定 skill 仓版本）。
- 来源仓 / 锁定 sha / 各 skill 版本 / 安装器记录见仓库根 `VENDORED_TRITON.json`。
- 知识库（KB）不在本目录：需要 KB 卡片或 `knowledge-query` 时设 `CANNBOT_KNOWLEDGE_ROOT`
  指向 cannbot-knowledge 的 `knowledge` 目录，或复用安装器克隆的
  `.vendor_src/cannbot_install/.cannbot/dependencies/triton-op-generator/cannbot-knowledge/knowledge`。

> 勿直接在本目录手工修改；如需临时改动请改后回到上游维护，再重跑脚本回填。
