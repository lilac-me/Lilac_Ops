# ecosystem-code-repository-npu-migration — 插件说明（Claude Code 与 Codex 双平台）

## 范围

- `plugins/ecosystem-code-repository-npu-migration/` 是本插件的基础设施真相（双清单、共享知识、打包/校验脚本、测试、bundle lock）；8 个 skill 的正本位于仓库根 `skills/ecosystem-code-repository-npu-migration/<function>/<skill-name>/`。
- 本插件同时支持 Claude Code 与 Codex：`.claude-plugin/plugin.json` 与 `.codex-plugin/plugin.json` 两份清单的版本和 skill 列表必须保持一致（`set_plugin_version.py` 统一同步）。无论哪个平台，都不引入 JavaScript 工作流或固定的 Python 阶段控制器。
- `skills/` 是由 `install.sh` / `install.bat` / `plugin_bundle.ensure_skill_links()` 生成的同名软链接视图（不入 git），保持两份插件清单所需的扁平 `skills/<skill-name>` 布局。skill 使用仓库规范富 frontmatter（version / domain: ecosystem-code-repository-npu-migration / function / tags / author）。
- skill 引用插件级 `references/`、`scripts/` 资源必须写 `<plugin-root>/...` 锚点，不得使用 `../../`：软链接路径下的 `..` 按链接目标（canonical 路径）解析，会指到 `skills/ecosystem-code-repository-npu-migration/` 下的错误位置。

## 编排边界

- `$ecosystem-code-repository-npu-migration:goal` 是仓库级入口。主 Agent 负责理解目标仓库、维护计划、调用 specialist skill、修改源码、执行真实命令和判断下一步。
- Python 脚本只负责确定性探测、smoke、发现、失败分类或产物校验，不负责阶段顺序、自动重试、队列、游标、尝试树或支持结论。
- `migration_progress.md` 和 `compatibility.json` 只是 context compaction 后的工作记录，不是流程控制状态，也不能代替真实 NPU 证据。

## 维护规则

- 稳定领域知识放在 `references/`，可重复的确定性操作放在对应 skill 的 `scripts/`，入口流程保留在 `SKILL.md`。
- 调整 skill 集合或 function 归属时，必须同步以下各处并重新生成 bundle lock：`plugin_bundle.py` 的 `EXPECTED_SKILLS` + `SKILL_FUNCTION`、`install.sh`、`install.bat`（含 `:stage_npu_skills_for_install` / `:restore_npu_skill_junctions`）、`SKILL.md` frontmatter、`.claude-plugin/plugin.json` 的 skills 数组（Claude 侧显式列出每个 skill）。
- 保留目标仓库现有 CPU/GPU 行为和未要求修改的版本路径。
- 不提交 `.npu-agent`、缓存、认证信息和 `__pycache__`。
- 修改 skill 后，同时检查它引用的 references、scripts、examples 和 schemas；不要只验证 Markdown 是否可读。

## 验证

```bash
NPU_MIGRATE_PYCACHE_DIR="$(mktemp -d)"
PYTHONPYCACHEPREFIX="$NPU_MIGRATE_PYCACHE_DIR" python -m py_compile $(find plugins/ecosystem-code-repository-npu-migration -type f -name '*.py' | sort)
python plugins/ecosystem-code-repository-npu-migration/scripts/build_bundle_lock.py --check
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_plugin_package.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_bundle.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_skill_design.py --json
python /home/zyh/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py plugins/ecosystem-code-repository-npu-migration
python -m pytest plugins/ecosystem-code-repository-npu-migration/tests
```

校验器与测试会自动重建软链接视图（fresh clone 可直接运行；Windows 无符号链接权限时先运行 `install.bat`）。这些检查只证明插件结构、资源引用和提示词契约；目标仓库是否真正支持昇腾 NPU，仍需目标仓库的实际运行证据。
