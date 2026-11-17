# MindSpeed-Agent 按需安装架构设计（v2）

> 状态：设计稿（v2 · 待评审落地）
> 上一版：全量安装（`docs/ecosystem-knowledge-graph-design.md` 依赖此安装形态）
> 目标平台：DSH（DeepSeek Harness）/ Claude Code / Codex 三端一致
> 目标用户：既想"装得少"、又想"用到时能快速补装"的 MindSpeed 开发者与新手

---

## 1. 背景：为什么"全量安装"不再成立

当前安装行为（`install.sh` / `install.bat`）：

1. 根插件 `mindspeed-agent`（marketplace source `.`）的
   `.claude-plugin/plugin.json` 声明 `skills: ["./skills"]`、`.codex-plugin` 声明
   `skills: "./skills/"` —— **等价于一次装载全部 40 个技能**；
2. marketplace.json 同时单独列出 `inference-model-optimize` / `ops` /
   `doc-agent` 子插件，与根插件全量装载**冗余**；
3. Claude Code 会话按插件计 **30-skill 上限**（超出被静默截断、技能不可发现，
   见 anthropics/claude-code#13343）；根插件 40 技能必然触发截断；
4. DSH 侧 install 脚本把 `skills/**/SKILL.md` **全量扁平投影**到 `.dsh/skills/`；
5. 结果：**安装即全量**，与"新手只关心一个领域"的诉求相悖；也撑大了每次会话
   的候选技能面，降低路由精度。

### 1.1 目标（用户已确认）

把安装形态从"全量一次装"改为：

```text
安装核心（core）→ 只含意图路由与通用底座
   │  ecosystem-routing skill + catalog/ + 少量通用技能
   ▼
用户说出诉求 → ecosystem-routing 命中目标插件/技能
   │  命中 llm → "请安装 mindspeed-llm 插件"（或自动执行）
   ▼
按需安装该领域插件 → 新开会话 → 目标 skills 可用
```

设计约束（用户确认）：

- **路由后显式按需安装 + 提示重开会话**（Claude Code 会话内无法可靠即装即用，
  见 anthropics/claude-code#35641：`/reload-plugins` 不加载新装插件的 skills）；
- **拆成领域子插件**（llm / vllm-推理 / ops / doc 等各自独立安装单元），
  每插件技能数收敛到 ≤30，从机制上消除截断；
- **DSH / Claude Code / Codex 三平台全部支持**。

---

## 2. 目标架构

```text
MindSpeed-Agent/  ← marketplace（单一 checkout 即市场）
├── .claude-plugin/plugin.json        # 根插件 mindspeed-agent（core 版）
├── .codex-plugin/plugin.json         # Codex 适配
├── .dsh-plugin/plugin.json           # DSH 适配（.dsh/skills 投影由按需工具驱动）
├── skills/common/common/ecosystem-routing/   # core 自带：唯一路由入口
├── skills/<domain>/...               # 全部技能保持 canonical 单源（不动）
├── plugins/
│   ├── mindspeed-llm/                # 领域插件：LLM 训练/迁移/FSDP2/精度（新建）
│   │   ├── .claude-plugin/plugin.json
│   │   ├── .codex-plugin/plugin.json
│   │   └── skills/*                  # junction/symlink → skills/llm-skills/...
│   ├── inference-model-optimize/     # 领域插件：vllm-ascend 推理部署/分析/优化（已有）
│   │   └── skills/*                  # junction/symlink → skills/inference-model-skills/...
│   ├── ops/                          # 领域插件：算子开发流水线（已有）
│   │   └── skills/*                  # junction/symlink → skills/ops-skills/...
│   └── doc-agent/                    # 领域插件：资料生成/审查（已有）
│       └── skills/*                  # junction/symlink → skills/doc-skills/...
├── catalog/plugins.yaml              # 增加 install 元数据（见 §4）
└── tools/install-unit.*              # 按需安装器（跨平台，见 §5）
```

**关键原则**：

1. 技能源 **仍然只有一份**（`skills/<domain>/...`），领域插件内只放
   junction/symlink 视图——与现行 doc-agent/ops 做法一致，不复制技能；
2. **core 永不超限**：根插件 manifest 的 skills 收敛为只含 `ecosystem-routing`
   （可随通用技能演进，始终 ≤30）；
3. 领域插件是"安装单元"（unit）：每 unit 关联 repo 域 + task-kind 域 +
   install 命令提示；`catalog/plugins.yaml` 记录 unit 与提示，路由 skill 只读它
   不硬编码；
4. 平台差异只体现在"装 unit 的命令"（§6）。

---

## 3. Marketplace 插件拆分表

marketplace.json 最终列表（`name → source → 装载 skills 计数`）：

| # | 插件名（marketplace） | source | 域/能力 | skills 计数 | 默认安装 |
| --- | --- | --- | --- | --- | --- |
| 1 | `mindspeed-agent`（core） | `.` | 意图路由 + catalog + 通用底座 | ≈1（ecosystem-routing；随通用演进） | ✅ 是 |
| 2 | `mindspeed-llm` | `./plugins/mindspeed-llm` | llm 训练：迁移 / FSDP2 / 精度 / 训练验证 | 16（llm-skills 全量） | ❌ 按需 |
| 3 | `inference-model-optimize` | `./plugins/inference-model-optimize` | vllm-ascend 推理：部署 / 分析 / 基准 / 剖析 / 寻优 | 12（inference-model-skills 主体） | ❌ 按需 |
| 4 | `ops` | `./plugins/ops` | 算子：Design→Implement→Optimize | 5 | ❌ 按需 |
| 5 | `doc-agent` | `./plugins/doc-agent` | 资料：生成 / 更新 / 四维审查 | 5 | ❌ 按需 |

> 全部 ≤30；去掉现在根插件"40 全量"与 marketplace 子插件冗余共存的问题。
> 计数以 `ci/checks/plugins.py` 的递归 SKILL.md 统计口径为准（§7 CI 会强校验）。
>
> 命名依据（用户确认）：marketplace **不新增推理类插件名**——现有
> `inference-model-optimize` / `ops` / `doc-agent` 即领域插件，仅 llm 训练域
> 尚无独立插件，故新建 `mindspeed-llm` 一个 unit。

### 3.1 技能归属映射（迁移对照）

canonical `skills/` 树 → 目标 unit：

| skills 路径（canonical） | 技能数 | 归属 unit |
| --- | --- | --- |
| `skills/common/common/ecosystem-routing` | 1 | core |
| `skills/llm-skills/**` | 16 | mindspeed-llm |
| `skills/inference-model-skills/**`（除顶层路由） | 12 | inference-model-optimize |
| `skills/ops-skills/**` | 5 | ops |
| `skills/doc-skills/**` | 5 | doc-agent |
| `skills/inference-model-skills/SKILL.md`（域路由说明） | 1（说明文档） | 并入 core 或 mindspeed-llm，随迁移评审决定 |

> 注：`skills/inference-model-skills/SKILL.md` 与
> `skills/llm-skills/llm-model-analysis/SKILL.md` 属"域级路由/分析"说明类技能，
> 迁移时按实际归属判定，避免目录层级 2 层技能（`skill.path-depth` WARN）在新
> 结构里扩散。

---

## 4. catalog 元数据扩展

`catalog/plugins.yaml` 每个插件节点新增（向后兼容，旧字段不动）：

```yaml
plugins:
  - id: mindspeed-agent
    name: mindspeed-agent
    platforms: [claude, codex, dsh]
    install:
      unit: core                 # core | on-demand
      default_installed: true
      skills_root: "skills/common"      # 该 unit 聚合的 canonical 技能根
      install_hint:
        claude: "claude plugin install mindspeed-agent@MindSpeed-Agent"
        codex:  "codex plugin add mindspeed-agent@MindSpeed-Agent"
        dsh:    "python tools/install-unit.py dsh mindspeed-agent"
    repo_affinity: [ ... ]       # 不变
  - id: mindspeed-llm
    install:
      unit: on-demand
      default_installed: false
      skills_root: "skills/llm-skills"
      install_hint:
        claude: "claude plugin install mindspeed-llm@MindSpeed-Agent"
        codex:  "codex plugin add mindspeed-llm@MindSpeed-Agent"
        dsh:    "python tools/install-unit.py dsh mindspeed-llm"
    ...
```

`generator.py` 的 `graph.json` 相应输出 `nodes.plugins[].install`，供
ecosystem-routing skill 读取"命中 unit → 该用哪条命令安装"。

---

## 5. 按需安装器（tools/install-unit.py）

单一 Python 入口，三平台复用；不引入第三方依赖（PyYAML 可选）：

```text
python tools/install-unit.py <platform: claude|codex|dsh> <unit: mindspeed-llm|...>
python tools/install-unit.py status                     # 列出 core 已装、按需未装
```

行为：

- `claude` / `codex`：输出（或可选自动执行）`install_hint` 命令；
  由 `--run` 开关决定仅打印还是执行；执行成功后提示"请新开会话使插件生效"；
- `dsh`：把该 unit 的 canonical 技能（`skills_root/**/SKILL.md`）投影为
  `.dsh/skills/<name>` junction/symlink（复用现有 install 脚本 dsh 分支逻辑，
  收敛到按 unit 投影），并提示新开会话重扫；
- `status`：读 `catalog/plugins.yaml` 的 `install.default_installed` + 各平台
  已装清单（claude plugin list / codex plugin list / .dsh/skills 目录）输出状态。

`install.sh` / `install.bat` 收敛为：**默认只装 core**，保留 `all`/历史参数
兼容（见 §7 迁移）。

---

## 6. 三平台按需安装流程

### 6.1 会话内（用户已装 core）

```text
用户:"我想优化 mindspeed-llm 的模型"
 → ecosystem-routing（core 自带）R1 消歧 → repo=mindspeed-llm
 → R2/R3 任务=perf-optimize → unit=mindspeed-llm
 → 检查：mindspeed-llm 插件是否已装？（claude plugin list / codex plugin list /
         .dsh/skills 中 mindspeed-llm 域是否投影）
 → 未装 → 输出 install_hint（§4）→ 用户执行 → 新开会话继续
 → 已装 → 直接进入该 unit 的 agent/skill 链
```

### 6.2 各平台命令速查

| 动作 | DSH | Claude Code | Codex |
| --- | --- | --- | --- |
| 安装 core（默认） | `install.sh dsh`（只投影 core） | `install.sh claude-code` | `install.sh codex` |
| 按需安装某 unit | `python tools/install-unit.py dsh mindspeed-llm` | `claude plugin install mindspeed-llm@MindSpeed-Agent` | `codex plugin add mindspeed-llm@MindSpeed-Agent` |
| 查看已装 | `Get-ChildItem .dsh/skills` | `claude plugin list` | `codex plugin list` |
| 装完生效 | 新开会话 | **必须新开会话**（reload 不可靠） | 新开会话 |

---

## 7. CI 与迁移

### 7.1 CI 扩展（ci/checks/ 增量）

1. `plugins.py`：把 `plugin.skill-cap` 从 WARN 提升为 **BLOCK**（拆分后每 unit
   必须 ≤30，不再允许存量豁免）；
2. 新增 `catalog` 校验：`plugins.yaml` 的 `install.skills_root` 必须真实存在、
   `install_hint` 与 marketplace 插件名一致（挂入 `ci/checks/knowledge_graph.py`）；
3. `marketplace.py`：校验 marketplace 列表 == core + 全部 on-demand units
   （防止新增 unit 忘登记）。

### 7.2 迁移路径（对现有用户）

- `install.sh all` / 无参数行为保持可用（= 装 core + 全部 unit），老用户脚本
  兼容；新增 `install.sh core`（默认推荐）与 `install.sh <unit>`；
- 领域技能 canonical 位置 `skills/<domain>` **不动**，仅新增领域插件目录与
  junction 视图 + marketplace 登记，git 历史与现有 skill PR 流程不受影响；
- 本项目 catalog / routing / guard 校验全部保持通过后再合入。

---

## 8. 验收

1. 全新安装 `install.sh dsh`（core）后：`.dsh/skills` 只含 `ecosystem-routing`
   与通用底座（**不再 40 个全量**）；
2. 金句 `"我想优化 mindspeed-llm 的模型"` → 路由命中 unit `mindspeed-llm` →
   输出可执行 install_hint → 执行并新开会话后，`mindspeed-llm-fsdp2-main-control`
   等 skills 可见可用；
3. 每个 marketplace 插件 `ci/guard.py` 校验 skills ≤30 且 **0 BLOCK**；
4. Claude Code / Codex / DSH 三条 install_hint 命令均在文档/status 中可查。

---

## 9. 风险与决策点

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| Claude Code reload 不加载新装 skills | 会话内"装完即用"不可靠 | 显式提示新开会话（用户已确认此语义） |
| unit 拆分导致 skill 归属争议 | 个别技能跨域（如 llm-model-analysis） | 迁移按 §3.1 表评审，归属可调整，不硬编码 |
| 旧 install 全量行为被破坏 | 老用户脚本依赖 | `all`/历史参数保留；core 为新增默认推荐 |
| 顶层说明类 SKILL.md（2 层目录） | 现为 WARN 的 path-depth | 迁移时并入对应 unit 的根说明或单独归 core |

---

## 10. 变更清单（落地顺序）

1. `catalog/plugins.yaml`：追加 `install` 元数据（§4）；
2. `catalog/checks_core.py` + `generator.py`：install 字段解析与图输出；
3. 新增领域插件目录 `plugins/mindspeed-llm`
   （.claude-plugin/.codex-plugin/plugin.json + skills junction 视图）并登记
   marketplace.json；`inference-model-optimize` / `ops` / `doc-agent` 转为
   unit 形态（manifest skills 收敛为各自领域，不再依赖根插件全量）；
4. 根插件 `.claude-plugin/plugin.json` / `.codex-plugin/plugin.json`：
   skills 收敛为 core（不再 `./skills` 全量）；
5. `tools/install-unit.py`（按需安装器）；
6. `install.sh` / `install.bat`：默认 core + `install.sh <unit>` + 保留 all；
7. CI：skill-cap 升 BLOCK、install 元数据校验、marketplace 完整性校验；
8. `ecosystem-routing` SKILL.md：加入"unit 未装 → install_hint + 重开会话"步骤；
9. README 安装章节重写；本文档并入仓库 docs/。
