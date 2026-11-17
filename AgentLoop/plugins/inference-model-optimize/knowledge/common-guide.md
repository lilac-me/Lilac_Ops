# 推理模型优化通用指南（common-guide）

> 本文件沉淀**跨模型通用**的优化经验与报错关键字速查，供各阶段 Agent 读取参考。
> **单模型 / 机型 / 输入输出场景的完整经验固化在 `knowledge/models/` 下的独立 md 文件**
> （命名：`模型-机型-输入x输出.md`，结构模板见 `knowledge/models/_TEMPLATE.md`）。
> 归档只收录实测数据（吞吐/TTFT/TPOT + 环境配置），不写未验证推测。

## 通用经验

### 拓扑与并行

（EP 是否必开、中等 TP 优先、`dcp ≤ tp/H` 约束、KV 头数边界等跨模型通用结论。）

- **MoE 模型：EP 优先开启**（35B +21%）
- **DCP 适用性**：`dcp ≤ tp/H`（H=kv_heads）；先加大 TP 再开启 DCP ；DCP 收益在长输出场景下最大
- **DCP + 并发上探**：DCP 的原理是沿 T 维在 TP rank 间分摊 KV cache，削减 KV cache 复制（tp/H 倍 → 1 倍），因此使能 DCP 后每卡显存释放（长输出场景可释放可观 KV 显存），并发上界随之抬升——评估 DCP 时必须同步测两个场景，只测「并发不变」会漏掉 DCP 的真正收益：
  1. **并发不变**：DCP 前后同一并发对比（验证 DCP 本身带来的吞吐/TPOT 增益与 KV 削减）
  2. **并发增加**：DCP 后按释放显存重算并发预算（`Maximum concurrency @ max-model-len`），上探 1.25x / 1.5x 档位——显存释放越多，越可能突破原并发饱和点，获得更大吞吐
    - 实测佐证（950DT 35B TP4/DP2/EP8+DCP2+KVOFF）：512→640 并发上探（+5.4%）
    - 判据：并发上探至吞吐回落（TTFT 排队恶化/预算超限）为止，取峰值档
- **并发寻优**：饱和点需实测（35B=384、27B=128）；高并发可能因 TTFT 排队灾难（568s）→ 吞吐优先也需控制并发；预算 = DP × 单 rank Maximum concurrency；**DCP 使能后预算重算，原饱和点不再是上限**

### 环境

（A3 vs 950DT 的环境差异：PATH、atb/set_env.sh、bisheng、磁盘、时钟漂移等。）

- 磁盘满时日志/结果写 /tmp（overlay 可用）

### 服务与 bench

- server 建议显式 `--served-model-name`，否则 served name=模型完整路径，bench 必须完全一致否则 404

### 排障原则

（先救后换、禁止并行 serve_start、启动超时先看 stderr 关键字等。）

- 先救后换：每个候选最多 2 次修复机会再迭代
- 启动超时先看 stderr 关键字（HCCL_BUFFSIZE_EP、EI 系列），勿只看 EngineCore 超时
- bench 结果丢失先查容器存活（可能被外部 SIGKILL），勿误读为引擎挂起
- 禁止 orchestrator/developer/bench 并行调用 serve_start（互相 SIGTERM）

## 报错关键字速查

（报错关键字 → 根因 → 修复 env / 命令 等。跨模型通用报错集中在此；模型特异性报错记入该模型自己的 md。）

| 报错关键字 | 根因 | 修复 |
| --- | --------------------------------------------- | ---------------------------------------------------------------- |
| `HCCL_BUFFSIZE_EP is too SMALL ... NEEDED_HCCL_BUFFSIZE` | MoE dispatch/combine CCL 缓冲不足 | `--extra-env HCCL_BUFFSIZE=<MB>`（> NEEDED） |
| `No closing quotation`（shlex） | PowerShell 剥引号 | 外层单引号 + 内层 `\"` 或等号形式 |
| pydantic DCP Value error | DCP 与 KV 头/TP 校验失败 | TP 需 > KV 头数 |
| `EI0020 Bind_IP_Port` | HCCL socket 端口耗尽 | `HCCL_NPU_SOCKET_PORT_RANGE=58000-59999` |
| `libatb.so OSError` | 缺 atb env | source `nnal/atb/set_env.sh` |
| `torch.OutOfMemoryError` in `async_all_to_all`（MoE a2a） | gmu 0.95+高并发 prefill 峰值挤占，无余量 | gmu 回退 0.9 |
| `Free memory on device (X/91.75 GiB) ... less than desired GPU memory utilization (0.9, Y)` | 残留 VLLM EngineCore/Worker 进程占显存（如 DCP 失败未清） | `kill -9 <EngineCore/Worker pid>`（容器内 php/python 列 ps 后清理）后再启动 |
| `Container ... is not running`（docker exec） | 容器已被外部停止（环境不可用） | 勿视为引擎挂起；确认容器/环境后再继续 |

## 归档约定

各阶段 Agent 运行期把新结论立即追加到本 run 的 `OPTIMIZATION-NOTES.md`，
带分类标签 `[模型]` / `[通用]` / `[负面]`；finalize 收敛去重后归档：

- **模型相关的（`[模型]` / `[负面]`）** → 归入对应模型的独立 md
  （`knowledge/models/模型-机型-输入x输出.md`）：已存在则更新/追加，不存在则复制
  `_TEMPLATE.md` 新建；同机型同根因只留一条，标注最早发现日期。
- **跨模型通用的（`[通用]`）** → 归入本文件「通用经验」；**报错类** → 本文件「报错关键字速查」。
