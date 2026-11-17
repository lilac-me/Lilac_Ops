#!/bin/bash
# ============================================================================
# 昇腾 NPU 训练环境变量基线（VLA / VLM / 世界模型）
#
# 用法：训练脚本开头 `source test/env_npu.sh`（交付时由 test/env_npu.sh 承载）
#
# 变量收录判据（两条，**变量**只收满足这两条的，不要加入其他变量）：
#   ① VLA/VLM 系实测通用（下注的 x/15 是 15 个 VLA/VLM/世界模型 test 脚本里的覆盖数）；
#   ② **偏离官方默认值** —— 写默认值是 no-op，一律不收
#      （所以这里没有 ASCEND_SLOG_PRINT_TO_STDOUT / ASCEND_GLOBAL_LOG_LEVEL /
#        ASCEND_GLOBAL_EVENT_ENABLE / HCCL_WHITELIST_DISABLE，它们的默认值就是脚本里那几个数）。
# 例外：文末的 Device 侧日志降级块（msnpureport）不是环境变量、不占判据名额，
#       属固定降噪动作，纯可选、可整段注释掉 —— 说明见 ../references/appendix-env.md §一。
#
# 变量含义、取值、出处、语料覆盖统计：../references/appendix-env.md
# 交付件里**不要**出现这个文件之外的临时调试变量，尤其是 ASCEND_LAUNCH_BLOCKING（见文末）。
# ============================================================================

# ---- 算子下发二级流水（VLA/VLM 15/15 都设，收益最直接）----
# 0 关 / 1 Level1（默认）/ 2 Level2。官方建议值就是 2。
export TASK_QUEUE_ENABLE=2

# ---- CPU 绑核（15/15）----
# 0 关（默认）/ 1 粗粒度 / 2 细粒度。不设 = 不绑核。
export CPU_AFFINITY_CONF=1

# ---- 显存（11/15；长序列模型建议开）----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 杂项 ----
export TOKENIZERS_PARALLELISM=false   # 关 tokenizer 多进程告警
export WANDB_MODE=offline             # 离线环境

# ---------------------------------------------------------------------------
# 按场景开（默认注释掉；开了就把这项记进档案卡口径）
# ---------------------------------------------------------------------------
# export MULTI_STREAM_MEMORY_REUSE=2   # FSDP2 场景实测有效（降低显存波动）
# export TORCH_HCCL_ZERO_COPY=1        # 仅 A3；beta 特性，依赖上面的 expandable_segments
# export COMBINED_ENABLE=1             # 非连续两算子组合优化
# export HCCL_SOCKET_IFNAME=xxx        # 多网卡且自动选错时指定通信网卡

# ---------------------------------------------------------------------------
# 仅精度跑开（有性能代价）—— **性能跑与交付件里必须保持注释**
# ---------------------------------------------------------------------------
# export HCCL_DETERMINISTIC=true
# export CLOSE_MATMUL_K_SHIFT=1

# ---------------------------------------------------------------------------
# Device 侧日志降级（可选，只减少日志噪音，不设也能跑；卡数按实际改）
#
# 注意两点，否则这段会变成噪音源：
#   ① 容器里 msnpureport 需要 --docker，否则每个卡号刷一行
#      「msnpureport in docker should be used with --docker.」而**它是打 stdout 的，
#        `2>/dev/null` 挡不住** —— 所以这里连 stdout 一起丢；
#   ② 没装/不可用时不该报错，先 `command -v` 探一下。
# ---------------------------------------------------------------------------
if command -v msnpureport >/dev/null 2>&1; then
    # 容器里必须带 --docker：不带的话命令**直接失败**（连 `-r` 读当前级别都报
    # 「should be used with --docker」）—— 语料里这段在容器中一直是空操作。
    # 这里用「读一次级别」来探测，不依赖 /.dockerenv 这类运行时特定标记。
    MSP_ARGS=""
    if ! msnpureport -r >/dev/null 2>&1 && msnpureport --docker -r >/dev/null 2>&1; then
        MSP_ARGS="--docker"
    fi
    for i in $(seq 0 7); do
        msnpureport $MSP_ARGS -g error -d $i >/dev/null 2>&1
    done
    msnpureport $MSP_ARGS -e disable >/dev/null 2>&1
fi

echo "[env_npu.sh] NPU environment configured."

# ============================================================================
# ⚠️ ASCEND_LAUNCH_BLOCKING=1 —— 只允许 debug 阶段临时用（同步执行，便于定位算子级报错）。
#    它会让算子同步下发、性能大幅下降，并且会让 TASK_QUEUE_ENABLE 整体失效（官方约束）。
#    **性能跑与交付件里必须删掉**；交付前用下面这条自查：
#        grep -rn "ASCEND_LAUNCH_BLOCKING" test/ && echo "交付件里残留了调试变量！"
#    （语料 67 个模型的 test 脚本里，这个变量一次都没出现过 —— 出现即漏删。）
#    危害机制与官方约束见 ../references/appendix-env.md §四。
# ============================================================================
