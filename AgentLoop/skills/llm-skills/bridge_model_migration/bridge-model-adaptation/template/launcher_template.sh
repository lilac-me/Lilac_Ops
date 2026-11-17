#!/bin/bash
# ============================================================
# MindSpeed-Bridge 模型启动脚本模板
# 基于 examples/models/vlm/qwen35_vl/qwen35_vl_35b_sft_4k_A3.sh 提炼
# 使用时：
#   1. 替换全部 <PLACEHOLDER>；
#   2. 按"并行度约束"校验 TP/PP/CP/EP 与 WORLD_SIZE；
#   3. 功能验证阶段未明确说明使用都短跑配置（TRAIN_ITERS=15、LOG_INTERVAL=1）；
# ============================================================
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:$PYTHONPATH

# ---- 昇腾 NPU 环境变量（按仓库最新样例保持即可）----
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- 是否加载原始权重（load_weights）----
export HF_LOAD_WEIGHTS=False

# ---- 分布式配置 ----
NPUS_PER_NODE=<NPUS_PER_NODE>        # 每节点 NPU 数，如 8 / 16
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- 并行切分 ----
TP=<TP>
PP=<PP>
EP=<EP>
CP=<CP>
CP_TYPE=<CP_TYPE>
# EP>1 且 TP>1 时必须 sequence_parallel=True（BRIDGE_OVERRIDES 中已默认开启）
# 最少卡数：PP * lcm(TP*CP, ETP*EP) <= WORLD_SIZE
# （MindSpeed recipe 默认 ETP=1；EP 与 TP*CP 能整除时 = PP * max(TP*CP, EP)）

# ---- 模型结构参数
NUM_LAYERS=<NUM_LAYERS>
NUM_MOE_EXPERTS=<NUM_MOE_EXPERTS>
MTP_LAYERS=<MTP_LAYERS>

# ---- 模型 / 训练 / 数据 ----
RECIPE="<model_family>_<model_size>_<train_mode>_config"
STEP_FUNC="<step_func>"
HF_PATH="<hf_path>"
SEQ_LENGTH=<SEQ_LENGTH>
TRAIN_ITERS=<TRAIN_ITERS>
SAVE_INTERVAL=<SAVE_INTERVAL>
GLOBAL_BATCH_SIZE=<GBS>
MICRO_BATCH_SIZE=<MBS>
LOG_INTERVAL=<LOG_INTERVAL>

LR=<LR>
MIN_LR=<MIN_LR>
LR_WARMUP_ITERS=<LR_WARMUP_ITERS>

# 数据集（以vlm-preloaded为例，其他场景参考cfg.dataset使用方法）
TRAIN_JSON="<train jsonl path>"
TRAIN_IMAGES="<train images fold>"
DATASET_OVERRIDES="\
    --dataset vlm-preloaded \
    dataset.train_data_path=${TRAIN_JSON} \
    dataset.image_folder=${TRAIN_IMAGES} \
    dataset.hf_processor_path=${HF_PATH} \
    dataset.seq_length=${SEQ_LENGTH} \
    "

# 训练参数覆盖
TRAINING_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE} \
    checkpoint.save_interval=${SAVE_INTERVAL} \
    optimizer.lr=${LR} \
    optimizer.min_lr=${MIN_LR} \
    scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    logger.log_interval=$LOG_INTERVAL \
    "
# 需要导出全量config.json时追加如下
#   logger.save_config_filepath="<path to save config.json>"

# 模型/训练/并行参数覆盖（入口固定为 run_recipe）
BRIDGE_OVERRIDES="\
    logger.log_interval=$LOG_INTERVAL \
    model.tensor_model_parallel_size=${TP} \
    model.sequence_parallel=True \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.context_parallel_size=${CP} \
    model.context_parallel_algo=${CP_TYPE} \
    model.num_layers=${NUM_LAYERS} \
    model.num_moe_experts=${NUM_MOE_EXPERTS} \
    model.mtp_num_layers=${MTP_LAYERS} \
    "

# 模型专属结构参数按需追加，例如：
#   model.use_ascend_gdn=False   # Ascend C gdn融合算子开关

mkdir -p ./logs
LOG_FILE=./logs/${RECIPE}_TP${TP}_PP${PP}_EP${EP}_CP${CP}.log

torchrun $DISTRIBUTED_ARGS --module mindspeed_bridge.scripts.training.run_recipe \
    --hf_path $HF_PATH \
    --recipe $RECIPE \
    --step_func $STEP_FUNC \
    $DATASET_OVERRIDES \
    $TRAINING_OVERRIDES \
    $BRIDGE_OVERRIDES \
    2>&1 | tee $LOG_FILE
