#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:0:30:00
#$ -o outputs/Llama-3.1-8B/$JOB_ID.log
#$ -e outputs/Llama-3.1-8B/$JOB_ID.log
#$ -p -3

# priority: -5: normal, -4: high, -3: highest

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# training config
CONFIG_FILE=torchtitan/models/llama/train_configs/llama3_8b.toml

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

DISTRIBUTED_ARGS="
  --parallelism.enable_compiled_autograd \
  --parallelism.tensor_parallel_degree ${TENSOR_PARALLEL_SIZE} \
  --parallelism.enable_async_tensor_parallel \
  --parallelism.context_parallel_degree ${CONTEXT_PARALLEL_SIZE} \
"
if $PIPELINE_PARALLEL_SIZE -gt 1; then
  NUM_MICRO_BATCHES=$((${GLOBAL_BATCH_SIZE} / (${MICRO_BATCH_SIZE} / ${DATA_PARALLEL_SIZE})))

  DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --parallelism.pipeline_parallel_degree ${PIPELINE_PARALLEL_SIZE} \
  --parallelism.pipeline_parallel_schedule "1F1B" \
  --parallelism.pipeline_parallel_microbatches ${NUM_MICRO_BATCHES} \
  "
  # schedule: 1F1B, Interleaved1F1B, FlexibleInterleaved1F1B, LoopedBFS, InterleavedZeroBubble
  # ref: https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161
else
  DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS}"
fi

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024  # torchtitan doesn't support global batch size
SEQUENCE_LENGTH=8192

TRAIN_STEPS=27500
LR_DECAY_ITERS=27500

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

TRAINING_CONFIGS="
  --training.batch_size ${MICRO_BATCH_SIZE} \
  --training.seq_len ${SEQUENCE_LENGTH} \
  --training.max_norm ${GRAD_CLIP} \
  --training.steps ${TRAIN_STEPS} \
  --training.mixed_precision_param "bfloat16" \
  --training.mixed_precision_reduce "float32" \
  --training.compile \
  --optimizer.name "AdamW" \
  --optimizer.lr ${LR} \
  --lr_scheduler.warmup_steps ${LR_WARMUP_STEPS} \
  --lr_scheduler.decay_type "cosine" \
  --lr_scheduler.min_lr ${MIN_LR} \
"

# output directory
OUTPUT_DIR=/gs/bs/tga-NII-LLM/checkpoints/torchtitan/llama-3.1-8B
OUTPUT_DIR=${OUTPUT_DIR}/output
PROFILE_DIR=${OUTPUT_DIR}/profiles
CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints

mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROFILE_DIR}
mkdir -p ${CHECKPOINT_DIR}

# wandb
export WANDB_ENTITY="okoge"
export WANDB_PROJECT="torchtitan"
export WANDB_NAME="llama-3.1-8B"

# run training
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -x CPATH \
  -x PATH \
  -bind-to none \
  python torchtitan/train.py \
    --job.config_file $CONFIG_FILE \
    --job.dump_folder ${OUTPUT_DIR} \
    ${TRAINING_CONFIGS} \
    ${DISTRIBUTED_ARGS} \
    --profiling.save_traces_folder ${PROFILE_DIR} \
    --checkpoint.folder ${CHECKPOINT_DIR} \
    --metrics.enable_wandb
