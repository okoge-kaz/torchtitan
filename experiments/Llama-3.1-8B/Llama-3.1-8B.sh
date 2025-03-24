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

# output directory
OUTPUT_DIR=/gs/bs/tga-NII-LLM/checkpoints
OUTPUT_DIR=${OUTPUT_DIR}/torchtitan/llama-3.1-8B/output
mkdir -p ${OUTPUT_DIR}
PROFILE_DIR=${OUTPUT_DIR}/torchtitan/llama-3.1-8B/profiles
mkdir -p ${PROFILE_DIR}
CHECKPOINT_DIR=${OUTPUT_DIR}/torchtitan/llama-3.1-8B/checkpoints
mkdir -p ${CHECKPOINT_DIR}

# wandb
export WANDB_ENTITY="okoge"
export WANDB_PROJECT="torchtitan"
export WANDB_JOB_NAME="llama-3.1-8B"

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
    --profiling.save_traces_folder ${PROFILE_DIR} \
    --checkpoint.folder ${CHECKPOINT_DIR} \
    --metrics.enable_wandb
