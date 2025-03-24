#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:1:00:00
#$ -o outputs/install/$JOB_ID.log
#$ -e outputs/install/$JOB_ID.log
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

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

pip install -r requirements.txt

# Install PyTorch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall

# install torchtitan
pip install -e .

# torchao
# https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#installation
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124
