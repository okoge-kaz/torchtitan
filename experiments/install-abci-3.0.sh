#!/bin/sh
#PBS -q rt_HG
#PBS -N install
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/install/

cd $PBS_O_WORKDIR
mkdir -p outputs/install

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.4
module load cudnn/9.1.1
module load nccl/2.21.5
module load hpcx/2.18.1

source .env/bin/activate

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall

# torchao
# https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#installation
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124
