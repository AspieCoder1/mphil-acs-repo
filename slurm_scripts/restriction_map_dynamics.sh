#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J restriction_map_dynamics
#SBATCH --output=slurm_output/restriction_map_dynamics/out/%A_%a.out
#SBATCH --error=slurm_output/restriction_map_dynamics/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-44
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODELS=( diag_sheaf bundle_sheaf general_sheaf )
DATASETS=( dblp acm imdb )

IDX=${SLURM_ARRAY_TASK_ID}

N_TRIALS=5
N_DATASETS=${#DATASETS[@]}
N_RUN=$((IDX / N_TRIALS))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))
DATASET=${DATASETS[DATA_IDX]}
MODEL=${MODELS[MODEL_IDX]}

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load gcc/11

export WANDB_CACHE_DIR="~/rds/hpc-work/.wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python sheaf_nc.py model="${MODEL}" dataset="${DATASET}" tags=["${MODEL}","${DATASET}",nc,sheaf,exp3]