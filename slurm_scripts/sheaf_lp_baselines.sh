#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J sheaf_link_baselines
#SBATCH --output=slurm_output/sheaf_link_baselines/out/%a.out
#SBATCH --error=slurm_output/sheaf_link_baselines/err/%a.err
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH --time=01:00:00
#SBATCH -a 0-19
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( general_sheaf )
DATASETS=( last_fm amazon_books )

N_TRIALS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODEL_PARAMS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
N_RUN=$((IDX / N_TRIALS))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

MODEL=${MODEL_PARAMS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}


export WANDB_CACHE_DIR="~/rds/hpc-work/.wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun  python sheaf_lp.py dataset="${DATASET}" model="${MODEL}" +tags=["${MODEL}","${DATASET}",lp,sheaf,exp2]