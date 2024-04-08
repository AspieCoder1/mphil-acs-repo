#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J gnnlp_baselines
#SBATCH --output=slurm_output/gnn_link_baselines/out/%A_%a.out
#SBATCH --error=slurm_output/gnn_link_baselines/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-179%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( gat gcn han hgcn hgt rgcn )
DATASETS=( last_fm amazon_books movie_lens )

N_TRIALS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODEL_PARAMS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
N_RUN=$((IDX / N_TRIALS))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

MODEL=${MODEL_PARAMS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}

export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python link_prediction.py trainer.devices=1 model="${MODEL}" dataset="${DATASET}" +tags=["${MODEL}","${DATASET}",lp,gnn,exp2,recsys]