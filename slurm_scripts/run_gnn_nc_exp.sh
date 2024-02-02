#!/bin/bash
#SBATCH -J gnn_nc_baselines
#SBATCH --output=gnn_nc_baselines/out/%A_%a.out
#SBATCH --error=gnn_nc_baselines/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-9%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( gat gcn han hgcn hgt rgcn )
DATASETS=( dblp acm imdb )

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
srun python node_classification.py model="${MODEL}" dataset="${DATASET}" +tags=["${MODEL}","${DATASET}",nc,gnn,exp1_1]