#!/bin/bash
#SBATCH -J gnn_baselines
#SBATCH --output=gnn_baselines/out/%A_%a.out
#SBATCH --error=gnn_baselines/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-119%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( GAT GCN HAN HGCN HGT RGCN )
DATASETS=( LastFM AmazonBooks )

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
srun python link_prediction.py trainer.strategy=dpp_find_unused_parameters trainer.devices=2 model="${MODEL}" dataset="${DATASET}" +tags=["${MODEL}","${DATASET}",lp,gnn,exp2]