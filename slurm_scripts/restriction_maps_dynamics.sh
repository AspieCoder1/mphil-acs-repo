#!/bin/bash
#SBATCH -J preprocess_umap
#SBATCH --output=slurm_output/restricton_map_dynamics/out/%A_%a.out
#SBATCH --error=slurm_output/restricton_map_dynamics/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=3:00:00
#SBATCH -a 0-8
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODELS=( diag_sheaf bundle_sheaf general_sheaf )
DATASETS=( dblp acm imdb )

IDX=${SLURM_ARRAY_TASK_ID}

N_MODELS=${#MODELS[@]}
MODEL_IDX=$(( IDX / N_MODELS ))
DATA_IDX=$(( IDX % N_MODELS ))
DATASET=${DATASETS[DATA_IDX]}
MODEL=${MODELS[MODEL_IDX]}

export WANDB_CACHE_DIR="~/rds/hpc-work/.wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python sheaf_nc.py model="${MODEL}" dataset="${DATASET}" tags=["${MODEL}","${DATASET}",nc,sheaf,exp3]