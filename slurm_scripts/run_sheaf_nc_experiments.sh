#!/bin/bash
#SBATCH -J sheafnc_baselines
#SBATCH --output=sheafnc_baselines/%A_%a.out
#SBATCH --error=sheafnc_baselines/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=10:00:00
#SBATCH -a 0-89%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( diag_sheaf bundle_sheaf general_sheaf )
DATASETS=( dblp acm imdb )

NUM_TRIALS=10
NUM_DATASETS=${#DATASETS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$(( IDX % (NUM_TRIALS * NUM_DATASETS) ))
DATA_IDX=$(( IDX % (NUM_TRIALS) ))

MODEL=${MODEL_PARAMS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}

export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python sheaf_nc.py model="${MODEL}" dataset="${DATASET}" +tags=exp1