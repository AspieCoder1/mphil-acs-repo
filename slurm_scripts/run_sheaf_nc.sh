#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J sheafnc_baselines
#SBATCH --output=slurm_output/sheafnc_baselines/out/%A_%a.out
#SBATCH --error=slurm_output/sheafnc_baselines/err/%A_%a.err
#SBATCH -A plio-sl2-gpu
#SBATCH --time=1:00:00
#SBATCH -a 0-19
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL


DATASETS=(dblp acm imdb pubmed_nc)

N_TRIALS=5
IDX=${SLURM_ARRAY_TASK_ID}
DATA_IDX=$((IDX / N_TRIALS))

DATASET=${DATASETS[DATA_IDX]}

export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python run_sheaf_nc.py experiment="sheaf_nc/${DATASET}"