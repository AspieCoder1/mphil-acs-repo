#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J preprocess_umap
#SBATCH --output=preprocess_umap/out/%A_%a.out
#SBATCH --error=preprocess_umap/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=3:00:00
#SBATCH -a 0-8
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODELS=( diag_sheaf bundle_sheaf general_sheaf)
DATASETS=( dblp acm imdb )

IDX=${SLURM_ARRAY_TASK_ID}

N_MODELS=${#MODELS[@]}
MODEL_IDX=$(( IDX / N_MODELS ))
DATA_IDX=$(( IDX % N_MODELS ))
DATASET=${DATASETS[DATA_IDX]}
MODEL=${MODELS[MODEL_IDX]}

module load gcc/11
source ~/venv/bin/activate
srun python preprocess_umap.py model="$MODEL" dataset="$DATASET"