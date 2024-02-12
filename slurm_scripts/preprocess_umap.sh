#!/bin/bash
#SBATCH -J preprocess_umap
#SBATCH --output=preprocess_umap/out/%A_%a.out
#SBATCH --error=preprocess_umap/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0:2%3
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none

DATASETS=( dblp acm imdb )

IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[IDX]}

module load gcc/11
source ~/venv/bin/activate
srun python preprocess_umap.py model="diag_sheaf" dataset="$DATASET"