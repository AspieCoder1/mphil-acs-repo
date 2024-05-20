#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J dti_hyperedge_feat_ablation
#SBATCH --output=slurm_output/dti_hyperedge_feat_ablation/out/%A_%a.out
#SBATCH --error=slurm_output/dti_hyperedge_feat_ablation/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-79%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

N_TRIALS=10
DATASETS=( DeepDTNet KEGG )
FEAT_TYPES=( var1 var2 var3 cp_decomp )

N_DATASETS=${#DATASETS[@]}
N_FEATS=${#FEAT_TYPES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
N_RUN=$(( IDX / N_SEEDS ))
FEAT_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

FEAT_TYPE=${FEAT_TYPES[FEAT_IDX]}
DATASET=${DATASETS[DATA_IDX]}
SPLIT=$(( IDX % N_TRIALS ))

export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python run_dti.py dataset.split="${SPLIT}" experiment="dti_predictions/SheafHyperGNN-TE_${DATASET}" model.he_feat_type="${FEAT_TYPE}" tags="[exp7]"
