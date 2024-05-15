#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J dti_predictions
#SBATCH --output=slurm_output/dti_predictions/out/%A_%a.out
#SBATCH --error=slurm_output/dti_predictions/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -a 0-139%10
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODELS=( AllDeepSets AllSetsTransformer HCHA HGNN SheafHyperGNN SheafHyperGNN-TE SheafHyperGNN-ensemble )
DATASETS=( DeepDTNet KEGG )

N_SEEDS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODEL_PARAMS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
N_RUN=$(( IDX / N_SEEDS ))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

MODEL=${MODELS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}
SPLIT=$(( IDX % N_SEEDS ))


export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python run_dti.py dataset.split="${SPLIT}" experiment="dti_predictions/${MODEL}_${DATASET}"


