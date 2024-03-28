#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J sheafnc_baselines
#SBATCH --output=slurm_output/sheafnc_baselines/out/%A_%a.out
#SBATCH --error=slurm_output/sheafnc_baselines/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=3:00:00
#SBATCH -a 0-359
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=( diag_sheaf bundle_sheaf general_sheaf )
SHEAF_LEARNERS=( node_type_concat edge_type_concat node_type edge_type )
DATASETS=( dblp acm imdb )

N_TRIALS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODEL_PARAMS[@]}
N_SHEAF_LEARNERS=${#SHEAF_LEARNERS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
BLOCK_IDX=$((IDX / N_TRIALS))
MODEL_IDX=$((BLOCK_IDX / N_MODELS))
DATA_SHEAF_IDX=$((BLOCK_IDX % (N_DATASETS * N_SHEAF_LEARNERS)))
SHEAF_LEARNER_IDX=$((DATA_SHEAF_IDX / N_SHEAF_LEARNERS))
DATA_IDX=$((DATA_SHEAF_IDX % N_DATASETS))


MODEL=${MODEL_PARAMS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}
SHEAF_LEARNER=${SHEAF_LEARNERS[SHEAF_LEARNER_IDX]}

export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python sheaf_nc.py model="${MODEL}" dataset="${DATASET}" sheaf_learner="${SHEAF_LEARNER}" tags=["${MODEL}","${DATASET}",nc,sheaf,exp1_1]