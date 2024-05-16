#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J sheaf_link_lastfm
#SBATCH --output=slurm_output/sheaf_link_baselines/out/%A_%a.out
#SBATCH --error=slurm_output/sheaf_link_baselines/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=01:00:00
#SBATCH -a 0-29
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

MODEL_PARAMS=(diag_sheaf bundle_sheaf general_sheaf)
SHEAF_LEARNERS=(type_ensemble type_ensemble node_type_concat)
DATASETS=(last_fm)

N_TRIALS=10

IDX=${SLURM_ARRAY_TASK_ID}
BLOCK_IDX=$((IDX / N_TRIALS))

MODEL=${MODEL_PARAMS[BLOCK_IDX]}
SHEAF_LEARNER=${SHEAF_LEARNERS[BLOCK_IDX]}

export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun  python run_sheaf_lp.py experiment="sheaf_lp/${MODEL}_last_fm" sheaf_learner="${SHEAF_LEARNER}"