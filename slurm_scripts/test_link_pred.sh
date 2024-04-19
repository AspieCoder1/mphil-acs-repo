#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J gnnlp_debug
#SBATCH --output=slurm_output/gnnlp_debug/%j.out
#SBATCH --error=slurm_output/gnnlp_debug/%j.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=1:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1


export WANDB_CACHE_DIR=".wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python link_prediction.py trainer.devices=1 trainer.max_epochs=5 +tags="[lp,gnn,exp2,recsys,debug]"
