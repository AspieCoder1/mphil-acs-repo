#!/bin/sh
#
# Copyright (c) 2024. Luke Braithwaite
#

#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none

export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
srun python -m wandb agent "acs-thesis-lb2027/gnn-baselines/decvy5v9"