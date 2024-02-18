#!/bin/sh
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --gpu-bind=none

export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
srun python link_prediction.py