#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

#SBATCH -J PubMedNC_Sweep
#SBATCH --output=slurm_output/sweeps/PubMedNC/out/%A_%a.out
#SBATCH --error=slurm_output/sweeps/PubMedNC/err/%A_%a.err
#SBATCH -A plio-sl2-gpu
#SBATCH --time=00:15:00
#SBATCH -a 0-60
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python -m wandb agent --count 1 "acs-thesis-lb2027/hetero-sheaf-paper/6760vano"