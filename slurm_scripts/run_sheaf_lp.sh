#!/bin/bash
#SBATCH -J sheaf_lp_profile
#SBATCH --output=sheaf_lp_profile/out/%A_%a.out
#SBATCH --error=sheaf_lp_profile/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=3:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

export WANDB_CACHE_DIR="~/rds/hpc-work/.wandb"
export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
sbatch run sheaf_lp.py