#!/bin/bash

#SBATCH -J SheafHyperGNN_DeepDTNet
#SBATCH --output=slurm_output/sweeps/SheafHyperGNN_DeepDTNet/out/%A_%a.out
#SBATCH --error=slurm_output/sweeps/SheafHyperGNN_DeepDTNet/err/%A_%a.err
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH --time=00:15:00
#SBATCH -a 0-60
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL

export WANDB_API_KEY="cc080145b244f97b7db093ba0e3de5088e7ee7aa"
source ~/venv/bin/activate
srun python -m wandb agent --count 1 "acs-thesis-lb2027/gnn-baselines/gcb9z67l"