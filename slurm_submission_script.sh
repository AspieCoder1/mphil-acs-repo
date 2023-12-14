#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

srun python link_prediction.py