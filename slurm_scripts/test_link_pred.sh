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

srun python link_prediction.py trainer.devices=1 +tags=["${MODEL}","${DATASET}",lp,gnn,exp2,recsys,debug]
