#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
#

#SBATCH -J process_tsne
#SBATCH --output=process_tsne/out/%A.out
#SBATCH --error=process_tsne/err/%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=2:00:00
#SBATCH -p cclake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256000

export NUMBA_DISABLE_JIT=1
source ~/venv/bin/activate
srun python process_tsne.py