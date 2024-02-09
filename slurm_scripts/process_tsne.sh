#!/bin/bash
#SBATCH -J process_tsne
#SBATCH --output=sheafnc_tsne/out/%A.out
#SBATCH --error=sheafnc_tsne/err/%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=10:00:00
#SBATCH -p cclake-himem
#SBATCH --nodes 1
#SBATCH --ntasks=56

module load gcc/11
source ~/venv/bin/activate
srun python process_tsne.py