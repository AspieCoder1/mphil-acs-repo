#!/bin/bash
#SBATCH -J process_tsne
#SBATCH --output=process_tsne/out/%A.out
#SBATCH --error=process_tsne/err/%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=10:00:00
#SBATCH -p cclake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000

source ~/venv/bin/activate
srun python process_tsne.py