#!/bin/bash
#SBATCH -J sheafnc_tsne
#SBATCH --output=sheafnc_tsne/out/%A.out
#SBATCH --error=sheafnc_tsne/err/%A.err
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --time=3:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none

module load gcc
export I_MPI_CC=gcc
export I_MPI_CXX=g++
export I_MPI_F90=gfortran
source ~/venv/bin/activate
srun python tsne_nc.py model="diag_sheaf" dataset="dblp"