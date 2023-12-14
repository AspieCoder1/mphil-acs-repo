#/bin/bash
#SBATCH --account COMPUTERLAB-GPU
#SBATCH --pascal pascal
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

python link_prediction.py