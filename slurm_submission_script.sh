#/bin/bash
#SBATCH -A COMPUTERLAB-GPU
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

python link_prediction.py