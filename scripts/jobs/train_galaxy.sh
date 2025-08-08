#!/bin/bash
#SBATCH -A e32704
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100
#SBATCH -N 1  # Number of nodes
#SBATCH -n 1  # Number of tasks
#SBATCH -t 8:00:00
#SBATCH --mem=16G

module purge
module load mamba
