#!/bin/bash
#SBATCH --job-name=galaxy_mnist_processing
#SBATCH --partition=short
#SBATCH -A e32704   # Account name
#SBATCH -p gengpu   # GPU Partition
#SBATCH --gres=gpu:a100:1   # Request 1 A100 GPU
#SBATCH -N 1    # Number of nodes
#SBATCH -n 1    # Number of tasks
#SBATCH -t 16:00:00  # Max runtime
#SBATCH --mem=32G   # Memory allocation
#SBATCH --output=logs/train_UPSR_%j.log   # Log file (SLURM_JOB_ID included)

module purge

eval "$(conda shell.bash hook)"
conda activate /home/tlf3755/.conda/envs/UPSR

# Ensure real-time logging
export PYTHONUNBUFFERED=1

 python train.py -opt options/galaxy_UPSR.yml