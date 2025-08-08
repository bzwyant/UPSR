#!/bin/bash
#SBATCH --job-name=galaxy_mnist_processing
#SBATCH --partition=short
#SBATCH -A e32704   # Account name
#SBATCH -p gengpu   # GPU Partition
#SBATCH --gres=gpu:a100:1   # Request 1 A100 GPU
#SBATCH -N 1    # Number of nodes
#SBATCH -n 1    # Number of tasks
#SBATCH -t 1:00:00  # Max runtime
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=crop_%j.log   # Log file (SLURM_JOB_ID included)


module load mamba

# Ensure real-time logging
export PYTHONUNBUFFERED=1

# Activate virtual environment
mamba activate UPSR

# get path to root folder
PATH_TO_DATASET="basicsr/datasets/galaxy_mnist"

# Run image processing script and log output
cd $PATH_TO_ROOT &&
time python scripts/data_preparation/crop_galaxy_mnist.py \
    --input_dir "${PATH_TO_DATASET}/train" \
    --output-128 "${PATH_TO_DATASET}/train/gt" \
    --output-64 "${PATH_TO_DATASET}/train/lq" \
    --batch-size 64
