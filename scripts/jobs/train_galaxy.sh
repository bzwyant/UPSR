#!/bin/bash
#SBATCH --job-name=galaxy_mnist_processing
#SBATCH -A e32704                               # Account name
#SBATCH --partition=gengpu                      # GPU Partition
#SBATCH --nodes=1                               # Single machine
#SBATCH --ntasks-per-node=2                     # 2 processes (1 per GPU)
#SBATCH --gres=gpu:a100:2                       # 2 A100 GPU on same machine
#SBATCH --time=16:00:00                         # Max runtime
#SBATCH --mem=64G                               # Memory allocation
#SBATCH --cpus-per-task=4                       # 4 CPUs per GPU
#SBATCH --output=logs/train/UPSR_galaxy_%j.log  # Log file (SLURM_JOB_ID included)


# Activate virtual environment
module purge

eval "$(conda shell.bash hook)"
conda activate /home/tlf3755/.conda/envs/UPSR

# Ensure real-time logging
export PYTHONUNBUFFERED=1

# Set threading for optimal performance
export OMP_NUM_THREADS=1

# Run training script
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py -opt options/galaxy_UPSR.yml --launcher pytorch 