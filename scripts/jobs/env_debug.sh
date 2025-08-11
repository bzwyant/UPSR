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
#SBATCH --output=logs/env_debug_%j.log   # Log file (SLURM_JOB_ID included)

module purge
module load mamba/23.1.0

# Initialize conda/mamba for batch environment
eval "$(conda shell.bash hook)"

# Activate environment
mamba activate UPSR

# Verify environment is working (optional debug)
echo "Python path: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run your script
python /gpfs/projects/e32704/ben/UPSR/scripts/data_preparation/crop_galaxy_mnist.py