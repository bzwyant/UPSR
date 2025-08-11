#!/bin/bash
#SBATCH --job-name=env_debug
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

# Initialize conda/mamba for batch environment
eval "$(conda shell.bash hook)"

# Activate environment
conda activate /home/tlf3755/.conda/envs/UPSR

# Verify environment is working (optional debug)
echo "Python path: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
