#!/bin/bash
#SBATCH --job-name=ribopo-train
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1                 # Be specific: request one A100 80GB GPU. Change if needed.
#SBATCH --cpus-per-task=24                # Request 8 CPUs if you set num_workers > 0 in your yaml.
#SBATCH --time=3-00:00:00                # Request 2 days of runtime
#SBATCH --output=/home/m/minghsun/projects/ribopo/dpo/slurm_logs/train_%j.log # IMPORTANT: Path is relative to the project root

# --- Best Practices ---
# Exit immediately if a command fails
set -e

# --- Job Diagnostics ---
echo "Starting job on $(hostname)"
echo "Working directory: $(pwd)" # This will confirm you are in the 'offline-dpo' root
echo "Using config file: $1"

# --- Environment and Execution ---
# Activate your Conda environment robustly
source /home/m/minghsun/miniconda3/bin/activate rna

# Execute the training script.
# The '-m dpo.train' module lookup works perfectly because the job runs from the project root.
exec python -m dpo.train --config $1