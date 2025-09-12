#!/bin/bash
#SBATCH --job-name=ribopo-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/home/m/minghsun/projects/ribopo/dpo/slurm_logs/eval_%j.log # Path is relative to the project root

# --- Best Practices ---
set -e

# --- Job Diagnostics ---
echo "Starting evaluation on $(hostname)"
echo "Working directory: $(pwd)"
echo "Config file: $1"
echo "Additional args: ${@:2}"

# --- Environment and Execution ---
source /home/m/minghsun/miniconda3/bin/activate rna

# Execute the evaluation script
exec python -m dpo.bench.eval_full --config $1 ${@:2}