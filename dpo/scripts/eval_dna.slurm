#!/bin/bash
#SBATCH --job-name=ribopoevaldna       # <-- CHANGE: A more specific name for this cluster
#SBATCH --partition=GPUA40               # <-- CHANGE: Target the A40 partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8                # NOTE: A40 nodes might have many CPUs, so 8 is a safe number.
#SBATCH --time=05:00:00                  # NOTE: The GPUA40 partition allows up to 4 days, so 5 hours is fine.
#SBATCH --output=/mnt/rna01/smh/projects/offline-dpo/dpo/slurm_logs/eval_%j.log 

# --- Best Practices ---
set -e

# --- Job Diagnostics ---
echo "Starting evaluation on $(hostname)"
echo "Working directory: $(pwd)"
echo "Config file: $1"

# --- Create output directories ---
mkdir -p dpo/eval_results
echo "Evaluation results will be saved to: dpo/eval_results/"

# --- Environment and Execution ---
# Full, robust path to the activate script for the DNA cluster's miniforge installation.
source /mnt/dna01/library-seq/luca/miniforge3/bin/activate grnade

# Execute the evaluation script
echo "Running evaluation..."
exec python -m dpo.bench.eval_full --config $1 ${@:2}