#!/bin/bash
#SBATCH --job-name=rna_pref_opt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/rna_train_%j.out
#SBATCH --error=logs/rna_train_%j.err

# Create logs directory
mkdir -p logs

# Load environment
module load cuda/11.8
source activate grnade

# Change to project directory
cd /path/to/offline-dpo

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Config: $1"
echo "Loss Type: ${2:-auto}"
echo "Started at: $(date)"

# Execute training - supports both SimPO and DPO
if [ -n "$2" ]; then
    python -m dpo.train --config $1 --loss_type $2
else
    python -m dpo.train --config $1
fi

echo "Finished at: $(date)"