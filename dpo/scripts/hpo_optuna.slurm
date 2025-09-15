#!/bin/bash
#SBATCH --job-name=rna_hpo
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --array=1-6
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# GPU allocation based on array task ID
# 1-3: SimPO on A100 GPUs
# 4-6: DPO on A40 GPUs
if [ $SLURM_ARRAY_TASK_ID -le 3 ]; then
    #SBATCH --gres=gpu:a100:1
    #SBATCH --partition=a100
    GPU_TYPE="A100"
    ALGO="simpo"
else
    #SBATCH --gres=gpu:a40:1
    #SBATCH --partition=a40
    GPU_TYPE="A40"
    ALGO="dpo"
fi

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/rna01/smh/projects/offline-dpo:$PYTHONPATH

# Activate conda environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rna-dpo

# Create logs directory
mkdir -p logs
mkdir -p dpo/hpo/optuna_results

# Determine config file
CONFIG_FILE="dpo/hpo/optuna_${SLURM_ARRAY_TASK_ID}.yaml"

# Set number of trials per job
TRIALS_PER_JOB=20

# Run HPO
echo "Starting HPO for ${ALGO} on ${GPU_TYPE} GPU"
echo "Config: ${CONFIG_FILE}"
echo "Trials: ${TRIALS_PER_JOB}"

python -m dpo.hpo.optuna_search \
    --config ${CONFIG_FILE} \
    --algo ${ALGO} \
    --trials ${TRIALS_PER_JOB} \
    --study_name "${ALGO}_hpo_region${SLURM_ARRAY_TASK_ID}" \
    --output_dir "dpo/hpo/optuna_results" \
    --seed $((42 + SLURM_ARRAY_TASK_ID))

echo "HPO completed for task ${SLURM_ARRAY_TASK_ID}"