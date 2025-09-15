#!/bin/bash
# Script to run a single HPO trial for testing

# Check if config number is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_number (1-6)>"
    echo "  1-3: SimPO configs (A100 settings)"
    echo "  4-6: DPO configs (A40 settings)"
    exit 1
fi

CONFIG_NUM=$1

# Determine algorithm based on config number
if [ $CONFIG_NUM -le 3 ]; then
    ALGO="simpo"
    echo "Running SimPO HPO with config ${CONFIG_NUM}"
else
    ALGO="dpo"
    echo "Running DPO HPO with config ${CONFIG_NUM}"
fi

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/rna01/smh/projects/offline-dpo:$PYTHONPATH

# Create directories
mkdir -p dpo/hpo/optuna_results
mkdir -p logs

# Run HPO with fewer trials for testing
python -m dpo.hpo.optuna_search \
    --config "dpo/hpo/optuna_${CONFIG_NUM}.yaml" \
    --algo ${ALGO} \
    --trials 3 \
    --study_name "${ALGO}_hpo_test_${CONFIG_NUM}" \
    --output_dir "dpo/hpo/optuna_results" \
    --seed $((42 + CONFIG_NUM))

echo "Test HPO completed"