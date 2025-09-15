#!/bin/bash
# Submit all HPO jobs to SLURM

echo "Submitting HPO jobs..."

# Submit array job for all 6 configs
sbatch dpo/scripts/hpo_optuna.slurm

echo "All HPO jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/hpo_*.out"
echo "Results will be in: dpo/hpo/optuna_results/"