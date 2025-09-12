#!/bin/bash
#SBATCH --job-name=ribopo-sweep
#SBATCH --partition=gpu-long
#SBATCH --gpus=1                         # Request one generic GPU.
#SBATCH --constraint="xgpe|xgpd|xgpf|xgpg" # Target less-contended GPUs (Titans, T4s, A100-40GB)
#SBATCH --cpus-per-task=20               # Request enough CPUs for your num_workers
#SBATCH --time=1-20:00:00
#SBATCH --array=0-7                      # Creates an array of 8 jobs for your 8 experiments
#SBATCH --output=dpo/slurm_logs/sweep_%A_task_%a.log # %A=ArrayID, %a=TaskID

# --- Best Practices ---
set -e

# --- Define the list of config files for the sweep ---
CONFIG_FILES=(
  dpo/configs/experiments/exp005_simpo_b2.0_g1.0_l0.05.yaml
  dpo/configs/experiments/exp006_simpo_b2.0_g1.2_l0.05.yaml
  dpo/configs/experiments/exp007_simpo_b2.0_g1.4_l0.05.yaml
  dpo/configs/experiments/exp008_simpo_b2.0_g1.6_l0.05.yaml
  dpo/configs/experiments/exp009_simpo_b2.5_g0.8_l0.05.yaml
  dpo/configs/experiments/exp010_simpo_b2.3_g1.0_l0.10.yaml
  dpo/configs/experiments/exp011_simpo_b2.0_g0.8_l0.05_lr1e-5.yaml
  dpo/configs/experiments/exp012_simpo_b2.0_g0.8_l0.10_lr1e-6.yaml
)

# --- Select the config for the current job ---
CURRENT_CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

# --- Dynamic Batch Size Logic ---
# Detect the node type and set an appropriate batch size
# The SLURMD_NODENAME variable is automatically set by Slurm
case $SLURMD_NODENAME in
  xgpg*) # A100-40GB nodes
    BATCH_SIZE=32
    ;;
  xgpe*) # Titan RTX (24GB) nodes
    BATCH_SIZE=16
    ;;
  xgpd*|xgpf*) # Titan V (12GB) or T4 (16GB) nodes
    BATCH_SIZE=8
    ;;
  *) # Default for any other node
    BATCH_SIZE=4
    ;;
esac

# --- Job Diagnostics ---
echo "Starting array job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID on $(hostname) ($SLURMD_NODENAME)"
echo "Using config file: $CURRENT_CONFIG"
echo "Dynamically set Batch Size: $BATCH_SIZE"

# --- Environment and Execution ---
source /home/m/minghsun/miniconda3/bin/activate rna
exec python -m dpo.train --config $CURRENT_CONFIG --batch_size $BATCH_SIZE