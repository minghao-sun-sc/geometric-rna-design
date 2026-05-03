#!/usr/bin/env bash
# Run SSTT eval for a checkpoint on a specific srun allocation.
# Used by the auto-eval watcher: when a training finishes on jobid X, we
# free up the GPU on X and immediately launch eval on the SAME jobid via
# srun --overlap so the eval uses the just-freed GPU.
#
# Usage: scripts/run_eval_on_jobid.sh <jobid> <ckpt_path> <tag>
set -uo pipefail
JOBID="${1:?usage: $0 <jobid> <ckpt> <tag>}"
CKPT="${2:?usage: $0 <jobid> <ckpt> <tag>}"
TAG="${3:?usage: $0 <jobid> <ckpt> <tag>}"

ROOT=/mnt/rna01/smh/projects/ribopo

exec srun --jobid="$JOBID" --overlap bash -c "
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval \"\$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)\"
mamba activate grnade
cd $ROOT
exec bash scripts/eval_phase2_checkpoint.sh '$CKPT' '$TAG'
"
