#!/usr/bin/env bash
# Dispatch a phase-2 training to an A100 / A40 node via srun --overlap.
# Uses an existing user allocation; assumes the target GPU within that
# allocation is idle (the user's babysitter jobs typically reserve GPU
# without actively using it).
#
# Usage:
#   bash scripts/dispatch_a100.sh <JOBID> <TAG> <CONFIG>
# Example:
#   bash scripts/dispatch_a100.sh 4790775 beta_001 dpo/configs/experiments_phase2/beta_001.yaml

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBID="${1:?usage: $0 <JOBID> <TAG> <CONFIG>}"
TAG="${2:?usage: $0 <JOBID> <TAG> <CONFIG>}"
CFG="${3:?usage: $0 <JOBID> <TAG> <CONFIG>}"

LOG="runs/phase2/logs/${TAG}.log"
PID_FILE="runs/phase2/logs/${TAG}.pid"
mkdir -p runs/phase2/logs

# Compose a self-contained command that activates the env and launches training.
# We pipe it through bash so the heredoc-like contents survive srun.
CMD=$(cat <<EOF
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval "\$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate grnade
cd ${ROOT}
echo "[dispatch:${TAG}] host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
exec python -m dpo.train --config ${CFG}
EOF
)

# Launch in background via nohup so this script doesn't block.
nohup srun --jobid="${JOBID}" --overlap bash -c "${CMD}" > "${LOG}" 2>&1 &
SRUN_PID=$!
echo "${SRUN_PID}" > "${PID_FILE}"
echo "[dispatch:${TAG}] launched srun PID=${SRUN_PID} on jobid=${JOBID}, log=${LOG}, pidfile=${PID_FILE}"
