#!/usr/bin/env bash
# Run a multiround IS-DPO training on a specific srun allocation.
# Used for Phase E2 (and the smoke test 17_isdpo_smoke).
#
# Usage:
#   bash scripts/run_isdpo_experiment.sh <jobid> <experiment_yaml>
# Examples:
#   bash scripts/run_isdpo_experiment.sh 4790774 multiround/config/experiments/17_isdpo_smoke.yaml
#   bash scripts/run_isdpo_experiment.sh 4790775 multiround/config/experiments/15_isdpo_off_R5.yaml
#   bash scripts/run_isdpo_experiment.sh 4815802 multiround/config/experiments/16_isdpo_on_R5.yaml

set -uo pipefail
JOBID="${1:?usage: $0 <jobid> <config>}"
CFG="${2:?usage: $0 <jobid> <config>}"

ROOT=/mnt/rna01/smh/projects/ribopo

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  exit 1
fi

TAG=$(basename "$CFG" .yaml)
LOG="$ROOT/runs/multiround/logs/${TAG}.log"
mkdir -p "$ROOT/runs/multiround/logs"

echo "[isdpo] firing $TAG on jobid $JOBID → log $LOG"

exec srun --jobid="$JOBID" --overlap bash -c "
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval \"\$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)\"
mamba activate grnade
cd $ROOT
exec python -m multiround.train --config '$CFG' 2>&1 | tee '$LOG'
"
