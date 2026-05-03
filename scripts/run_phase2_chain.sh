#!/usr/bin/env bash
# Sequential phase-2 experiment chain.
#
# Waits for an optionally-passed PID to finish first, then runs each training
# in turn (waiting for each to complete before launching the next). This is the
# safe option when the GPU can only accommodate ~1 training at a time.
#
# Usage:
#   nohup bash scripts/run_phase2_chain.sh [WAIT_PID] > runs/phase2/logs/chain.log 2>&1 &
#
# Examples:
#   # wait for the running thermo-surplus run, then start the β-sweep + loss ablation
#   nohup bash scripts/run_phase2_chain.sh 473162 > runs/phase2/logs/chain.log 2>&1 &

set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# bashrc has an unbound-variable issue under `set -u`, so source carefully.
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval "$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate grnade
set -u

mkdir -p runs/phase2/logs

WAIT_PID="${1:-}"
if [[ -n "${WAIT_PID}" ]]; then
  echo "[chain] waiting for PID ${WAIT_PID} to exit..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 30; done
  echo "[chain] PID ${WAIT_PID} exited; chain begins at $(date)"
fi

# Order of experiments to run sequentially. β=0.12, 0.5, 1.0 are existing in the
# original Table 4 of the paper, so we prioritise the 2 *new* β values plus the
# 3 alternative-loss runs. If you also want fresh re-runs of β=0.12/0.5/1.0 for
# consistency, append them below.
EXPS=(
  "beta_001        dpo/configs/experiments_phase2/beta_001.yaml"
  "beta_005        dpo/configs/experiments_phase2/beta_005.yaml"
  "ipo_b012        dpo/configs/experiments_phase2/ipo_b012.yaml"
  "kto_b012        dpo/configs/experiments_phase2/kto_b012.yaml"
  "pareto_dpo_b012 dpo/configs/experiments_phase2/pareto_dpo_b012.yaml"
  # uncomment below to additionally re-run the standard β values for consistency
  # "beta_012        dpo/configs/experiments_phase2/beta_012.yaml"
  # "beta_05         dpo/configs/experiments_phase2/beta_05.yaml"
  # "beta_10         dpo/configs/experiments_phase2/beta_10.yaml"
)

for entry in "${EXPS[@]}"; do
  read -r tag cfg <<<"${entry}"
  log="runs/phase2/logs/${tag}.log"

  # Idempotency: skip if a best.pt for this experiment already exists
  # (e.g., dispatched to A100 nodes in parallel and already finished).
  save_dir=$(python -c "
import yaml
with open('${cfg}') as f: c = yaml.safe_load(f)
print(c['paths']['save_dir'])
" 2>/dev/null)
  run_name=$(python -c "
import yaml
with open('${cfg}') as f: c = yaml.safe_load(f)
print(c['wandb']['run_name'])
" 2>/dev/null)
  best_pt="${save_dir}/${run_name}/best.pt"
  if [[ -n "${save_dir}" && -e "${best_pt}" ]]; then
    echo "[chain] SKIP ${tag} (best.pt already exists at ${best_pt})"
    continue
  fi

  echo "[chain] === starting ${tag} at $(date) ==="
  python -m dpo.train --config "${cfg}" > "${log}" 2>&1
  rc=$?
  echo "[chain] === finished ${tag} at $(date), exit=${rc} ==="
  if [[ ${rc} -ne 0 ]]; then
    echo "[chain] WARNING: ${tag} exited non-zero (${rc}); continuing"
  fi
done

echo "[chain] full chain complete at $(date)"
