#!/usr/bin/env bash
# Phase-2 experimental sweep launcher.
#
# Usage:
#   bash scripts/run_phase2_sweep.sh thermo_surplus      # train DPO on thermo-surplus pairs
#   bash scripts/run_phase2_sweep.sh beta_sweep          # 5-config β-recovery curve (sequential)
#   bash scripts/run_phase2_sweep.sh beta <value>        # single β run, e.g. `bash ... beta 0.5`
#
# Assumes you are already on a GPU node (use `bash ribopo_v4/scripts/cluster_attach.sh <JOBID>`
# to attach to an existing allocation first). Logs go to runs/phase2/logs/.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source ~/.bashrc.bak.2026-0319-1628
eval "$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate grnade

mkdir -p runs/phase2/logs

EXP="${1:-help}"

run_one () {
  local cfg="$1"
  local tag="$2"
  local log="runs/phase2/logs/${tag}.log"
  echo "[launch] cfg=${cfg} tag=${tag} log=${log}"
  nohup python -m dpo.train --config "${cfg}" > "${log}" 2>&1 &
  echo "[launched] PID=$!"
}

case "$EXP" in
  thermo_surplus)
    run_one dpo/configs/experiments_phase2/thermo_surplus_m25.yaml thermo_surplus_m25
    ;;
  beta_sweep)
    for tag in 001 005 012 05 10; do
      run_one "dpo/configs/experiments_phase2/beta_${tag}.yaml" "beta_${tag}"
      sleep 5
    done
    ;;
  beta)
    val="${2:?usage: $0 beta <value>}"
    tag=$(echo "$val" | tr -d '.')
    run_one "dpo/configs/experiments_phase2/beta_${tag}.yaml" "beta_${tag}"
    ;;
  loss_ablation)
    for loss in ipo kto pareto_dpo; do
      run_one "dpo/configs/experiments_phase2/${loss}_b012.yaml" "${loss}_b012"
      sleep 5
    done
    ;;
  loss)
    name="${2:?usage: $0 loss <ipo|kto|pareto_dpo>}"
    run_one "dpo/configs/experiments_phase2/${name}_b012.yaml" "${name}_b012"
    ;;
  *)
    cat <<EOF
Phase-2 experiment launcher.

Available experiments:
  thermo_surplus    DPO on GC-controlled pair set (1 run)
  beta_sweep        β ∈ {0.01, 0.05, 0.12, 0.5, 1.0} (5 runs, sequential)
  beta <val>        single β run, e.g. \`beta 0.5\`
  loss_ablation     IPO, KTO, Pareto-DPO at β=0.12 (3 runs, sequential)
  loss <name>       single loss run, e.g. \`loss ipo\`

Examples:
  bash scripts/run_phase2_sweep.sh thermo_surplus
  bash scripts/run_phase2_sweep.sh beta 0.5
  bash scripts/run_phase2_sweep.sh beta_sweep
  bash scripts/run_phase2_sweep.sh loss_ablation
EOF
    exit 1
    ;;
esac
