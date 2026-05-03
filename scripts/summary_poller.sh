#!/usr/bin/env bash
# Independent summary poller — polls for eval_summary.json appearance for each
# phase-2 training tag and emits a clean event line when each lands. Once all
# expected files exist, runs the aggregator and exits with ALL_DONE.
#
# This is a fallback for when the auto_eval_watcher's eval-timeout fires
# prematurely; the actual evals keep running and eventually write the file.

set +u                # bashrc compatibility
set -o pipefail

ROOT=/mnt/rna01/smh/projects/ribopo
cd "$ROOT"

source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval "$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate grnade

TAGS=(
  thermo_surplus_m25
  beta_001
  beta_005
  ipo_b012
  kto_b012
  pareto_dpo_b012
  pareto_stage2_b012
)

declare -A SEEN
SEEN_COUNT=0
TARGET=${#TAGS[@]}

ts() { date '+%H:%M:%S'; }

while true; do
  for tag in "${TAGS[@]}"; do
    summary="runs/phase2/eval/${tag}/eval_summary.json"
    if [[ -e "$summary" && -z "${SEEN[$tag]:-}" ]]; then
      SEEN[$tag]=1
      SEEN_COUNT=$((SEEN_COUNT + 1))
      # Inline a compact metrics summary
      metrics=$(python -c "
import json
d = json.load(open('${summary}'))
keys = ['recovery','scMCC','MFE','RMSD','TM','pLDDT','diversity','GC','vienna_pS0','vienna_Tm','inf_nwc']
parts = []
for k in keys:
    v = d.get(k)
    if v is None: continue
    try: parts.append(f'{k}={float(v):.3f}')
    except: pass
print(' '.join(parts))
" 2>/dev/null)
      echo "[$(ts)] DONE_EVAL ($SEEN_COUNT/$TARGET) tag=$tag $metrics"
    fi
  done

  if (( SEEN_COUNT >= TARGET )); then
    echo "[$(ts)] ALL_DONE all $TARGET eval_summary.json files present"
    echo "[$(ts)] running aggregator..."
    python scripts/aggregate_phase2_results.py 2>&1 | tail -20
    echo "[$(ts)] aggregator done"
    break
  fi

  sleep 60
done
