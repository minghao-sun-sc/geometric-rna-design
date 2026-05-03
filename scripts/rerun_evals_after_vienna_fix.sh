#!/usr/bin/env bash
# Re-run the 5 evals that ran with the BUGGY Vienna aggregator (np.mean instead of np.nanmean).
# Fires each on a different live srun jobid in parallel via --overlap.
#
# Affected evals (need rerun for clean MFE/ED):
#   - thermo_surplus_m25 (already-completed; designs not persisted)
#   - ipo_b012 / kto_b012 / pareto_dpo_b012 (in progress with old code, designs not persisted)
#   - beta_005 (in progress with old code, designs not persisted)
#
# UNAFFECTED (already use the fixed code; no rerun needed):
#   - beta_001 / pareto_stage2_b012 (their evals fire after the trainings finish, AFTER fix was deployed)
#
# Usage:
#   1. Wait for `runs/phase2/eval/<tag>/eval_summary.json` to exist for ALL 7 tags
#      (i.e. the broken-code chain settled, including beta_001 / pareto_stage2 evals).
#   2. squeue -u smh   # find live jobids; need at least 5 with > 4h time-left.
#   3. Edit the JOBIDS array below.
#   4. bash scripts/rerun_evals_after_vienna_fix.sh
#
# The eval results land in runs/phase2/eval/<tag>/eval_summary.json (overwriting
# the buggy ones). Old results are auto-archived to <tag>/eval_summary.bug.json.

set -eo pipefail
ROOT=/mnt/rna01/smh/projects/ribopo
cd "$ROOT"
mkdir -p runs/phase2/logs

# === EDIT ME: assign one live jobid to each tag ===
declare -A JOBIDS=(
  [thermo_surplus_m25]=""
  [ipo_b012]=""
  [kto_b012]=""
  [pareto_dpo_b012]=""
  [beta_005]=""
)

declare -A CKPTS=(
  [thermo_surplus_m25]=runs/phase2/thermo_surplus/m25/thermo_surplus_m25_b0.12/best.pt
  [ipo_b012]=runs/phase2/loss_ablation/ipo/ipo_b0.12/best.pt
  [kto_b012]=runs/phase2/loss_ablation/kto/kto_b0.12/best.pt
  [pareto_dpo_b012]=runs/phase2/loss_ablation/pareto_dpo/pareto_dpo_b0.12/best.pt
  [beta_005]=runs/phase2/beta_sweep/b0.05/beta_sweep_b0.05/best.pt
)

# Validate
missing=0
for tag in "${!JOBIDS[@]}"; do
  if [[ -z "${JOBIDS[$tag]}" ]]; then
    echo "ERROR: JOBIDS[$tag] is empty — please assign a live srun jobid."
    missing=1
  fi
  if [[ ! -e "${CKPTS[$tag]}" ]]; then
    echo "ERROR: ckpt missing for $tag: ${CKPTS[$tag]}"
    missing=1
  fi
done
[[ $missing -eq 1 ]] && { echo "Aborting."; exit 1; }

# Archive old summaries before overwriting
for tag in "${!JOBIDS[@]}"; do
  old="runs/phase2/eval/${tag}/eval_summary.json"
  if [[ -e "$old" ]]; then
    cp "$old" "runs/phase2/eval/${tag}/eval_summary.bug.json"
    echo "[archive] $old -> ${old%.json}.bug.json"
  fi
done

# Fire all 5 in parallel
for tag in "${!JOBIDS[@]}"; do
  jobid="${JOBIDS[$tag]}"
  ckpt="${CKPTS[$tag]}"
  log="runs/phase2/logs/${tag}_eval_v2.log"
  echo "[launch] $tag on jobid=$jobid log=$log"
  setsid nohup bash "$ROOT/scripts/run_eval_on_jobid.sh" "$jobid" "$ckpt" "$tag" \
    > "$log" 2>&1 < /dev/null &
  disown $! 2>/dev/null
done

echo
echo "All 5 reruns dispatched. Tail logs:"
echo "  for t in thermo_surplus_m25 ipo_b012 kto_b012 pareto_dpo_b012 beta_005; do"
echo "    echo \"=== \$t ===\"; tail -3 runs/phase2/logs/\${t}_eval_v2.log;"
echo "  done"
echo
echo "When all 5 eval_summary.json files are refreshed, run:"
echo "  python scripts/aggregate_phase2_results.py"
