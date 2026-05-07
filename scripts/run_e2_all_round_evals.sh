#!/usr/bin/env bash
# Evaluate every round_N_best.pt from both E2 runs (15_isdpo_off, 16_isdpo_on).
# Used after Phase E2 finishes to populate the scMCC-vs-R + KL-vs-R curves.
#
# Usage:
#   bash scripts/run_e2_all_round_evals.sh <jobid_pool>...
#
# Example (with 4 freed jobids):
#   bash scripts/run_e2_all_round_evals.sh 4817732 4817733 4817734 4817509
#
# Distributes the 10 (5 rounds × 2 runs) evals across the supplied jobids in round-robin.

set -uo pipefail
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <jobid> [<jobid> ...]"
    exit 1
fi

ROOT=/mnt/rna01/smh/projects/ribopo
cd "$ROOT"

JOBIDS=("$@")
N_JOBS=${#JOBIDS[@]}

# Find latest run dirs for each tag
RUN_OFF=$(ls -dt runs/multiround/plan_a_15_isdpo_off_R5_* 2>/dev/null | head -1)
RUN_ON=$(ls -dt runs/multiround/plan_a_16_isdpo_on_R5_* 2>/dev/null | head -1)

if [[ -z "$RUN_OFF" || -z "$RUN_ON" ]]; then
    echo "ERROR: could not find E2 run dirs"
    echo "  off: $RUN_OFF"
    echo "  on:  $RUN_ON"
    exit 1
fi

echo "[e2-evals] OFF run: $RUN_OFF"
echo "[e2-evals] ON  run: $RUN_ON"
echo "[e2-evals] jobid pool: ${JOBIDS[*]} (n=$N_JOBS)"

i=0
for round in 1 2 3 4 5; do
    for run_label in off on; do
        if [[ "$run_label" == "off" ]]; then
            run_dir="$RUN_OFF"
        else
            run_dir="$RUN_ON"
        fi
        ckpt="${run_dir}/round_0${round}/checkpoints/round_${round}_best.pt"
        tag="e2_${run_label}_R${round}"
        if [[ ! -f "$ckpt" ]]; then
            echo "[skip] $tag: $ckpt does not exist yet"
            continue
        fi
        # Skip if eval already done
        if [[ -f "runs/phase2/eval/${tag}/eval_summary.json" ]]; then
            echo "[skip] $tag: eval_summary.json already exists"
            continue
        fi
        jobid=${JOBIDS[$((i % N_JOBS))]}
        log="runs/phase2/logs/${tag}.log"
        mkdir -p "runs/phase2/logs"
        echo "[fire] $tag → jobid=$jobid"
        setsid nohup bash scripts/run_eval_on_jobid.sh "$jobid" "$ckpt" "$tag" \
            > "$log" 2>&1 < /dev/null &
        disown
        sleep 5
        i=$((i + 1))
    done
done

echo "[e2-evals] launched $i evals; tail logs at runs/phase2/logs/e2_*.log"
