#!/usr/bin/env bash
# Auto-eval watcher.
#
# Polls each training (PID + expected best.pt), and when a training exits with
# its best.pt present, launches SSTT eval on the same srun jobid via --overlap.
# Emits clear EVENT lines on stdout for the Monitor tool to capture.
#
# Each entry:  tag | jobid | pid_or_pidfile | best_pt
# - jobid is the srun allocation that hosted the training (eval re-uses it)
# - pid_or_pidfile is either a numeric PID (thermo-surplus, launched directly)
#   or a path to a PID file written by scripts/dispatch_a100.sh
# - best_pt is the expected path to the final best checkpoint
#
# Eval timeout: 60 minutes per eval (else marked eval_failed and we move on).

set -uo pipefail
ROOT=/mnt/rna01/smh/projects/ribopo
cd "$ROOT"
mkdir -p runs/phase2/logs runs/phase2/eval

ENTRIES=(
  "thermo_surplus_m25|4792654|473162|runs/phase2/thermo_surplus/m25/thermo_surplus_m25_b0.12/best.pt"
  "beta_001|4790775|runs/phase2/logs/beta_001.pid|runs/phase2/beta_sweep/b0.01/beta_sweep_b0.01/best.pt"
  "beta_005|4722450|runs/phase2/logs/beta_005.pid|runs/phase2/beta_sweep/b0.05/beta_sweep_b0.05/best.pt"
  "ipo_b012|4790774|runs/phase2/logs/ipo_b012.pid|runs/phase2/loss_ablation/ipo/ipo_b0.12/best.pt"
  "kto_b012|4792653|runs/phase2/logs/kto_b012.pid|runs/phase2/loss_ablation/kto/kto_b0.12/best.pt"
  "pareto_dpo_b012|4790772|runs/phase2/logs/pareto_dpo_b012.pid|runs/phase2/loss_ablation/pareto_dpo/pareto_dpo_b0.12/best.pt"
  "pareto_stage2_b012|4790775|runs/phase2/logs/pareto_stage2_b012.pid|runs/phase2/pareto_stage2/pareto_stage2_b0.12/best.pt"
)

declare -A STATE         # tag -> running | evaling | done | failed | eval_failed
declare -A EVAL_START    # tag -> unix epoch when eval started
declare -A EVAL_PID      # tag -> background PID of eval shell

EVAL_TIMEOUT_SEC=28800   # 8 hours per eval before giving up (full SSTT can take 6h)
HEARTBEAT_SEC=300        # emit summary every 5 minutes
START=$(date +%s)
LAST_HB=0

ts() { date '+%H:%M:%S'; }

resolve_pid() {
  local x="$1"
  if [[ "$x" =~ ^[0-9]+$ ]]; then
    echo "$x"
  elif [[ -e "$x" ]]; then
    cat "$x" 2>/dev/null || echo "?"
  else
    echo "?"
  fi
}

pid_alive() {
  local p="$1"
  [[ "$p" =~ ^[0-9]+$ ]] && ps -p "$p" >/dev/null 2>&1
}

emit() {
  echo "[$(ts)] $*"
}

emit "WATCHER_START tracked=${#ENTRIES[@]}"
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r tag jobid pid_or_pidfile best_pt <<< "$entry"
  STATE[$tag]="running"
  emit "INIT $tag jobid=$jobid pid_or_pidfile=$pid_or_pidfile expects=$best_pt"
done

while true; do
  for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r tag jobid pid_or_pidfile best_pt <<< "$entry"
    s="${STATE[$tag]}"
    if [[ "$s" == "done" || "$s" == "failed" || "$s" == "eval_failed" ]]; then
      continue
    fi

    pid=$(resolve_pid "$pid_or_pidfile")
    alive=$(pid_alive "$pid" && echo true || echo false)

    if [[ "$s" == "running" ]]; then
      if [[ -e "$best_pt" && "$alive" == "false" ]]; then
        emit "DONE_TRAIN tag=$tag pid=$pid ckpt=$best_pt"
        # Launch eval on the same srun jobid (re-uses the just-freed GPU).
        # Use `setsid nohup` so the eval survives if this watcher is later killed.
        eval_log=runs/phase2/logs/${tag}_eval.log
        setsid nohup bash "$ROOT/scripts/run_eval_on_jobid.sh" "$jobid" "$best_pt" "$tag" \
          > "$eval_log" 2>&1 < /dev/null &
        epid=$!
        EVAL_PID[$tag]=$epid
        EVAL_START[$tag]=$(date +%s)
        STATE[$tag]="evaling"
        emit "EVAL_LAUNCH tag=$tag jobid=$jobid eval_pid=$epid log=$eval_log"
      elif [[ ! -e "$best_pt" && "$alive" == "false" ]]; then
        emit "FAILED tag=$tag pid=$pid (no best.pt; training likely crashed)"
        STATE[$tag]="failed"
      fi
    elif [[ "$s" == "evaling" ]]; then
      summary="runs/phase2/eval/${tag}/eval_summary.json"
      if [[ -e "$summary" ]]; then
        # Show a quick summary of the eval result
        rec=$(grep -E '"recovery"|"scMCC"|"MFE"|"RMSD"|"GC"' "$summary" 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')
        emit "DONE_EVAL tag=$tag summary=$summary metrics=$rec"
        STATE[$tag]="done"
      else
        # check eval timeout
        now=$(date +%s)
        started=${EVAL_START[$tag]:-$now}
        if (( now - started > EVAL_TIMEOUT_SEC )); then
          emit "EVAL_FAILED tag=$tag (timeout after $((now-started))s)"
          STATE[$tag]="eval_failed"
        fi
      fi
    fi
  done

  # heartbeat
  now=$(date +%s)
  if (( now - LAST_HB >= HEARTBEAT_SEC )); then
    LAST_HB=$now
    n_running=0; n_evaling=0; n_done=0; n_failed=0
    for entry in "${ENTRIES[@]}"; do
      IFS='|' read -r tag _ _ _ <<< "$entry"
      case "${STATE[$tag]}" in
        running) n_running=$((n_running+1));;
        evaling) n_evaling=$((n_evaling+1));;
        done)    n_done=$((n_done+1));;
        failed|eval_failed) n_failed=$((n_failed+1));;
      esac
    done
    emit "HEARTBEAT elapsed=$((now-START))s running=$n_running evaling=$n_evaling done=$n_done failed=$n_failed"
  fi

  # all settled?
  all_settled=true
  for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r tag _ _ _ <<< "$entry"
    case "${STATE[$tag]}" in
      done|failed|eval_failed) ;;
      *) all_settled=false; break;;
    esac
  done
  if $all_settled; then
    emit "ALL_SETTLED done=$n_done failed=$n_failed elapsed=$((now-START))s"
    break
  fi

  sleep 60
done
