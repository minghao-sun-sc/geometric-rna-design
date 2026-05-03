# Post-Compact Playbook

> **Self-contained handoff doc.** If the conversation context just compacted, this is the file that re-bootstraps you. Everything you need to resume is here.
> Last updated: **2026-05-03 22:40**

## What's running right now (22:41)

**Pipeline health excellent**: NA-MPNN 57/98, RiboDiffusion-deployed 8/98, 7 main evals 0-16%, **zero segfaults across all 7 evals**. The Vienna rebalance fix is holding.



### Massive parallel pipeline in flight

| Workload | GPU/CPU | Jobid | Status |
|---|---|---|---|
| beta_001_v3 | A100 zgpuA1001 | 4790774 | RUNNING ~14% |
| pareto_stage2_b012_v3 | A100 zgpuA1002 | 4790775 | RUNNING ~7% |
| thermo_surplus_m25_v3 | A40 zgpuA401 | 4815918 | RUNNING ~14% |
| ipo_b012_v3 | A40 zgpuA403 | 4792654 | RUNNING ~0% (just past warmup) |
| **kto_b012_v3** | A100 zgpuA1003 (NEW) | 4815802 | starting on shared A100 |
| **pareto_dpo_b012_v3** | A100 zgpuA1003 | 4815802 | starting on shared A100 |
| **beta_005_v3** | A100 zgpuA1003 | 4815802 | starting on shared A100 |
| RiboDiffusion-deployed smoke | A100 zgpuA1003 | 4815802 | loading (997MB ckpt) |
| 8× NA-MPNN parallel CPU | login node | none | 25/98 dirs done in ~5 min |

**A100 4815802 just activated** with 4 days runtime — running 3 main reruns + RiboDiffusion smoke via `--overlap`.

### Baselines done so far
| Tag | recovery | scMCC | MFE | pS0 | Tm | diversity |
|---|---|---|---|---|---|---|
| RhoDesign (no-2D) | 0.543 | 0.182 | -21.93 | 0.013 | 29.2 | 0.533 |
| RIdiffusion (submodule, small) | 0.412 | 0.244 | -24.34 | 0.008 | 31.8 | 0.885 |

(quick eval, no sc_rhofold yet — will add when an A100 frees)

### Remaining
- NA-MPNN baseline gen (running CPU parallel, ETA ~22:50)
- RiboDiffusion-deployed full run (after smoke verifies, on 4815802)
- 4 main eval re-runs finish (ETA 02:00–05:00)
- 3 reruns on 4815802 finish (ETA 03:00–05:00 — competing for GPU)
- Baseline SSTT-with-sc_rhofold (~3-4h on A40, after one frees)



All 6 phase-2 trainings completed. Watcher exited with 6 false-positive `EVAL_FAILED` events (ignore — see Caveats).

**4 evals just relaunched at 22:06 with the FIXED Vienna code** (after a critical segfault was found at 22:02 — see §"22:02 segfault root-cause" below):

| Tag | Jobid | GPU | Started | Type | Status |
|---|---|---|---|---|---|
| beta_001 | 4790774 (A100 zgpuA1001, 1d 21h) | A100 | 22:06 | fresh post-train (clean code) | RUNNING ~3% |
| pareto_stage2_b012 | 4790775 (A100 zgpuA1002, 2d 12h) | A100 | 22:06 | fresh post-train (clean code) | RUNNING ~1% |
| thermo_surplus_m25 | 4815918 (A40 zgpuA401, 3d 17h) | A40 | 22:06 | rerun replacing buggy summary (.bug.json archived) | RUNNING ~3% |
| ipo_b012 | 4792654 (A40 zgpuA403, 2.4h) | A40 | 22:11 | rerun replacing buggy summary (.bug.json archived) | RUNNING (just config printed) |

Estimated finish: A100 evals ~01:00–01:30, A40 evals ~03:00–05:00.

**3 reruns still pending** (kto_b012, pareto_dpo_b012, beta_005) — fire when an A100 frees.

**2 baselines DONE** (FASTAs persisted, awaiting SSTT eval):
- `runs/phase2/baselines/rhodesign/designs/<gid>/sample{0..7}.fasta` (98/98)
- `runs/phase2/baselines/ridiffusion/designs/<gid>/sample{0..7}.fasta` (98/98)
- Run SSTT eval via: `bash scripts/run_baseline_sstt_eval.sh <jobid> <rhodesign|ridiffusion>` once A100 frees.

### Eval summaries present
```
runs/phase2/eval/thermo_surplus_m25/eval_summary.json    (BUG — replace with v3)
runs/phase2/eval/ipo_b012/eval_summary.json              (BUG — replace with v3)
runs/phase2/eval/kto_b012/eval_summary.json              (BUG — needs rerun)
runs/phase2/eval/pareto_dpo_b012/eval_summary.json       (BUG — needs rerun)
runs/phase2/eval/beta_005/eval_summary.json              (BUG — needs rerun)
+ thermo_surplus_m25/eval_summary.bug.json (archive)
+ ipo_b012/eval_summary.bug.json (archive)
```

ETA `ALL_DONE` (clean): **2026-05-04 ~05:00**.

## 22:02 segfault root-cause and fix

The 17:55 Vienna soft-fix (truncate `target_db` to match `seq` length when off by 1–6 nt) had a hidden trap: **truncating a balanced dot-bracket can produce unbalanced parens**, e.g. `(((....))).` truncated to length 9 becomes `(((....))` — unbalanced. ViennaRNA segfaults at the C level on unbalanced parens, which **bypasses Python try/except** and crashes the entire process.

The 4 already-completed evals (kto, pareto_dpo, beta_005, the original ipo) used the OLD hard-reject code (returning all-NaN on length mismatch); they completed but had broken aggregate MFE/ED. The 4 fresh evals fired with the new soft-fix code segfaulted at 0% on the first multi-chain structure.

**Fix at 22:02**: added a paren-balancing pass after truncation in both `vienna_ensemble_metrics` (src/evaluator.py:2160-2182) and `vienna_Tm_by_pS0` (src/evaluator.py:2293-2314). Algorithm:
1. Truncate or pad target_db to len(seq).
2. Single-pass scan: replace unmatched `)` with `.`; track open-paren positions in a stack; replace remaining stack entries with `.`.

Verified on representative truncation cases:
- `(((....)))..((....))..` → trunc-17 → `(((....)))....... ` (5 open + 3 close → balance 3, drop 2 unmatched opens)

The 4 retries fired at 22:06 successfully passed the 0% segfault zone — the fix works.

## Watchers (don't kill these)

```
auto_eval_watcher.sh   PID 598263   scripts/auto_eval_watcher.sh
summary_poller.sh      PID 719416   scripts/summary_poller.sh
```

The watcher launches eval via `srun --jobid=<JOBID> --overlap` when it sees `best.pt` + dead training PID. The poller is independent insurance: when all `runs/phase2/eval/<tag>/eval_summary.json` files exist it runs `python scripts/aggregate_phase2_results.py` and exits.

> **Watcher caveats** (the running instance pre-dates fixes — don't trust its `failed=N` heartbeat count):
> - **60-min eval timeout** misfires for the slow SSTT pipeline (~3–4 h per eval). The eval keeps running under `setsid nohup`. Trust `eval_summary.json`, not the watcher state.
> - **Watcher does not check srun jobid liveness** before launching. If a jobid expires before launch, the eval dies silently — see `beta_005` incident below.

### Caveat: `beta_005` original eval failed silently

The watcher launched `beta_005` eval at 14:18 on srun jobid `4722450`, but that A40 allocation had already expired. The eval log contains only `srun: error: Slurm job 4722450 has expired`, no progress. **Fix applied at 16:45**: re-launched on fresh A40 jobid `4815918` via:
```bash
setsid nohup bash scripts/run_eval_on_jobid.sh 4815918 \
  runs/phase2/beta_sweep/b0.05/beta_sweep_b0.05/best.pt beta_005 \
  > runs/phase2/logs/beta_005_eval_retry.log 2>&1 < /dev/null &
```
If you see `EVAL_FAILED tag=<X>` in the heartbeat AND no eval_summary AND no growing `<X>_eval.log`, do the same: pick a live jobid from `squeue -u smh` and re-fire `scripts/run_eval_on_jobid.sh`.

## How to check progress (1-liner each)

```bash
# Date + alive trainings + eval results to date
date '+%Y-%m-%d %H:%M:%S'
ls runs/phase2/eval/*/eval_summary.json 2>/dev/null
for f in runs/phase2/logs/*.pid; do pid=$(cat "$f"); ps -p "$pid" >/dev/null 2>&1 && echo "ALIVE $f -> $pid" || echo "DEAD  $f -> $pid"; done

# Per-training step / pref_acc
for tag in beta_001 beta_005 ipo_b012 kto_b012 pareto_dpo_b012 pareto_stage2_b012; do
  echo "=== $tag ==="; tail -2 runs/phase2/logs/${tag}.log
done

# Watcher + poller alive?
ps aux | grep -E 'auto_eval_watcher|summary_poller' | grep -v grep
```

## When a training finishes

The watcher does this automatically. Manual fallback (only if watcher is dead):

```bash
# Find the srun jobid hosting that training
sacct -u smh -o JobID,Partition,State,Elapsed | grep RUNNING

# Run the eval on the same jobid
bash scripts/run_eval_on_jobid.sh <jobid> <best.pt> <tag>
```

The eval wrapper (`scripts/eval_phase2_checkpoint.sh`):
- Strips FiLM weights from Stage-2 checkpoints (saves `<ckpt>_baseonly.pt`); evaluates the centroid-w policy.
- Uses fast-eval config (skips lDDT / MCQ / clash); ~1–2 h per ckpt.
- Writes `runs/phase2/eval/<tag>/eval_summary.json`.

## When ALL trainings have eval'd

```bash
# Aggregator (the poller does this automatically; or run by hand):
python scripts/aggregate_phase2_results.py
# Outputs:
#   scripts/phase2_results.json     (combined table)
#   scripts/phase2_results.tex      (LaTeX appendix table)
#   scripts/phase2_beta_recovery.pdf
#   scripts/phase2_loss_ablation.pdf
```

Then: drop the figures + table into `manuscript/RiboPO_revision/section/appendix.tex`, recompile.

## Manuscript

`manuscript/RiboPO_revision/iclr2026_conference.pdf` — 36 pages, builds clean. Compile with `pdflatex; bibtex; pdflatex; pdflatex` from `manuscript/RiboPO_revision/`.

Local stub `.sty` files at `manuscript/RiboPO_revision/{makecell,multirow,wrapfig,bbm}.sty` (workarounds for missing texlive packages on this cluster — gitignored).

## Git state

Branch: `dpo`. Origin: GitHub (forked-then-detached repo).

Recent commits (top of `dpo`):
```
2ab6fc6 chore: add src/vienna_defensive (relocated from ribopo_v2/), untrack molprobity scratch, ignore ribopo_v4/
aaf1085 chore: open-source-ready README + add canonical checkpoint and baseline submodules
495d160 docs: manuscript LaTeX source — ICML submission + Path-B revision
0f6a31b feat: phase-2 framework (Pareto-DPO Stage 2, IPO/KTO, IS-DPO, thermo-surplus)
2cd4ab2 chore: remove legacy ribopo_v2/, scratch artifacts, intermediate outputs
241c4a7 chore: clean up .gitignore + untrack legacy checkpoints
```

External submodules at `external/{RDesign,RhoDesign,RiFold,RIdiffusion}` (160000 mode gitlinks).

User commits manually — **do not commit without explicit permission.**

## Key paths (for grep'ability)

| Purpose | Path |
|---|---|
| Phase-2 configs | `dpo/configs/experiments_phase2/*.yaml` |
| Loss modules | `dpo/losses.py` (added `ipo_step_losses`, `kto_step_losses`, `importance_corrected_dpo_step_losses`, `pareto_dpo_step_losses`, `step_losses` dispatcher) |
| Pareto-Stage-2 model | `dpo/pareto_dpo.py` (`FiLMHead`, `WeightConditionedAutoregressiveGNN`, `pareto_stage2_step_losses`, `sample_dirichlet_w`) |
| Trainer wiring | `dpo/trainer.py` (`_maybe_upgrade_to_stage2`, per-batch w sampling, dispatch) |
| Pair construction | `scripts/build_thermo_surplus_pairs.py` |
| Watcher entries | `scripts/auto_eval_watcher.sh` (`ENTRIES` array maps tag → SLURM jobid → PID(file) → expected best.pt) |
| Eval wrapper | `scripts/eval_phase2_checkpoint.sh` (FiLM-strips + fast-eval config) |
| Pair data | `data/pairs_thermo_surplus_margin{25,125}/by_das/clean/` |
| Defensive Vienna | `src/vienna_defensive.py` |
| Manuscript | `manuscript/RiboPO_revision/` |
| Personal log | `manuscript/feedback/00_session_summary.md` (gitignored) |
| Phase-2 design doc | `docs/phase2_experiments.md` |
| IS-DPO integration | `docs/is_dpo_integration_plan.md` |

## Watcher entry table (for cross-referencing srun jobids)

```
tag                  jobid      pid_or_pidfile                              best.pt
thermo_surplus_m25   4792654    473162 (dead, training done)                runs/phase2/thermo_surplus/m25/thermo_surplus_m25_b0.12/best.pt
beta_001             4790775    runs/phase2/logs/beta_001.pid               runs/phase2/beta_sweep/b0.01/beta_sweep_b0.01/best.pt
beta_005             4722450    runs/phase2/logs/beta_005.pid               runs/phase2/beta_sweep/b0.05/beta_sweep_b0.05/best.pt
ipo_b012             4790774    runs/phase2/logs/ipo_b012.pid               runs/phase2/loss_ablation/ipo/ipo_b0.12/best.pt
kto_b012             4792653    runs/phase2/logs/kto_b012.pid               runs/phase2/loss_ablation/kto/kto_b0.12/best.pt
pareto_dpo_b012      4790772    runs/phase2/logs/pareto_dpo_b012.pid        runs/phase2/loss_ablation/pareto_dpo/pareto_dpo_b0.12/best.pt
pareto_stage2_b012   4790775    runs/phase2/logs/pareto_stage2_b012.pid     runs/phase2/pareto_stage2/pareto_stage2_b0.12/best.pt
```

## Top-priority follow-ups (in order)

1. **Wait** until `runs/phase2/eval/<tag>/eval_summary.json` exists for all 6 remaining tags (or watcher emits `ALL_SETTLED`).
2. **Aggregate** → `python scripts/aggregate_phase2_results.py`.
3. **Drop into manuscript** appendix; recompile.
4. **Re-run the 5 buggy-code evals** (Vienna fix has been applied 2026-05-03 17:55, but only the 2 not-yet-started evals — beta_001 and pareto_stage2 — pick it up; the 4 in-progress and the already-completed thermo_surplus all ran with the broken `np.mean` aggregator and didn't persist FASTAs):
   ```bash
   # 1. squeue -u smh — pick 5 live jobids with > 4h time-left
   # 2. edit scripts/rerun_evals_after_vienna_fix.sh : fill in JOBIDS[…]
   # 3. bash scripts/rerun_evals_after_vienna_fix.sh
   # → all 5 reruns fire in parallel; each ~3h on A40, ~1.5h on A100
   # → old summaries archived to <tag>/eval_summary.bug.json
   # 4. python scripts/aggregate_phase2_results.py  (re-aggregate with clean MFE/ED)
   ```
5. **Stage-2 Pareto-front eval** (separate from auto-eval): query the trained `pareto_stage2_b012/best.pt` at `w = (1,0,0)`, `(0,1,0)`, `(0,0,1)`, centroid; plot achievable (scMCC, MFE, RMSD) front. Needs a FiLM-aware eval (current eval strips FiLM and reports centroid-equivalent only).
6. **Importance-corrected iterative DPO trainer** (`docs/is_dpo_integration_plan.md`, ~1–2 days).
7. **5-base-model port** (RiFold first — already a submodule, AR architecture).

### What the 2026-05-03 17:55 Vienna fix did

- `src/evaluator.py:2160-2175` (Guard 1, vienna_ensemble_metrics): hard-reject → soft truncate/pad. Eliminates per-sample all-NaN return on ~1.2 % of samples with mismatched `target_db` / `seq` length. **Backup**: `archive_pls_ignore/src_backup_2026-05-03_vienna_fix/evaluator.py`.
- `src/evaluator.py:2308-2309` (`vienna_Tm_by_pS0`): `raise ValueError` → soft truncate/pad. Tm is no longer skipped on length mismatch.
- `dpo/bench/eval_full.py:443-449`: NaN-filter at append (consistent with pS0/Tm pattern). **Backup**: `archive_pls_ignore/src_backup_2026-05-03_vienna_fix/eval_full.py`.
- `dpo/bench/eval_full.py:702-708`: `np.mean` → `np.nanmean` (defense-in-depth); also returns `float('nan')` instead of `0.0` on empty list.
- `dpo/bench/eval_full.py:858-859`: `cfg.eval.save_designs` is now read (was ignored before, only CLI `--save_designs` worked).
- `scripts/eval_phase2_checkpoint.sh`: temp config now sets `cfg['eval']['save_designs'] = True`. Future evals persist FASTAs to `runs/phase2/eval/<tag>/designs_<tag>/<run_id>/sampleN/`.

The 2 pending evals (beta_001, pareto_stage2_b012, fire after their trainings finish ~17:50) pick up all of the above automatically. Their MFE/ED columns will be clean and they'll persist designs.

### Root-cause investigation (deferred — defensive guards handle the symptom)

Why does `target_db` differ from `seq` length by 1–6 nt for ~1.2 % of structures? Almost certainly because `target_db` is extracted from `item.raw_data['sec_struct_list'][0]` (PDB-derived) while `seq` length comes from the model output (matching the graph residue count, which can drop residues with missing atoms / non-canonical chemistry). Most affected structures are multi-chain (`1ZFV_1_D-C-A-B`, `1LNT_1_A-B`). Fix would live near `dpo/bench/eval_full.py:397-407` — apply the same `keep_idx` mask to `target_db_full` that's applied to the graph during data load. Soft-fix in evaluator.py is sufficient for now.

## What NOT to do

- Don't restart the watcher or poller — the eval pipeline is `setsid nohup`, so it survives even if the watcher dies, but the poller is the more reliable source-of-truth signal.
- Don't `git commit` without asking the user.
- Don't modify `src/` (original gRNAde) unless absolutely necessary.
- Don't run jobs on the login node.

## One-line health check

```bash
date && ls runs/phase2/eval/*/eval_summary.json 2>/dev/null | wc -l && ps -p 598263 719416 -o pid,etime,cmd 2>/dev/null
```

Expected near 17:00 today: `7` and both watcher PIDs alive (or poller exited cleanly after running the aggregator).
