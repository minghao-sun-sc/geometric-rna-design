# RiboPO Phase-2 Experiments

> **Status as of 2026-05-03 11:20**
> Thermo-surplus complete (training + SSTT eval). Six remaining trainings 62–84% complete; eval will auto-launch via the watcher when each `best.pt` lands.

This document tracks the post-ICML next-phase experiments. Goal: ICLR 2027 submission with the **Pareto-dominance + heteroscedastic preference learning + trust-region** reframe (Path B).

---

## 0. Live state

### Trainings (PIDs, all alive)

| Tag | Node / GPU | PID | Step | Epoch | pref_acc | best.pt expected | Notes |
|---|---|---|---|---|---|---|---|
| beta_001 | A100 | 547528 | ~4200 / 6500 | ep 6 | 0.94 | `runs/phase2/beta_sweep/b0.01/beta_sweep_b0.01/best.pt` | β-sweep |
| beta_005 | A100 | 547543 | ~5350 / 6500 | ep 8 | 0.96 | `runs/phase2/beta_sweep/b0.05/beta_sweep_b0.05/best.pt` | β-sweep |
| ipo_b012 | A100 | 547557 | ~5450 / 6500 | ep 8 | 0.87 | `runs/phase2/loss_ablation/ipo/ipo_b0.12/best.pt` | IPO loss (squared) — high `loss` is normal, not DPO scale |
| kto_b012 | A40 | 547571 | ~5275 / 6500 | ep 8 | 0.94 | `runs/phase2/loss_ablation/kto/kto_b0.12/best.pt` | KTO loss |
| pareto_dpo_b012 | A40 | 547600 | ~4975 / 6500 | ep 7 | 0.96 | `runs/phase2/loss_ablation/pareto_dpo/pareto_dpo_b0.12/best.pt` | Stage-1 (stochastic w, no FiLM) |
| **pareto_stage2_b012** | A100 | 551215 | ~4100 / 6500 | ep 6 | 0.96 | `runs/phase2/pareto_stage2/pareto_stage2_b0.12/best.pt` | **Stage-2 (FiLM head)** — flagship |

ETA `ALL_DONE` ≈ **2026-05-03 17:00–19:00** (sequential ev­als after each training: ~30–60 min on fast-eval config that skips lDDT/MCQ/clash).

### Watchers (background, persist across compact)

| Process | PID | Role |
|---|---|---|
| `scripts/auto_eval_watcher.sh` | 598263 | Polls each training PID file; on `best.pt` + dead PID, launches `scripts/run_eval_on_jobid.sh` on the same srun jobid via `--overlap`. |
| `scripts/summary_poller.sh` | 719416 | Polls `runs/phase2/eval/<tag>/eval_summary.json`; on full set → runs `scripts/aggregate_phase2_results.py` and exits. |

The two are redundant on purpose: the watcher had a 60-min eval timeout that misfires for the slow SSTT pipeline (it took ~7h for thermo-surplus), but the eval keeps running thanks to `setsid nohup`. The independent poller catches results regardless of watcher state.

### Watcher caveat

Heartbeat at 11:19 reports `failed=1`. That is **`thermo_surplus_m25` mis-classified as `eval_failed`** because the eval exceeded the 60-min watcher timeout (the eval actually completed and `eval_summary.json` is correctly written). Source has been bumped to 8h, but the running watcher pre-dates the fix. Treat watcher state as advisory; the per-tag `eval_summary.json` is the source of truth.

---

## 1. Results so far

### `thermo_surplus_m25` (DPO trained on GC-controlled pair set)

Training: 9 epochs (ended `pref_acc=0.974`). Eval: SSTT panel (no lDDT / MCQ / clash to keep <2 h).

```json
{ "tag": "thermo_surplus_m25",
  "recovery": 0.4959, "scMCC": 0.6500,
  "RMSD": 11.21, "TM": 0.2689, "pLDDT": 0.6298,
  "diversity": 0.8754,
  "inf_all": 0.4856, "inf_wc": 0.1339, "inf_nwc": -0.0472,
  "rmsd_within_8A": 0.4298, "plddt_above_070": 0.4375,
  "vienna_pS0": 0.0304, "vienna_Tm": 41.085,
  "perplexity": 1.2123, "clashscore": 627.4 }
```

vs. gRNAde baseline @ T=0.1, n=8 (from `dpo/eval_results/`):

| Metric | gRNAde | thermo_surplus_m25 | Δ | vs. original RiboPO R2 |
|---|---|---|---|---|
| scMCC ↑ | 0.606 | **0.650** | **+7.3 %** | +1.6 pp better |
| Tm (°C) ↑ | 36.30 | **41.09** | **+4.79 °C** | +3.6 °C better |
| vienna_pS0 ↑ | 0.0027 | **0.0304** | **+1011 %** | +324 pp better |
| rmsd_within_8A ↑ | 0.401 | 0.430 | +2.9 pp | comparable |
| plddt_above_070 ↑ | 0.289 | **0.438** | **+14.9 pp** | comparable |
| INF-WC ↑ | 0.091 | **0.134** | +47 % | comparable |
| INF-NWC ↑ | -0.066 | **-0.047** | +28 % | comparable |
| recovery ↑ | 0.529 | 0.496 | -6.2 % | intentional Pareto trade |
| RMSD ↓ (raw) | 11.4 | 11.2 | -0.18 | comparable |

**Take-away.** Removing the GC-confounded ~24 % of pairs gives **the strongest thermodynamic gains we have ever observed** — `vienna_pS0` (probability the designed sequence folds to the target) jumps 11×, `Tm` +4.8 °C — while structure metrics (RMSD, INF, pLDDT) match or beat the original RiboPO at the cost of a 3 pp recovery drop.

Note: `vienna_mfe` and `vienna_ED_per_nt` came back NaN. Defensive Vienna handles edge cases that crash the main metric — `pS0` and `Tm` are unaffected. Tracking issue: investigate the NaN cases before manuscript v2.

### Other expected results (when trainings settle)

- **β-sweep figure (β = 0.01, 0.05, 0.12, 0.5, 1.0)**: tightens 3QvL Concern 4 by showing the recovery / scMCC trade-off curve.
- **IPO / KTO / Pareto-DPO Stage 1 / Stage 2** ablation: head-to-head on the same pairs, same compute, same eval — for the loss-ablation table in the appendix.

---

## 2. What's already built

### Data
- **Thermodynamic-surplus pair dataset** at `data/pairs_thermo_surplus_margin{25,125}/by_das/clean/{train,val,test}.clean.jsonl`
  - Built via `python scripts/build_thermo_surplus_pairs.py`
  - Refits MFE = a + b·L + c·(GC·L) on the candidate pool. m25 coefficients: `MFE = -1.89 + 0.122·L − 0.791·(GC·L)`, R² = 0.85
  - Drops pairs where the winner's GC-corrected residual is not strictly better than the loser's: m25 train 14628 → 11162 (76.3 % kept); m125 train 20811 → 16051 (77.1 % kept)
  - Full report at `scripts/thermo_surplus_report.json`

### Loss modules (`dpo/losses.py`)
- `ipo_step_losses` — Azar et al. 2024 IPO with squared loss
- `kto_step_losses` — Ethayarajh et al. 2024 KTO adapted to paired data
- `importance_corrected_dpo_step_losses` — Theorem-2 mitigation: per-pair clipped IS weights for static-pair multi-round DPO
- `pareto_dpo_step_losses` — Stage-1 stub: stochastic Dirichlet w, no architectural change
- `step_losses(loss_type, ...)` dispatcher + `_LOSS_REGISTRY`

### Pareto-DPO Stage 2 (`dpo/pareto_dpo.py`)
- `FiLMHead` — 1 024 extra params per layer; init γ = β = 0 → equivalent to base at init
- `WeightConditionedAutoregressiveGNN` — wraps gRNAde backbone; `forward(batch, w=None)` defaults to centroid `w = (1/M, …, 1/M)`
- `pareto_stage2_step_losses` — policy is weight-conditioned, reference is frozen base
- `sample_dirichlet_w` — per-batch Dirichlet sampling
- Total params: 2 148 968 (1 024 FiLM + 2 147 944 base). Residual init verified: `max |h - FiLM(h)|` = 0 at init.

### Configs (`dpo/configs/experiments_phase2/`)
All 11 configs use `wandb.mode: offline` (cluster auth issue):
- β-sweep: `beta_{001,005,012,05,10}.yaml`
- Loss ablation: `ipo_b012.yaml`, `kto_b012.yaml`, `pareto_dpo_b012.yaml`, `pareto_stage2_b012.yaml`
- Thermo-surplus: `thermo_surplus_m25.yaml`
- Sweep base: `beta_sweep_base.yaml`

### Scripts (`scripts/`)
- `build_thermo_surplus_pairs.py` — fits MFE regression, drops GC-driven pairs
- `analyze_thermo_surplus.py` — appendix figure
- `dispatch_a100.sh` — `srun --jobid=<JOBID> --overlap` dispatcher
- `run_eval_on_jobid.sh` — runs SSTT eval on the same srun jobid
- `auto_eval_watcher.sh` — polls each training PID, fires eval when training finishes
- `summary_poller.sh` — independent poller for `eval_summary.json` arrivals
- `eval_phase2_checkpoint.sh` — wraps `dpo.bench.eval_full` with FiLM-stripping for Stage-2 ckpts and a fast-eval config (skip lDDT/MCQ/clash)
- `aggregate_phase2_results.py` — combined results table + figures
- `plot_beta_recovery_curve.py` — β-recovery trade-off figure
- `run_phase2_chain.sh` — sequential chain launcher with idempotent skip-if-best.pt-exists

### Manuscript (`manuscript/RiboPO_revision/`)
- §3.2 ε-Pareto-dominance preference framing
- §3.2.1 heteroscedastic preference learning
- §3.3 Pareto-constrained DPO with trust-region
- 3 new theorem appendix sections (heteroscedastic derivation, trust-region drift, off-policy bias)
- 7 new evidence sections; abstract / intro / conclusion / related rewritten
- Compiles to 36 pages, no warnings

---

## 3. What still needs hands-on work after the chain settles

1. **Aggregate** — `python scripts/aggregate_phase2_results.py` produces:
   - `scripts/phase2_results.json` (combined table)
   - `scripts/phase2_results.tex` (LaTeX appendix table)
   - `scripts/phase2_beta_recovery.pdf` (β-trade-off figure)
   - `scripts/phase2_loss_ablation.pdf` (DPO/IPO/KTO/Pareto bar chart)
2. **Drop figures into manuscript** — `manuscript/RiboPO_revision/section/appendix.tex`, recompile.
3. **Pareto-Stage-2 inference Pareto-front study** — query the trained Stage-2 policy at `w = (1,0,0)`, `(0,1,0)`, `(0,0,1)`, centroid; plot the achievable (scMCC, MFE, RMSD) front. Currently the eval pipeline strips FiLM and reports the centroid-equivalent (base-only) metrics; the Pareto-front study needs a separate `WeightConditionedAutoregressiveGNN`-aware eval.
4. **Investigate `vienna_mfe`/`vienna_ED_per_nt` NaN** in thermo-surplus eval — happens despite defensive Vienna; downstream `pS0` / `Tm` are clean, so likely a length-edge case in the MFE branch only.
5. **Importance-corrected iterative DPO trainer** — wire `importance_corrected_dpo_step_losses` into `multiround/trainer.py`. Per `docs/is_dpo_integration_plan.md`, ~1–2 days; needs (a) candidate resampling between rounds, (b) precompute `log π_θ_{r-1}(s | G)` for each pair, (c) pass IS arrays through. Goal: verify Theorem 2 mitigation (KL drift sublinear vs. linear).
6. **5-base-model port** — RDesign and RhoDesign require porting DPO to non-AR / Transformer architectures; RIdiffusion needs a discrete-diffusion adapter. ~3–5 days each. Submodules already wired: `external/{RDesign,RhoDesign,RiFold,RIdiffusion}`.

---

## 4. Recommended near-term order

| When | Action |
|---|---|
| Now ↦ ~17:00 today | Hands-off. Watcher + poller will fire evals as `best.pt` files land. |
| ~17:00–19:00 today | Aggregate. `python scripts/aggregate_phase2_results.py`. |
| Tomorrow morning | Drop figures + LaTeX table into `manuscript/RiboPO_revision/`. Recompile. Sanity-check β-curve. |
| Day +1 | Stage-2 Pareto-front evaluation (separate FiLM-aware eval). |
| Day +2–3 | Wire importance-corrected iterative DPO. Run R = 1…8, plot scMCC trajectory vs. static-pair. |
| Week +1 | RiFold port (already a submodule, AR architecture, low effort). |

---

## 5. Known caveats

1. **wandb mode = offline** — the cluster has an auth issue. Sync after the fact via `wandb sync wandb/offline-run-<id>`.
2. **GPU node access** — use `srun --jobid=<JOBID> --overlap --pty bash -l` to attach to existing allocations; SLURM jobids for current trainings: A100 = `4790775` and `4790774` and `4792653`; A40 = `4722450`, `4790772`, `4792654` (see `scripts/auto_eval_watcher.sh` `ENTRIES`).
3. **vienna_defensive.py** — relocated from `ribopo_v2/` to `src/`; imported by `src/evaluator.py` as `from src.vienna_defensive import defensive_vienna_mfe, defensive_vienna_ensemble`. Don't move again.
4. **Stage-2 SSTT eval strips FiLM** — `eval_phase2_checkpoint.sh` detects `film.*` keys and saves `<ckpt>_baseonly.pt`. The reported metrics are the base-equivalent (centroid-w) policy. The full Pareto-front study is a separate follow-up.
5. **Watcher 60-min eval timeout pre-dates the 8 h fix** — see §0 caveat. Don't restart the watcher; rely on the summary poller.

---

## 6. Bottom line

The full experimental machinery for the Path B reframe is in place and running. Once the chain settles (~5–7 h from now), the manuscript can be augmented with:

- An appendix figure showing the GC confound is real and removable (already in hand: thermo-surplus bumps `pS0` 11× and `Tm` +4.8 °C with only -3 pp recovery).
- A β-recovery trade-off curve as Fig. X in main text or appendix.
- Head-to-head IPO / KTO / Pareto-DPO Stage 1 / Stage 2 row in the loss-component ablation table.
- Pareto-DPO Stage 2 results as the **flagship multi-objective preference-learning result** (Section 4.X) — the methodological centerpiece of the ICLR 2027 reframe.
