# RiboPO Phase-2 Experiments

This document tracks the post-ICML next-phase experiments. Goal: ICLR 2027 submission with the Pareto-dominance + heteroscedastic preference learning + trust-region reframe.

## What's already built (this session, 2026-05-02)

### Data
- **Thermodynamic-surplus pair dataset** at `data/pairs_thermo_surplus_margin{25,125}/by_das/clean/{train,val,test}.clean.jsonl`.
  - Built via `python scripts/build_thermo_surplus_pairs.py`.
  - Refits MFE = a + b·L + c·(GC·L) on the candidate pool (R² ≈ 0.84 on m25, 0.83 on m125).
  - Drops pairs where the winner's GC-corrected residual is not strictly better than the loser's: ~23% of m25 pairs (3,466 / 14,628 train) and ~23% of m125 (4,760 / 20,811 train) were GC-driven and dropped.
  - The remaining pairs encode preferences that survive the GC confound.
  - Full report in `scripts/thermo_surplus_report.json`.

### Loss modules (`dpo/losses.py`)
- `ipo_step_losses` — Azar et al. 2024 IPO with squared loss. Designed to be more robust to preference noise than DPO.
- `kto_step_losses` — Ethayarajh et al. 2024 KTO adapted to paired data. Pointwise sigmoid losses, useful for binary good/bad labelling.
- `pareto_dpo_step_losses` — Stage-1 stub: stochastic Dirichlet weight sampling without architectural change. Stage-2 (full Pareto-DPO with weight-conditioned policy) requires a FiLM head — see Pareto-DPO design below.
- `importance_corrected_dpo_step_losses` — Theorem-mitigation: per-pair clipped importance weights for static-pair multi-round DPO.

### Configs (`dpo/configs/experiments_phase2/`)
- `thermo_surplus_m25.yaml` — DPO on the GC-controlled pair set.
- `beta_001.yaml`, `beta_005.yaml`, `beta_012.yaml`, `beta_05.yaml`, `beta_10.yaml` — five β values for the trade-off curve.
- `beta_sweep_base.yaml` — common base config (override `dpo.beta` and `wandb.run_name`).
- All configs use `wandb.mode: offline` (entity issue on cluster).

### Launcher
- `scripts/run_phase2_sweep.sh` — wraps `python -m dpo.train` with nohup logging.

## Recommended experimental order

### Sprint 1 (1-2 weeks): immediate wins from existing infrastructure
1. **Train thermo-surplus DPO** on the new pair set, compare scMCC / RMSD / MFE / GC against the m25 baseline. *Already launched in background as of 2026-05-02 14:04, log at `runs/phase2/logs/thermo_surplus_m25.log`.*
2. **β-recovery sweep**: launch `bash scripts/run_phase2_sweep.sh beta_sweep` to train all 5 β values sequentially. Each is ~3-4 hours @ 10 epochs with 11K-16K pairs (down from 16h @ 20 epochs in the original paper because we use shorter epochs for the sweep figure).
3. **Eval each checkpoint** with the canonical SSTT pipeline:
   ```bash
   python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml \
       --ckpt runs/phase2/beta_sweep/b{X}/best.pt --tag beta_b{X}
   ```
4. **Build the β-recovery figure**: x-axis β ∈ {0.01, 0.05, 0.12, 0.5, 1.0}, y-axes recovery, MFE, scMCC, RMSD. Settles 3QvL Con 4. Single matplotlib figure.

### Sprint 2 (2-3 weeks): IPO/KTO ablation + Pareto-DPO Stage 1
5. **IPO baseline** at β=0.12. Same data, same compute. Add a "loss type" config switch.
6. **KTO baseline** at β=0.12. Same data.
7. **Pareto-DPO Stage 1** (stochastic weight sampling, no arch change) — single training run.
8. **Head-to-head table**: DPO / IPO / KTO / Pareto-DPO Stage 1 / SimPO on the same pair set, same compute.

### Sprint 3 (3-4 weeks): Pareto-DPO Stage 2 + iterative DPO with IS correction
9. **Pareto-DPO Stage 2**: implement the FiLM-conditioned policy head; sample w ~ Dirichlet at training; query at inference. See design doc below.
10. **Importance-corrected iterative DPO**: precompute log π_θ_{r-1} on the pair set before round r; pass to `importance_corrected_dpo_step_losses`. Run R=1...8 and verify scMCC degradation is slowed vs static-pair multi-round.

### Sprint 4 (3-4 weeks): generality + reward-decomposition
11. **5-base-model transfer**: gRNAde + RiFold + RDesign + RhoDesign + RIdiffusion. RDesign and RhoDesign require porting the DPO loss to non-AR / Transformer architectures — moderate effort.
12. **Reward-decomposition study**: train DPO with single-objective preferences (pLDDT-only, RMSD-only, MFE-only, scMCC-only). Disentangles which gain comes from which signal.

## Pareto-DPO Stage-2 design (full weight-conditioning)

### Architecture
- Add a small FiLM (Feature-wise Linear Modulation) layer to the gRNAde decoder that takes `w ∈ Δ^{M-1}` (M=3) as input.
- FiLM produces per-dimension scale/shift `(γ_w, β_w)` applied to the hidden state `h_t` at each AR step: `h_t' = γ_w ⊙ h_t + β_w`.
- Total parameter cost: M * 2 * h_dim ≈ 3 * 2 * 128 = 768 new params per FiLM layer; negligible.

### Training
- Sample `w ~ Dirichlet(1)` at each training step (or per-pair).
- Standard BTL on the Pareto-dominant pair set, with the log-ratio scaled by `w^T φ̂`, where `φ̂ ∈ R^M` is the per-metric standardised reward gap (precomputed at pair construction).
- This gives a policy that is consistent with *every* `w` simultaneously, exposing the implicit reward to the full Pareto cone during training.

### Inference
- Query with any `w` to traverse the Pareto front.
- Default: `w = (1/M, ..., 1/M)` recovers the unweighted policy.
- Application: a biologist can request "design RNAs maximizing structural fidelity" (`w_RMSD = 1`) or "maximizing thermodynamic stability" (`w_MFE = 1`) at inference, without retraining.

### Evaluation
- Plot the achievable (scMCC, MFE) Pareto front as `w` varies over Δ².
- Compare against single-policy RiboPO (current method): is RiboPO a slice of the Pareto-DPO front, or does Pareto-DPO expand it?

### Implementation skeleton (TODO)
```python
# 1. Modify model_factory.py to add FiLM layer to AutoregressiveMultiGNNv1
class FiLMHead(nn.Module):
    def __init__(self, w_dim, h_dim):
        super().__init__()
        self.gamma = nn.Linear(w_dim, h_dim)
        self.beta  = nn.Linear(w_dim, h_dim)
    def forward(self, h, w):
        return self.gamma(w) * h + self.beta(w)

# 2. Wrap policy:
class WeightConditionedAR(AutoregressiveMultiGNNv1):
    def __init__(self, ..., w_dim=3):
        super().__init__(...)
        self.film = FiLMHead(w_dim, self.h_dim)
    def forward(self, graph, w):
        h = self.encode(graph)
        h = self.film(h, w)
        return self.decode(h)

# 3. Training loop (in dpo/trainer.py):
for batch in loader:
    w = torch.distributions.Dirichlet(torch.ones(3)).sample().to(device)
    # Pass w through the policy AND the reference (reference uses w too for fair comparison)
    losses = pareto_dpo_step_losses(
        model_with_w=lambda g, s: model(g, s, w=w),
        ref_model_with_w=lambda g, s: ref(g, s, w=w),
        batch=batch, beta=beta, ...
    )
```

Estimated effort: 1-2 weeks (architecture mods + training run + eval).

## Importance-corrected iterative DPO design

### Algorithm
```
Input: π_ref (frozen), pair set D = D_round_1 from candidates of π_ref.
For r = 1, 2, ..., R:
    if r >= 2:
        # Resample candidates from π_θ_{r-1} for fresh on-policy data
        D_round_r = resample_pairs(π_θ_{r-1}, n_candidates=8/backbone)
        # Compute importance weights w.r.t. previous round's policy
        for pair in D_round_r:
            pair.log_prev_w = log π_θ_{r-1}(s_w | G).detach()
            pair.log_prev_l = log π_θ_{r-1}(s_l | G).detach()
    else:
        D_round_r = D
    train(π_θ_r, π_ref, D_round_r, loss=importance_corrected_dpo_step_losses)
```

### Expected outcome
Theorem~\ref{thm:off_policy} predicts linear KL drift `KL(π_θ_R || π_ref) ~ R β E[Δr]` for static pairs. With IS correction, drift should be sublinear or bounded. Empirical signature: scMCC trajectory should *not* collapse at R5+ (vs. the current static-pair plateau-then-decline pattern).

### Implementation
- Modify `multiround/trainer.py` to add a `resample_pairs` step between rounds.
- Use `importance_corrected_dpo_step_losses` instead of `dpo_step_losses`.
- Cost: ~2 weeks (resampling pipeline + IS bookkeeping + ablation R=1...8).

## Eval pipeline

After each checkpoint, eval with:
```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml \
    --ckpt <path> --output_dir runs/phase2/eval/<tag>
```

Ensure SSTT panel includes: recovery, scMCC, MFE, P(target), Tm, scRMSD, scTM, scGDT, INF (all/WC/NWC), GC content.

## Known caveats

1. **wandb mode = offline**: the cluster has an auth issue. Sync after the fact via `wandb sync wandb/offline-run-<id>`.
2. **GPU node access**: use `srun --jobid=<JOBID> --overlap --pty bash -l` to attach to existing allocations.
3. **vienna_defensive.py** has been relocated from `ribopo_v2/` to `src/`; no further changes needed.

## TL;DR — what's live as of 2026-05-02 14:14

1. ✅ **Thermo-surplus pair dataset** built; ~24% of pairs were GC-driven and dropped.
2. ✅ **Analysis figure** at `scripts/thermo_surplus_figure.pdf` showing GC-driven vs kept pairs (publishable for the appendix).
3. ✅ **Thermo-surplus DPO training** running (PID 473162, started 14:04, first checkpoint at step 50 = 14:14, rate ≈ 13 sec/step → ~6 hours per 10-epoch run). Logs at `runs/phase2/logs/thermo_surplus_m25.log`.
4. ✅ **β-recovery sweep configs** ready at `dpo/configs/experiments_phase2/beta_{001,005,012,05,10}.yaml`. Launch with `bash scripts/run_phase2_sweep.sh beta_sweep`.
5. ✅ **Loss modules** added: `ipo_step_losses`, `kto_step_losses`, `importance_corrected_dpo_step_losses`, `pareto_dpo_step_losses` (Stage-1 stochastic-w stub) in `dpo/losses.py`, with a `step_losses(loss_type, ...)` dispatcher.
6. ✅ **IPO/KTO/Pareto-DPO configs** at `dpo/configs/experiments_phase2/{ipo,kto,pareto_dpo}_b012.yaml`. Launch with `bash scripts/run_phase2_sweep.sh loss_ablation`.
7. ✅ **DPOTrainer wired** for the loss dispatcher: setting `loss_type: <ipo|kto|pareto_dpo|dpo_is>` in any config will use the new loss without further code changes.
8. ✅ **Pareto-DPO Stage 2 (weight-conditioned policy)** at `dpo/pareto_dpo.py`. `WeightConditionedAutoregressiveGNN` wraps the gRNAde backbone with a 1024-param FiLM head; residual init means it is *exactly* equivalent to the base model when loaded from a vanilla checkpoint, then learns to use w during training. This is the **ICLR 2027 methodological centerpiece**.
9. ✅ **β-recovery plot script** at `scripts/plot_beta_recovery_curve.py` (auto-builds the figure once eval summaries are in).
10. ✅ **Eval helper** at `scripts/eval_phase2_checkpoint.sh <ckpt> <tag>` — wraps the canonical SSTT pipeline and writes a compact `eval_summary.json` per checkpoint.

## What still needs hands-on work

- **Pareto-DPO Stage 2 training loop**: `dpo/pareto_dpo.py` provides the model wrapper, but the trainer doesn't yet call `wrapped(batch, w=...)`. Add a `WrapperWithWeight` config and a small adapter in `dpo/trainer.py` (~30 LOC). Then sample `w ~ Dirichlet(1)` per batch in the inner loop and pass to forward. Estimated: 2-3 hours.
- **Importance-corrected iterative DPO trainer**: `multiround/trainer.py` needs (a) candidate resampling between rounds, (b) precomputing `log π_θ_{r-1}(s^w | G)` for each pair before round r, (c) passing the IS arrays to `importance_corrected_dpo_step_losses`. Estimated: 1-2 days.
- **5-base-model port**: RDesign and RhoDesign require porting the DPO loss to non-AR / Transformer architectures. RIdiffusion needs a discrete-diffusion adapter. Each is its own 3-5 day porting task; gRNAde and RiFold are already done.

## Recommended near-term order

Day 1-3: Let the thermo-surplus DPO finish (~6 hr). Eval with `scripts/eval_phase2_checkpoint.sh`. Compare to the existing `dpo/ckpts/multi_b0.12_rd2.pt` as the raw-MFE baseline.

Day 1 (parallel): Launch the β-sweep with `bash scripts/run_phase2_sweep.sh beta_sweep`. Five runs × ~6 hours sequentially = ~30 hours. Better: run on 5 separate GPU allocations in parallel for ~6 hours total.

Day 4: Eval all sweep checkpoints, run `scripts/plot_beta_recovery_curve.py`, drop the figure into the manuscript appendix.

Day 5-6: Launch IPO + KTO + Pareto-DPO Stage 1 (`bash scripts/run_phase2_sweep.sh loss_ablation`). Eval each.

Day 7+: Begin Pareto-DPO Stage 2 trainer wiring. Begin importance-corrected iterative DPO trainer.

## Bottom line

The full experimental machinery for the Path B reframe is in place. With ~6-8 GPU-hours of trainings completing in parallel, the manuscript can be augmented with:
- A new appendix figure showing the GC confound is real and removable.
- A β-recovery trade-off curve as Fig. X in the main text or appendix.
- Head-to-head IPO/KTO/Pareto-DPO ablation row in the loss-component ablation table.
- Pareto-DPO Stage 2 results as a flagship Section 4.X "Multi-objective preference learning" main result if Stage-2 training completes by deadline.
