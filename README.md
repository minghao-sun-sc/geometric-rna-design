# RiboPO: Pareto-Preference Optimization for Structure- and Stability-Aware RNA Design

Preference-optimization framework for RNA inverse folding that addresses two RNA-specific failure modes: **heterogeneous noise** across physical proxies (deterministic 2D folding, stochastic 3D prediction, GC-confounded free energy) and **sequence–structure degeneracy** that enables compositional reward hacking via GC enrichment.

The framework wraps gRNAde with three RNA-specific design choices:

1. **ε-Pareto-dominance preference set** on standardized per-metric features.
2. **Variability-aware margin** (`α_r · σ_m`) — a heteroscedastic Bradley–Terry confidence threshold.
3. **Frozen-reference DPO with a decreasing-margin curriculum** — mirror descent on a KL trust region with a quantitative off-policy bias bound.

Two extensions on top:
- **Thermodynamic-surplus pair filter** that re-fits MFE on length and GC and drops GC-driven pairs (constructive defense against compositional reward hacking).
- **Pareto-DPO Stage 2** — a weight-conditioned policy via FiLM that adds 1024 parameters and lets users traverse the Pareto front at inference time without retraining.

## Headline numbers (DAS test, 98 structures)

| Axis | Metric | gRNAde | RiboPO | Δ |
|---|---|---|---|---|
| 2D | EternaFold scMCC | 0.61 | 0.69 | +13.2% (paired Wilcoxon p=1.2e-3) |
| Thermo | Vienna MFE (kcal/mol) | −30.4 | −34.0 | −11.8% (p=7.0e-6) |
| Thermo | P(target structure) | 0.0027 | 0.0215 | **+687%** (p=1.6e-5) |
| Thermo | Melting Tm (°C) | 36.30 | 37.51 | +1.2 °C |
| 3D | Designability (RMSD<8 Å) | 0.425 | 0.490 | +15.3 pp |
| Func | INF non-canonical | −0.065 | −0.050 | +24% (p=1e-3) |
| Practical | pass@1 (joint criterion) | 0.038 | **0.258** | exceeds gRNAde pass@64 (0.154); 64× sample efficiency |

GC-controlled regression confirms 69% of the MFE gain is GC-independent (p=1.3e-7).

## Repository layout

```
ribopo/
├── src/                          # gRNAde core (unmodified)
├── dpo/                          # single-round DPO + SimPO + IPO + KTO + Pareto-DPO
│   ├── losses.py                 # all preference-optimization loss functions
│   ├── pareto_dpo.py             # weight-conditioned policy (Stage 2)
│   ├── trainer.py                # DPOTrainer with loss dispatcher + Stage-2 wiring
│   ├── train.py                  # CLI entry; --loss_type {dpo,simpo,ipo,kto,pareto_dpo,dpo_is}
│   ├── bench/eval_full.py        # SSTT eval pipeline
│   └── configs/experiments_phase2/ # canonical phase-2 configs
├── multiround/                   # multi-round DPO with curriculum
├── data/
│   ├── pairs_margin125/          # baseline preference pairs (0.125σ margin)
│   ├── pairs_margin25/           # baseline preference pairs (0.25σ margin)
│   └── pairs_thermo_surplus_*/   # GC-controlled thermodynamic-surplus pair sets
├── scripts/
│   ├── build_thermo_surplus_pairs.py  # re-fit MFE regression and re-filter pairs
│   ├── analyze_thermo_surplus.py      # appendix figure
│   ├── dispatch_a100.sh               # srun --jobid --overlap dispatcher
│   ├── auto_eval_watcher.sh           # polling watcher firing eval on training exit
│   ├── eval_phase2_checkpoint.sh      # SSTT eval helper
│   ├── aggregate_phase2_results.py    # combined results table + figures
│   └── plot_beta_recovery_curve.py
├── docs/
│   ├── phase2_experiments.md     # design + status of Phase-2 sprint
│   └── is_dpo_integration_plan.md # importance-corrected iterative DPO plan
├── manuscript/
│   ├── RiboPO_revision/          # editable manuscript working tree (LaTeX)
│   └── RiboPO_ICML_submit/       # frozen ICML 2026 submission
└── tools/                        # external (RhoFold+, EternaFold, Vienna, x3dna, MolProbity)
```

## Quick start

```bash
# Setup
mamba activate grnade  # see env.md for environment details

# Train RiboPO at the canonical β=0.12 with the variability-aware curriculum
python -m dpo.train --config dpo/configs/experiments_phase2/beta_012.yaml

# Train Pareto-DPO Stage 2 (weight-conditioned policy)
python -m dpo.train --config dpo/configs/experiments_phase2/pareto_stage2_b012.yaml

# Train on the GC-controlled thermodynamic-surplus pair set
python -m dpo.train --config dpo/configs/experiments_phase2/thermo_surplus_m25.yaml

# Loss-ablation (IPO / KTO / Pareto-DPO Stage 1)
python -m dpo.train --config dpo/configs/experiments_phase2/ipo_b012.yaml
python -m dpo.train --config dpo/configs/experiments_phase2/kto_b012.yaml
python -m dpo.train --config dpo/configs/experiments_phase2/pareto_dpo_b012.yaml

# Full SSTT evaluation
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml
```

Build the thermodynamic-surplus dataset from the existing margin-25/125 pair sets:

```bash
python scripts/build_thermo_surplus_pairs.py
```

This re-fits `MFE = a + b·L + c·(GC·L)` on the candidate pool and keeps only pairs where the winner has a strictly more negative MFE residual than the loser.

## Method summary

### Preference construction
Given backbone $\mathcal{G}$ and per-metric quality vector $\phi(s) = (\text{pLDDT}, -\text{RMSD}, -\text{MFE})$:

```
D_r = { (s_w, s_l) : φ(s_w) ≽_ε φ(s_l) }   (ε-Pareto dominance)
ε_r,m = α_r · σ_m  (variability-aware margin per metric)
α_r ∈ {0.25, 0.125}  (decreasing curriculum across rounds)
```

Quality gate on the winner: pLDDT > 0.70 AND RMSD < 8.0 Å.

### Theoretical guarantees
- **Theorem 1 (trust-region drift bound):** Frozen-reference + decreasing-α DPO has cumulative drift $\text{KL}(\pi_R \| \pi_{\text{ref}}) \leq G^2 / (2\beta) \sum_r \alpha_r^2$, finite when $\sum \alpha_r^2 < \infty$.
- **Theorem 2 (off-policy bias bound):** Static-pair multi-round DPO has gradient bias $O(\sqrt{R/\beta})$, predicting empirical R5+ degradation when KL exceeds $\log C$ for candidate-pool size $C$.

### Pareto-DPO Stage 2 (optional)
A 1024-parameter FiLM head modulates the gRNAde encoder embeddings on a sampled scalarization weight $w \sim \text{Dirichlet}(\mathbf{1})$. Residual init makes the wrapper exactly equal to the base when loaded from a vanilla gRNAde checkpoint. At inference, query π(s | G, w) for any w to traverse the achievable Pareto front.

## Hyperparameters (canonical)

| Parameter | Symbol | Value |
|---|---|---|
| pLDDT floor (winner) | κ | 0.70 |
| RMSD ceiling (winner) | γ | 8.0 Å |
| Margin (Rounds 1–2) | α_r | 0.25 × σ_m |
| Margin (Rounds 3–5) | α_r | 0.125 × σ_m |
| DPO temperature | β | 0.12 |
| SFT anchor weight | λ_SFT | 0.10 |
| Learning rate | — | 1.8 × 10⁻⁴ |
| Batch size (per pair) | — | 32 |
| Warmup steps | — | 1,000 |

## Data

Preference pair datasets:
- `data/pairs_margin125/` — 0.125σ margin (canonical for ICML submission)
- `data/pairs_margin25/` — 0.25σ margin
- `data/pairs_thermo_surplus_margin{25,125}/` — GC-controlled thermodynamic-surplus filter applied (this repo)

Test split: `data/das_split.pt` (98 structures from DAS benchmark).


## License

See `LICENSE`. The `src/` directory inherits gRNAde's license; original copyright notices preserved.

## Acknowledgements

Built on top of [gRNAde](https://github.com/chaitjo/geometric-rna-design) (Joshi et al. 2025). Evaluation uses RhoFold+, EternaFold, ViennaRNA, USalign, MolProbity, and x3dna-DSSR.
