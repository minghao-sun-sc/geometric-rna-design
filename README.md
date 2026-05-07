# RiboPO

**Pareto-Preference Optimization for Structure- and Stability-Aware RNA Inverse Folding.**

RiboPO is a preference-optimization framework for RNA inverse folding that addresses two RNA-specific obstacles: **heterogeneous noise** across physical proxies (deterministic 2D folding, stochastic 3D prediction, GC-confounded free energy) and **sequence–structure degeneracy** that admits compositional reward hacking via GC enrichment. The framework wraps a frozen-reference DPO policy on a gRNAde backbone with three RNA-specific design choices:

1. ε-Pareto-dominance preference labels on per-metric standardized features.
2. A variability-aware margin schedule `α_r · σ_m` (heteroscedastic Bradley–Terry).
3. Frozen-reference DPO with a decreasing-margin curriculum and a structural quality gate against GC-driven shortcuts.

Optional extensions:

- **Thermodynamic-surplus pair filter** that drops GC-driven preference pairs by re-fitting MFE on length and GC.
- **Importance-corrected iterative DPO** (clipped, self-normalized geometric-mean pair ratio at the frozen reference) for multi-round training beyond the static-pair regime.
- **Stage-2 weight-conditioned policy** via a 1024-parameter FiLM head for inference-time Pareto-front traversal.

---

## Repository layout

```
.
├── src/                         # gRNAde core (geometric encoder, decoder, evaluator)
├── dpo/                         # single-round DPO / SimPO / IPO / KTO / Pareto-DPO
│   ├── losses.py                # all preference-optimization loss functions
│   ├── pareto_dpo.py            # weight-conditioned policy (FiLM Stage 2)
│   ├── trainer.py               # training loop with loss dispatcher and Stage-2 wiring
│   ├── train.py                 # CLI: --loss_type {dpo, simpo, ipo, kto, pareto_dpo}
│   ├── bench/eval_full.py       # SSTT evaluation pipeline
│   ├── ckpts/                   # canonical paper checkpoints (.pt files)
│   └── configs/                 # YAML configs (defaults + experiments_phase2/)
├── multiround/                  # multi-round DPO with curriculum and clipped IS
├── data/
│   ├── pairs_margin125/         # ε=0.125·σ preference pairs (canonical)
│   ├── das_split.pt             # DAS test split index
│   └── README.md
├── scripts/                     # data prep, baseline eval, analysis utilities
├── external/                    # baseline submodules (RDesign, RhoDesign, RiFold, RIdiffusion)
├── checkpoints/                 # gRNAde upstream checkpoints (download separately; see below)
├── configs/                     # gRNAde upstream configs
├── tools/                       # third-party binaries (Vienna, EternaFold, x3dna; install separately)
├── main.py                      # gRNAde upstream training entry
├── gRNAde.py                    # gRNAde upstream inference entry
├── evaluate_baselines.py        # SSTT evaluation harness for baseline models
├── env.md                       # environment / dependency notes
├── LICENSE
└── README.md
```

---

## Setup

### 1. Conda environment

A working RNA-design environment with PyTorch + PyTorch-Geometric is required. The end-to-end recipe (CUDA, PyG wheels, ViennaRNA, optional dependencies) is documented in `env.md`. In summary:

```bash
mamba create -n grnade python=3.10 -y
mamba activate grnade
mamba install pytorch=2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
uv pip install torch_geometric
uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
uv pip install wandb pyyaml ipdb python-dotenv tqdm einops ml_collections
mamba install -c bioconda usalign viennarna cd-hit -y
```

### 2. External tools (install under `tools/`)

| Tool                | Used for                                  |
|---------------------|--------------------------------------------|
| ViennaRNA / RNAfold | secondary structure, MFE, ensemble defect  |
| EternaFold          | scMCC (held-out 2D oracle)                 |
| RhoFold+            | 3D structure prediction (training oracle)  |
| USalign             | TM-score / RMSD / GDT                      |
| x3dna-DSSR (v2.4)   | INF metrics (canonical / non-canonical)    |
| MolProbity          | clash / quality post-relaxation            |

Each tool ships with its own install instructions; install under `tools/<name>` and the default config paths will resolve from the repository root.

### 3. Data

- DAS test split (98 structures): `data/das_split.pt` (tracked).
- Preference pairs: `data/pairs_margin125/by_das/clean/{train,val,test}.clean.jsonl` (tracked).
- Larger artifacts (raw candidate pools, baseline outputs, processed feature tensors) are not tracked; the construction scripts live in `scripts/` and `dpo/scripts/`.

### 4. gRNAde upstream checkpoint

The frozen reference policy is the canonical gRNAde autoregressive single-state DAS checkpoint. Download from the gRNAde release and place under `checkpoints/gRNAde_ARv1_1state_das.h5` (paths are configured in `dpo/configs/defaults.yaml`).

### 5. Configuration

The YAML configs use **relative paths from the repository root**. Run all commands with the repository root as the current working directory.

Optional environment variables to set before training:

- `WANDB_PROJECT` and `WANDB_ENTITY` (or edit `dpo/configs/defaults.yaml::wandb`)
- `CUDA_VISIBLE_DEVICES`

---

## Quick start

### Train RiboPO (single-round DPO at the canonical operating point)

```bash
python -m dpo.train --config dpo/configs/experiments_phase2/beta_012.yaml
```

### Train multi-round RiboPO with the decreasing-margin curriculum

```bash
python -m multiround.train --config multiround/config/experiments/15_dpo_dynamic_margins.yaml
```

### Train multi-round with clipped importance-correction (extended-rounds regime)

```bash
python -m multiround.train --config multiround/config/experiments/16_isdpo_on_R5.yaml
```

### Train the thermodynamic-surplus variant (GC-controlled pair filter)

```bash
python -m dpo.train --config dpo/configs/experiments_phase2/thermo_surplus_m25.yaml
```

### Train a Stage-2 weight-conditioned policy (FiLM)

```bash
python -m dpo.train --config dpo/configs/experiments_phase2/pareto_stage2_b012.yaml
```

### Loss-form ablations (IPO / KTO / Pareto-DPO Stage 1)

```bash
python -m dpo.train --config dpo/configs/experiments_phase2/ipo_b012.yaml
python -m dpo.train --config dpo/configs/experiments_phase2/kto_b012.yaml
python -m dpo.train --config dpo/configs/experiments_phase2/pareto_dpo_b012.yaml
```

### Full SSTT evaluation on the DAS test set

```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml
```

The evaluator runs 2D scoring (EternaFold scMCC), 3D refolding (RhoFold+ + USalign), thermodynamics (ViennaRNA partition-function ensemble + ED + P(target) + Tm), and quality / contact metrics (MolProbity + x3dna INF). Results are written to `runs/<tag>/eval_summary.json` plus per-structure tables.

### Build the thermodynamic-surplus pair set from the base 0.125-σ pairs

```bash
python scripts/build_thermo_surplus_pairs.py
```

This re-fits `MFE = a + b·L + c·(GC·L)` on the candidate pool and retains pairs whose winner has a strictly more negative GC-corrected MFE residual than the loser.

### Same-pool reranking control

```bash
bash scripts/eval_pareto_front.sh
```

---

## Method summary

### Preference construction

Given backbone `G` and per-metric quality vector `φ(s) = (pLDDT, −RMSD, −MFE)`:

```
D_r = { (s_w, s_l) : φ(s_w) ≽_ε φ(s_l) }   ε-Pareto dominance
ε_r,m = α_r · σ_m                           variability-aware margin
α_r ∈ {0.25, 0.25, 0.125, 0.125, 0.125}     decreasing curriculum across rounds
```

Quality gate on the winner: `pLDDT(s_w) > 0.70` AND `RMSD(s_w) < 8 Å`.

### Loss

Standard frozen-reference DPO with an SFT anchor on the chosen sequences:

```
L = L_DPO + λ_SFT · L_SFT(s_w)
```

### Hyperparameters (canonical paper operating point)

| Parameter             | Value     |
|-----------------------|-----------|
| pLDDT floor (winner)  | 0.70      |
| RMSD ceiling (winner) | 8.0 Å     |
| Margin (rounds 1–2)   | 0.25 · σ_m|
| Margin (rounds 3–5)   | 0.125 · σ_m|
| DPO inverse-temp β    | 0.12      |
| SFT anchor weight     | 0.10      |
| Learning rate         | 1.8e-4    |
| Candidate pool size C | 8         |
| Reference policy      | gRNAde ARv1 1-state DAS (frozen) |

### Importance-corrected iterative DPO (multi-round)

For round `r ≥ 2` the BTL log-likelihood is reweighted by a clipped, self-normalized geometric-mean pair ratio anchored at the frozen reference, with upper clip `ρ̄ = 5`:

```
w_pair^(r) = min( ρ̄, sqrt( π_{r-1}(s_w) · π_{r-1}(s_l) / π_ref(s_w) · π_ref(s_l) ) )
```

followed by per-batch self-normalization. This is a heuristic variance-controlled surrogate (not the unbiased pair-level Radon–Nikodym ratio); it is motivated by a pair-level Rényi-2 off-policy bias bound and is used to extend usable rounds beyond the static-pair regime.

### Stage-2 weight-conditioned policy (FiLM)

A small FiLM head modulates scalar features in the gRNAde decoder as `h ↦ h(1 + γ(w)) + β(w)`, with `γ, β: R^M → R^{h_dim}` linear and zero-initialized (residual init: at initialization the wrapped policy is exactly the base policy). Total: 1024 extra parameters. During training, sample `w ∼ Dirichlet(1)` per batch; at inference, query at any `w ∈ Δ^{M-1}` to reach a corresponding Pareto operating point. The Pareto-DPO Stage-1 variant uses the same pair set with stochastic Dirichlet `w` but no FiLM head.

---

## Reproducing the canonical results

Canonical RiboPO checkpoints are tracked under `dpo/ckpts/`:

| File                                            | Configuration                  |
|-------------------------------------------------|---------------------------------|
| `dpo/ckpts/beta0.12_lambda0.10_best_pref_acc.pt`| Single-round DPO (β=0.12)       |
| `dpo/ckpts/multi_b0.12_rd1.pt`                  | Multi-round DPO, R=1            |
| `dpo/ckpts/multi_b0.12_rd2.pt`                  | Multi-round DPO, R=2 (primary)  |
| `dpo/ckpts/multi_b0.12_rd4.pt`                  | Multi-round DPO, R=4 (best 2D)  |

Run the SSTT evaluation harness against any of these to reproduce the headline numbers in the paper.

---

## Data and license

- Preference pair sets (`data/pairs_margin125/`) and the DAS test split (`data/das_split.pt`) are tracked under this repository's license.
- The `src/` directory and the gRNAde upstream entry points (`main.py`, `gRNAde.py`) inherit gRNAde's license; original notices are preserved.
- External tools and baseline submodules retain their own licenses; this repository links to them as submodules and does not redistribute their code.

See `LICENSE` for the top-level repository license.

---

## Notes for reproducibility

- **Single training seed.** All shipped checkpoints were trained with one seed; multi-seed retrains are not included here.
- **In-distribution evaluation.** The DAS test set is in-distribution to the gRNAde training corpus; out-of-distribution / family-heldout panels are out of scope for this release.
- **Hardware.** Training was performed on a single A40 GPU (single-round) and on A40 / A100 (multi-round). Memory footprint stays well below 24 GB at the canonical hyperparameters.
- **Wall-clock.** Single-round DPO ≈ 4 h on an A40; each multi-round step adds another ≈ 4 h sample / score / train cycle.
