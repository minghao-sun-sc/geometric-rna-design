Here’s a clean, drop-in **Markdown source** for an offline **DPO-RNA** framework built directly on top of **gRNAde** and grounded in your existing preference pairs and the protein DPO paper’s recipe.

------

# DPO-RNA: Offline Preference Optimization for RNA Inverse Folding (gRNAde Base)

**Goal.** Fine-tune **gRNAde**—a structure-conditioned, autoregressive RNA inverse-folding model with a geometric GNN encoder—using pairwise structural preferences to increase fold fidelity at fixed backbones. gRNAde natively supports single- and multi-state (ensemble) backbones, making it a strong base for preference-tuning.  

**Why gRNAde.** gRNAde is a geometric deep-learning pipeline for RNA inverse design that conditions sequence generation on one or more 3D backbone structures (multi-state capable) and achieves strong recovery and speed; it’s the first explicitly multi-state inverse-folding pipeline with an efficient PyG implementation. 

Note: In no case you modify the gRNAde project original codes. The only codes that you can modify are in the dpo/ folder.

------

## 1) Data: Preference Pairs (Already Built)

We use your curated offline pairs constructed from gRNAde candidates scored by **RhoFold+** (RMSD, pLDDT) and **ViennaRNA** (MFE). Example JSON items and construction notes (10 candidates per backbone; RMSD/pLDDT/MFE recorded) are in `rna_preference_pairs_note.md`. 

**Cleaning.** Removed sequences with RMSD ≥ 128 and pLDDT < 0.30; post-clean counts are summarized there. 

**Correlations.** Strong negative correlation between pLDDT and RMSD; MFE weakly related to both → treat MFE as a subsidiary signal. 

**Pairing (current).** Winners satisfy RMSD < 8 and pLDDT > 0.7; *decision* uses a **0.125·σ** margin for RMSD/pLDDT; add a weak MFE tie-breaker requiring `MFE_w < MFE_l`. 

> **Canonicalization for training.** Convert to **JSONL** per round/epoch (`runs/<run>/epochs/rXX_eYY/pairs.jsonl`). Each line holds:
>
> ```json
> {"pdb_file": "...", "winner_seq": "...", "loser_seq": "...", "weight": 0.0-1.0}
> ```

------

## 2) Training Objective

We follow the **reference-tethered DPO** setup with a small **SFT anchor on winners**, mirroring the protein DPO paper’s strategy of blending chosen+rejected with a winners-only NLL to avoid degeneracy while still learning from rejections. 

### 2.1 DPO loss (reference-tethered)

**Pre-setting.** Grid-search β; start from β = **0.163**.

$$\mathcal{L}_{\text{DPO}}= \mathbb{E}_{(T,S_w,S_l)} \left[ -\log\sigma\!\left( \beta \log \tfrac{\pi_\theta(S_w|T)}{\pi_{\text{ref}}(S_w|T)} -\beta \log \tfrac{\pi_\theta(S_l|T)}{\pi_{\text{ref}}(S_l|T)} \right) \right]$$.

**Reference reset.** Set $\pi_{\text{ref}}\leftarrow \pi_{\theta}$ at the **start of each round**. This “multi-round DPO” pattern is effective in practice and central to the protein paper’s framework.  

### 2.2 SFT anchor (winners only)

Use a small anchor on winners to stabilize token-level behavior:

$\mathcal{L}_{\text{SFT}}=\mathbb{E}_{(T,S_w)}[-\log \pi_\theta(S_w|T)]$, $\quad \lambda = \mathbf{0.153}$.

> In proteins, adding winners-only SFT mitigates minimizing shared subsequences in rejected samples and improves stability; we mirror that here. 

### 2.3 Total loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \lambda\,\mathcal{L}_{\text{SFT}}$$.

------

## 3) Pair Weights from Confidence (optional but recommended)

Given dataset standard deviations $$\sigma_{\text{plddt}}, \sigma_{\text{rmsd}}, \sigma_{\text{mfe}}$$ (from your stats), define:

Δplddt=plddtw−plddtl,Δrmsd=rmsdl−rmsdw,Δmfent=(−mfewL)−(−mfelL).\Delta_{\text{plddt}} = \text{plddt}_w - \text{plddt}_l,\quad \Delta_{\text{rmsd}} = \text{rmsd}_l - \text{rmsd}_w,\quad \Delta_{\text{mfe}}^{\text{nt}} = \left(-\frac{\text{mfe}_w}{L}\right) - \left(-\frac{\text{mfe}_l}{L}\right).z=a⋅Δplddtσplddt+b⋅Δrmsdσrmsd+c⋅Δmfentσmfe,(a,b,c)=(0.7,0.3,0.2).z = a\cdot \frac{\Delta_{\text{plddt}}}{\sigma_{\text{plddt}}} + b\cdot \frac{\Delta_{\text{rmsd}}}{\sigma_{\text{rmsd}}} + c\cdot \frac{\Delta_{\text{mfe}}^{\text{nt}}}{\sigma_{\text{mfe}}},\quad (a,b,c)=(0.7, 0.3, 0.2).w=σ ⁣(k⋅(z−m)),k=2, m=0.5.w = \sigma\!\big(k\cdot(z - m)\big),\quad k=2,\ m=0.5.

Apply ww as a multiplicative factor on −log⁡σ(⋅)-\log \sigma(\cdot) in LDPO\mathcal{L}_{\text{DPO}}. This emphasizes clear wins and down-weights borderline pairs—consistent with the **“quality over quantity”** observation in the protein paper. 

------

## 4) Schedule & Hyperparameters (offline, fixed pairs)

- **Rounds:** 3
- **Epochs/round:** 2
- **Pair refresh:** none (offline); reshuffle each epoch
- **Reference reset:** at the start of each round
- **Optimizer:** AdamW
- **Learning rate:** 1e-4; **cosine annealing** with **warmup_ratio = 0.1**
- **β (DPO temperature):** start 0.163; grid search {0.1, 0.163, 0.25}
- **λ (SFT winners):** 0.153
- **Batch size:** pick for GPU memory (e.g., 128 tokens-per-device equivalent via grad-accum)
- **LoRA:** enable (e.g., rank r=16) to preserve base model priors while adapting. (Used similarly in protein DPO experiments.) 
- Gradient Accumulation: 8

> **Note on scale.** The protein study reports multi-round DPO bringing substantial structure-similarity gains and highlights that **contrastive pair quality matters more than raw count**—your cleaned, margin-filtered pairs fit this guidance.  

------

## 5) Implementation Plan (repo-local)

We add a self-contained `dpo/` package to the **gRNAde** repo:

suggested:

```
gRNAde/  (in practical, offline)
├─ dpo/
│  ├─ __init__.py
│  ├─ trainer.py            # DPO loop (ref-tether, SFT anchor, weighting)
│  ├─ data.py               # JSONL pair loader, bucketing by length/backbone
│  ├─ utils.py              # schedule, logging, metrics, β grid tools
│  ├─ ref_manager.py        # snapshot/reset π_ref each round
│  ├─ lora.py               # optional LoRA attach/detach to gRNAde policy
│  └─ cli.py                # train_offline_dpo, eval, resume
├─ .... #existing gRNAde files
└─ runs/...
```

### 5.1 Policy/Ref API surface

- **Policy** πθ\pi_\theta: existing gRNAde Transformer decoder + geometric encoder. We only need:
  - `logprobs = policy.logprob(sequence, backbone_graphs)` (sum over tokens)
  - supports **single- or multi-state** backbones (gRNAde’s strength).
- **Reference** πref\pi_{\text{ref}}: **frozen clone** of policy at **start of each round**.

> Use exact conditioning **T** (backbone graph[s]) for both policy and ref when evaluating Sw,SlS_w, S_l. This mirrors the protein DPO setup: generate candidates via inverse-folding model, evaluate structure with a folding model (done offline here), and optimize with DPO on chosen vs rejected.  


------

## 6) Evaluation

- **During training (no 3D calls):** report DPO logits statistics, SFT NLL, and policy/reference average log-prob gaps for SwS_w vs SlS_l.
- **Post-training:** on a held-out backbone set, sample N sequences with gRNAde and **score offline** (RhoFold+ RMSD & pLDDT; ViennaRNA MFE). Track:
  - **Fold fidelity:** median RMSD ↓ and pLDDT ↑; (optionally) an RNA structural similarity metric if available.
  - **Energy proxy:** MFE/length (auxiliary).
  - **Diversity:** unique-n-gram ratio / edit distance. (optional)
- **Multi-round check:** expect **suppression of low-quality** and upward shift of best-case fold similarity—observed in protein DPO. 


------

## 9) Practical Notes & Rationale

- **Keep pairs fixed** during offline training (reshuffle only). **Multi-rounds** are still valuable because the **reference reset** tethers the model’s step size per round—an effective pattern validated in the protein study. 
- **Quality over quantity.** Your margins + cleaning already implement a quality filter. Avoid over-inflating pairs; the protein paper shows many more contrastives can dilute signals. 
- **Why winners-only SFT.** Prevents degeneracy from down-weighting common subsequences in losers, matching the protein setting’s intent. 
- **gRNAde multi-state.** If your targets are conformational ensembles, gRNAde’s multi-state encoder is a direct fit; DPO training is unchanged (just ensure `T` packs all states consistently for policy/ref).
- **No 3D calls in training.** All folding/energy calls happened in dataset construction; DPO itself optimizes **log-likelihood ratios** on pairs. 

------

### References (for this plan)

- **gRNAde** architecture & multi-state backbone conditioning (ICLR 2025). 
- **Protein DPO** pipeline & findings: in-silico folding feedback → chosen/rejected → DPO; iterative rounds; quality>quantity.   
- **Your pair construction** (RMSD, pLDDT, MFE; cleaning; thresholds; margins).   

