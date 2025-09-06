# DPO-RNA (offline) — gRNAde adapter

This folder contains a **zero-touch** DPO finetuning stack for gRNAde:
- **No edits** to `src/` or `gRNAde.py`
- Uses **DAS** split
- **Cosine annealing (warm restarts)** schedule
- **Gradient accumulation = 8**
- **≥20 epochs** by default (2 rounds × 10 epochs/round)

---

## Folder contents

## Data prerequisites

1. **Graphs**  
   - `data/processed.pt` — canonical gRNAde processed graphs (unchanged)  
   - `data/das_split.pt` — split indices (train/val/test)

2. **Preference pairs** (JSONL; one object per line)
```json
{
  "backbone_id": "7pzb_A",
  "winner":  "AUGC...U",
  "loser":   "AUGU...U",
  "weight":  1.0,                // optional
  "seq_mask": [1,1,1,...,1]      // optional; 1 = designable position
}

backbone_id must match the IDs used in processed.pt and belong to the split you’re training on.

Lengths must agree: len(winner) == len(loser) == data.seq.numel().


Pair Preferences:
data/pairs_margin125/train.jsonl
data/pairs_margin125/val.jsonl


Train:
python -m dpo.train_dpo --config dpo/configs/default.yaml --wandb --run_name DPO_das_beta0.2_lr1e-4

Multi-GPU:
python -m dpo.train_lightning --config dpo/configs/default.yaml --wandb --project DPO-RNA --run_name DPO_das_beta0.2_lr1e-4_lightning --devices -1 --precision bf16-mixed



Eval:
python -m dpo.eval_from_pdb --config dpo/configs/eval.yaml

The script logs native sequence perplexity, writes sampled sequences to a FASTA, and can optionally compute an EternaFold MCC-style proxy (if ViennaRNA is installed) to mirror the tutorial’s forward-folding idea.








