"""Compute KL(π_θ_R || π_ref) for a sequence of multi-round checkpoints.

Used by Phase E3 (KL-drift sanity check). For each provided checkpoint, we estimate
KL divergence between the trained policy and the frozen reference policy on a held-out
batch of preference pairs. Theorem 2 predicts that under static-pair multi-round DPO:
- WITHOUT importance correction: KL grows ~linearly in round count (off-policy bias).
- WITH importance correction: KL stays bounded (mitigation).

Estimator: per-pair KL(π_θ || π_ref) using the empirical formula
  KL ≈ E_{s ~ π_θ} [log π_θ(s) - log π_ref(s)]
which we approximate by computing both teacher-forced log-probs on the FROZEN winners
from the test pair set. (The stricter Monte-Carlo estimator would sample from π_θ; we
use the weaker but cheaper teacher-forced bound that's sufficient to show the shape.)

Usage:
    python scripts/compute_kl_drift.py \\
        --ref runs/phase2/pareto_stage2/...best.pt \\
        --policies r1=run_off/round_01/checkpoints/round_1_best.pt \\
                   r2=run_off/round_02/checkpoints/round_2_best.pt ... \\
        --pairs data/pairs_margin25/by_das/clean/test.clean.jsonl \\
        --out runs/multiround/kl_drift_off.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dpo.bench.eval_full import _load_model_filmlike_aware
from dpo.data import DPOPairDataset, collate_batch_pairs
from dpo.losses import seq_logprob
from dpo.utils import load_processed_pt
from src.constants import PROJECT_PATH


def load_model(cfg, ckpt_path, device):
    """Load a model, FiLM-aware, eval mode."""
    model, is_film = _load_model_filmlike_aware(cfg, ckpt_path, device)
    model.eval()
    return model, is_film


def kl_per_batch(policy, ref, batch, max_len=None):
    """Per-pair |log π_θ(s) - log π_ref(s)| for both winner and loser.

    Returns the mean of the two; this is a (rough) per-pair KL surrogate.
    """
    with torch.no_grad():
        lp_w = seq_logprob(policy, batch.graph, batch.winner_seq, max_len=max_len)
        lp_l = seq_logprob(policy, batch.graph, batch.loser_seq, max_len=max_len)
        lr_w = seq_logprob(ref, batch.graph, batch.winner_seq, max_len=max_len)
        lr_l = seq_logprob(ref, batch.graph, batch.loser_seq, max_len=max_len)
        # Per-pair KL surrogate: E[log π_θ - log π_ref] over (w,l)
        kl_w = (lp_w - lr_w)
        kl_l = (lp_l - lr_l)
        return float(kl_w.mean().item()), float(kl_l.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference checkpoint (gRNAde-base)")
    ap.add_argument("--policies", nargs="+", required=True,
                    help="round=ckpt_path entries, e.g. r1=path/to/round_1_best.pt")
    ap.add_argument("--pairs", required=True, help="JSONL pair file (typically test split)")
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--max_pairs", type=int, default=200,
                    help="cap pairs (KL surrogate doesn't need many)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Build a minimal cfg for the loader
    from types import SimpleNamespace as SN
    cfg = SN(model=SN(
        node_in_dim=[15, 4], node_h_dim=[128, 16],
        edge_in_dim=[131, 3], edge_h_dim=[64, 4],
        num_layers=4, drop_rate=0.5, out_dim=4,
        w_dim=3,
    ))

    # Load reference
    print(f"Loading reference: {args.ref}")
    ref, _ = load_model(cfg, args.ref, device)

    # Build dataset (small held-out batch)
    feat_cfg = SN(split="test", radius=0.0, top_k=32, num_rbf=32, num_posenc=32,
                  max_num_conformers=1, noise_scale=0.0, distance_eps=1e-3, device="cpu")
    ds = DPOPairDataset(args.pairs, args.processed_pt, feat_cfg, split_name="test", device="cpu")
    n = min(len(ds), args.max_pairs)
    print(f"Computing KL on {n} test pairs")

    # Loop policies
    results = {}
    for entry in args.policies:
        if "=" not in entry:
            print(f"[skip] expected round=path, got {entry!r}")
            continue
        round_tag, ckpt = entry.split("=", 1)
        print(f"\n=== {round_tag}: {ckpt} ===")
        try:
            policy, _ = load_model(cfg, ckpt, device)
        except Exception as e:
            print(f"failed to load: {e}")
            continue

        kl_w_list, kl_l_list = [], []
        for i in range(n):
            try:
                pair = ds[i]
                kl_w, kl_l = kl_per_batch(policy, ref, pair)
                kl_w_list.append(kl_w)
                kl_l_list.append(kl_l)
            except Exception as e:
                pass

        results[round_tag] = {
            "ckpt": ckpt,
            "n_pairs_evaluated": len(kl_w_list),
            "kl_winner_mean": float(np.mean(kl_w_list)) if kl_w_list else None,
            "kl_winner_std": float(np.std(kl_w_list)) if kl_w_list else None,
            "kl_loser_mean": float(np.mean(kl_l_list)) if kl_l_list else None,
            "kl_pair_mean": float(0.5 * (np.mean(kl_w_list) + np.mean(kl_l_list))) if kl_w_list else None,
        }
        print(f"  KL_winner = {results[round_tag]['kl_winner_mean']:.4f} ± {results[round_tag]['kl_winner_std']:.4f}")
        print(f"  KL_loser  = {results[round_tag]['kl_loser_mean']:.4f}")
        print(f"  KL_pair   = {results[round_tag]['kl_pair_mean']:.4f}")

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
