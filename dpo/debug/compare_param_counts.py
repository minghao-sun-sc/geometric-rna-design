# dpo/debug/compare_param_counts.py
import argparse
from collections import OrderedDict
import yaml
import torch

# --- minimal, repo-independent config loader ---
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --- build ARv1 from cfg (mirrors your default.yaml keys) ---
def build_arv1(cfg):
    from src.models import AutoregressiveMultiGNNv1
    m = AutoregressiveMultiGNNv1(
        node_in_dim=tuple(cfg["node_in_dim"]),
        edge_in_dim=tuple(cfg["edge_in_dim"]),
        node_h_dim=tuple(cfg["node_h_dim"]),
        edge_h_dim=tuple(cfg["edge_h_dim"]),
        num_layers=int(cfg["num_layers"]),
        out_dim=int(cfg["out_dim"]),
        drop_rate=float(cfg["drop_rate"]),
    )
    return m

def flat_named_param_sizes(m):
    return OrderedDict((n, p.numel()) for n, p in m.named_parameters())

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    # base (no LoRA)
    base = build_arv1(cfg).cpu()

    # policy (LoRA applied)
    policy = build_arv1(cfg).cpu()
    from dpo.lora import apply_lora
    lora_cfg = cfg.get("lora", {})
    policy = apply_lora(
        policy,
        r=int(lora_cfg.get("r", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        train_bias=(lora_cfg.get("train_bias", "none") == "all"),
    )

    base_sizes   = flat_named_param_sizes(base)
    policy_sizes = flat_named_param_sizes(policy)

    base_total   = sum(base_sizes.values())
    policy_total = sum(policy_sizes.values())
    delta_total  = policy_total - base_total

    print(f"base total params:   {base_total:,}")
    print(f"policy total params: {policy_total:,}")
    print(f"Δ (policy - base):   {delta_total:,}  <-- expected ~ LoRA params\n")

    # sanity check: how many trainable in policy?
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"policy trainable params: {trainable:,}")

    print("\n---- per-tensor differences (policy - base) ----")
    all_keys = sorted(set(base_sizes) | set(policy_sizes))
    delta_sum = 0
    for k in all_keys:
        b = base_sizes.get(k, 0)
        p = policy_sizes.get(k, 0)
        if b != p:
            print(f"{k:60s}  {b:>9,} -> {p:>9,}  Δ={p-b:+,}")
            delta_sum += (p - b)
    print("SUM Δ =", f"{delta_sum:,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
