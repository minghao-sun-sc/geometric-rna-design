#!/usr/bin/env python
"""
Test the specific item that was causing the mismatch.
"""

import sys
import torch
import yaml
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from dpo.patches import patch_featurizer_three_bead
patch_featurizer_three_bead()

sys.path.append(str(Path(__file__).parent.parent.parent))

from dpo.data import PreferencePairDataset

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    dc = cfg["data"]
    
    dataset = PreferencePairDataset(
        processed_pt=dc["processed_pt"],
        split_file=dc["split_file"],
        pairs_path=dc["pairs_path_train"],
        split="train",
        max_num_conformers=dc["max_num_conformers"],
        radius=dc["radius"],
        top_k=dc["top_k"],
        num_rbf=dc["num_rbf"],
        num_posenc=dc["num_posenc"],
        noise_scale=dc["noise_scale"],
        device="cpu",
        use_seq_mask=dc.get("use_seq_mask", True),
        strict_length_check=bool(dc.get("strict_length_check", False)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    print("=== TESTING ITEM 900 (Previously Problematic) ===")
    
    data, y_w, y_l, weight, node_mask, gid = dataset[900]
    
    graph_len = data.seq.numel()
    target_len = y_w.numel()
    mask_len = node_mask.numel() if node_mask is not None else 0
    
    print(f"Item 900 ({gid}):")
    print(f"  Graph seq: {graph_len} tokens")
    print(f"  Target y_w: {target_len} tokens")
    print(f"  Target y_l: {y_l.numel()} tokens")
    print(f"  Node mask: {mask_len} elements")
    
    if graph_len == target_len == y_l.numel() == mask_len:
        print(f"  ✅ ALL ALIGNED: {graph_len} elements")
    else:
        print(f"  ❌ Still misaligned!")
    
    # Check the mask values
    if node_mask is not None:
        active_positions = node_mask.sum().item()
        print(f"  Mask active positions: {active_positions}")
        print(f"  Mask: {node_mask.tolist()}")

if __name__ == "__main__":
    main()