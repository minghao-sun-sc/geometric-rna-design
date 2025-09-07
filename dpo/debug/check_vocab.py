#!/usr/bin/env python
"""
Debug script to check vocabulary and sequences for tokenization issues.
"""

import sys
import torch
import yaml
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from dpo.patches import patch_featurizer_three_bead
patch_featurizer_three_bead()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dpo.data import PreferencePairDataset, _normalize_seq

def main():
    # Load config
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    dc = cfg["data"]
    
    # Create dataset
    print("Creating dataset...")
    dataset = PreferencePairDataset(
        processed_pt=dc["processed_pt"],
        split_file=dc["split_file"],
        pairs_path=dc["pairs_path_val"],
        split="val",
        max_num_conformers=dc["max_num_conformers"],
        radius=dc["radius"],
        top_k=dc["top_k"],
        num_rbf=dc["num_rbf"],
        num_posenc=dc["num_posenc"],
        noise_scale=dc["noise_scale"],
        device="cpu",
        use_seq_mask=dc.get("use_seq_mask", True),
        strict_length_check=bool(dc.get("strict_length_check", True)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"letter_to_num mapping: {dataset.letter_to_num}")
    
    # Check a few sequences
    print("\nChecking first 5 sequences:")
    for i in range(min(5, len(dataset))):
        entry = dataset.pairs[i]
        wseq = entry["winner_seq"]
        lseq = entry["loser_seq"]
        
        print(f"\nSample {i}:")
        print(f"  Winner raw: '{wseq}' (len={len(wseq)})")
        print(f"  Loser raw:  '{lseq}' (len={len(lseq)})")
        
        wseq_norm = _normalize_seq(wseq)
        lseq_norm = _normalize_seq(lseq)
        print(f"  Winner norm: '{wseq_norm}'")
        print(f"  Loser norm:  '{lseq_norm}'")
        
        # Check for unknown characters
        for c in wseq_norm:
            if c not in dataset.letter_to_num:
                print(f"  ERROR: Character '{c}' not in vocabulary!")
        
        for c in lseq_norm:
            if c not in dataset.letter_to_num:
                print(f"  ERROR: Character '{c}' not in vocabulary!")
        
        # Try tokenizing
        try:
            w_tokens = dataset._encode_seq(wseq)
            l_tokens = dataset._encode_seq(lseq)
            print(f"  Winner tokens: {w_tokens.tolist()}")
            print(f"  Loser tokens:  {l_tokens.tolist()}")
        except Exception as e:
            print(f"  ERROR tokenizing: {e}")
    
    # Test __getitem__
    print("\nTesting __getitem__ on first sample...")
    try:
        data, y_w, y_l, weight, node_mask, gid = dataset[0]
        print(f"  Success! y_w shape: {y_w.shape}, y_l shape: {y_l.shape}")
        print(f"  y_w min/max: {y_w.min()}/{y_w.max()}")
        print(f"  y_l min/max: {y_l.min()}/{y_l.max()}")
        print(f"  Vocab size: {len(dataset.letter_to_num)}")
    except Exception as e:
        print(f"  ERROR in __getitem__: {e}")

if __name__ == "__main__":
    main()