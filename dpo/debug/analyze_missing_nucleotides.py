#!/usr/bin/env python
"""
Analyze how missing nucleotides affect our DPO data alignment.
"""

import sys
import json
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
    
    # Load pairs data directly to analyze
    pairs_path = dc["pairs_path_train"]
    
    print("=== ANALYZING PAIRS DATA ===")
    
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[:10]  # First 10 pairs
        
    for i, line in enumerate(lines):
        pair = json.loads(line)
        pdb_file = pair.get("pdb_file", "unknown")
        winner_seq = pair["winner_seq"]
        loser_seq = pair["loser_seq"]
        
        print(f"\nPair {i}: {pdb_file}")
        print(f"  Winner length: {len(winner_seq)}")
        print(f"  Loser length: {len(loser_seq)}")
    
    print("\n=== ANALYZING DATASET ITEMS ===")
    
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
    
    # Check specific items
    problematic_items = []
    
    for i in range(min(100, len(dataset))):
        try:
            data, y_w, y_l, weight, node_mask, gid = dataset[i]
            
            # Get the raw data
            entry = dataset.pairs[i]
            raw_idx = entry["_local_index"]
            raw = dataset.raw_list[raw_idx]
            
            # Get original sequence from raw data
            raw_seq = raw.get("sequence", "")
            graph_seq_len = data.seq.numel()
            winner_seq = entry["winner_seq"]
            loser_seq = entry["loser_seq"]
            
            # Check for mismatches
            if len(raw_seq) != graph_seq_len:
                problematic_items.append({
                    "index": i,
                    "gid": gid,
                    "raw_seq_len": len(raw_seq),
                    "graph_seq_len": graph_seq_len,
                    "winner_len": len(winner_seq),
                    "loser_len": len(loser_seq),
                    "has_window": entry.get("_window") is not None
                })
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    
    if problematic_items:
        print(f"\nFound {len(problematic_items)} items with potential issues:")
        for item in problematic_items[:5]:  # Show first 5
            print(f"\nItem {item['index']} ({item['gid']}):")
            print(f"  Raw sequence length: {item['raw_seq_len']}")
            print(f"  Graph sequence length: {item['graph_seq_len']}")
            print(f"  Winner sequence length: {item['winner_len']}")
            print(f"  Loser sequence length: {item['loser_len']}")
            print(f"  Uses windowing: {item['has_window']}")
            
            if item['raw_seq_len'] != item['graph_seq_len']:
                print(f"  ⚠️  Raw vs Graph mismatch: {item['raw_seq_len']} != {item['graph_seq_len']}")
                print(f"     This suggests {item['raw_seq_len'] - item['graph_seq_len']} nucleotides were filtered")
            
            if item['winner_len'] != item['graph_seq_len'] and not item['has_window']:
                print(f"  ❌ Winner length doesn't match graph!")
    else:
        print("\nNo problematic items found in first 100 items")
    
    # Check if we have mask_coords in the raw data
    print("\n=== CHECKING FOR mask_coords IN RAW DATA ===")
    sample_raw = dataset.raw_list[0]
    if hasattr(sample_raw, 'mask_coords'):
        print(f"✅ mask_coords found in raw data")
        print(f"   Shape: {sample_raw.mask_coords.shape}")
        print(f"   Type: {type(sample_raw.mask_coords)}")
        print(f"   Valid positions: {sample_raw.mask_coords.sum()} / {len(sample_raw.mask_coords)}")
    else:
        print("❌ No mask_coords attribute in raw data")
        print(f"   Available attributes: {[k for k in dir(sample_raw) if not k.startswith('_')]}")

if __name__ == "__main__":
    main()