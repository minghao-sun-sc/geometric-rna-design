#!/usr/bin/env python
"""
Debug script to test the dataloader and understand batch structure.
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

from dpo.data import PreferencePairDataset, collate_pairs
from torch.utils.data import DataLoader

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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pairs,
    )
    
    # Get one batch
    print("\nGetting one batch...")
    for batch in dataloader:
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
        
        if isinstance(batch, (list, tuple)):
            for i, item in enumerate(batch):
                print(f"  Item {i}: type={type(item)}, ", end="")
                if hasattr(item, 'shape'):
                    print(f"shape={item.shape}")
                elif hasattr(item, '__len__'):
                    print(f"len={len(item)}")
                else:
                    print(f"value={item}")
                    
                # Check if it's a PyG batch
                if hasattr(item, 'batch'):
                    print(f"    PyG batch detected, has attributes: {list(item.keys())}")
                    
        # Test what _unwrap_batch would do
        print("\nAfter _unwrap_batch simulation:")
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            unwrapped = batch[0]
            print(f"  Type: {type(unwrapped)}")
            if hasattr(unwrapped, 'y_w'):
                print("  Has y_w: YES")
            else:
                print("  Has y_w: NO")
            if hasattr(unwrapped, 'batch'):
                print(f"  PyG batch attributes: {list(unwrapped.keys())}")
                
        break  # Only need one batch
    
    print("\nExpected batch structure from collate_pairs:")
    print("  (data_batch, y_w, y_l, w, node_mask, gids)")
    print("\nThe issue: _unwrap_batch takes only batch[0] (data_batch),")
    print("losing y_w, y_l, w, node_mask, gids!")

if __name__ == "__main__":
    main()