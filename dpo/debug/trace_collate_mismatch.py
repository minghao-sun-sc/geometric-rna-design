#!/usr/bin/env python
"""
Trace the exact source of the collate/mask size mismatch.
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

from dpo.data import PreferencePairDataset, collate_pairs
from torch.utils.data import DataLoader

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
    
    print("=== TRACING INDIVIDUAL DATASET ITEMS ===")
    
    # Check individual items before collation
    for i in range(5):
        data, y_w, y_l, weight, node_mask, gid = dataset[i]
        
        print(f"\nItem {i} (gid: {gid}):")
        print(f"  Graph seq: {data.seq.shape} -> {data.seq.numel()} tokens")
        print(f"  Target y_w: {y_w.shape} -> {y_w.numel()} tokens")
        print(f"  Target y_l: {y_l.shape} -> {y_l.numel()} tokens")
        print(f"  Node mask: {node_mask.shape if node_mask is not None else 'None'} -> {node_mask.numel() if node_mask is not None else 'N/A'} elements")
        
        # Check if mask is aligned with graph or targets
        graph_len = data.seq.numel()
        target_len = y_w.numel()
        mask_len = node_mask.numel() if node_mask is not None else graph_len
        
        if graph_len != target_len:
            print(f"  ❌ GRAPH-TARGET MISMATCH: graph={graph_len}, target={target_len}")
        if mask_len != graph_len:
            print(f"  ❌ GRAPH-MASK MISMATCH: graph={graph_len}, mask={mask_len}")
        if mask_len != target_len:
            print(f"  ❌ TARGET-MASK MISMATCH: target={target_len}, mask={mask_len}")
            
        if graph_len == target_len == mask_len:
            print(f"  ✅ All aligned: {graph_len} elements")
    
    print(f"\n=== TRACING COLLATE FUNCTION ===")
    
    # Create a small batch to trace collation
    batch_items = [dataset[i] for i in range(3)]
    
    print("Before collation:")
    for i, (data, y_w, y_l, weight, node_mask, gid) in enumerate(batch_items):
        print(f"  Item {i}: graph={data.seq.numel()}, y_w={y_w.numel()}, y_l={y_l.numel()}, mask={node_mask.numel() if node_mask is not None else 'None'}")
    
    # Apply collate function
    try:
        batch = collate_pairs(batch_items)
        print(f"\nAfter collation:")
        print(f"  Batch graphs: {batch.num_graphs}")
        print(f"  batch.seq: {batch.seq.shape}")
        print(f"  batch.y_w: {batch.y_w.shape}")  
        print(f"  batch.y_l: {batch.y_l.shape}")
        print(f"  batch.node_mask: {batch.node_mask.shape}")
        
        # Check the alignment
        seq_total = batch.seq.numel()
        y_w_total = batch.y_w.numel()
        y_l_total = batch.y_l.numel()
        mask_total = batch.node_mask.numel()
        
        print(f"\nAlignment check:")
        print(f"  seq total: {seq_total}")
        print(f"  y_w total: {y_w_total}")
        print(f"  y_l total: {y_l_total}")
        print(f"  mask total: {mask_total}")
        
        if seq_total == y_w_total == y_l_total == mask_total:
            print(f"  ✅ All aligned at batch level")
        else:
            print(f"  ❌ BATCH LEVEL MISMATCH!")
            
        # Trace the exact discrepancy
        print(f"\n=== DETAILED TRACE ===")
        print(f"Expected: seq({seq_total}) == y_w({y_w_total}) == y_l({y_l_total}) == mask({mask_total})")
        
        if seq_total != mask_total:
            print(f"The model will process {seq_total} tokens but mask has {mask_total} elements")
            print("This suggests the mask is not properly aligned with the graph structure")
    
    except Exception as e:
        print(f"Error in collation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()