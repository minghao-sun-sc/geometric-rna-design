#!/usr/bin/env python
"""
Trace the exact batch that causes the size mismatch during training.
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
    
    # Use the same dataloader setup as training
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Same as training
        shuffle=False,  # Disable to get reproducible error
        collate_fn=collate_pairs,
        num_workers=0
    )
    
    print("=== CHECKING TRAINING-SIZE BATCHES ===")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Graphs: {batch.num_graphs}")
        print(f"  batch.seq: {batch.seq.shape} -> {batch.seq.numel()} total")
        print(f"  batch.y_w: {batch.y_w.shape} -> {batch.y_w.numel()} total") 
        print(f"  batch.y_l: {batch.y_l.shape} -> {batch.y_l.numel()} total")
        print(f"  batch.node_mask: {batch.node_mask.shape} -> {batch.node_mask.numel()} total")
        
        seq_total = batch.seq.numel()
        y_w_total = batch.y_w.numel()
        y_l_total = batch.y_l.numel()
        mask_total = batch.node_mask.numel()
        
        if seq_total == y_w_total == y_l_total == mask_total:
            print(f"  ✅ All aligned: {seq_total} elements")
        else:
            print(f"  ❌ MISMATCH FOUND!")
            print(f"    seq: {seq_total}")
            print(f"    y_w: {y_w_total}")
            print(f"    y_l: {y_l_total}")
            print(f"    mask: {mask_total}")
            
            print(f"\n  === TRACING INDIVIDUAL ITEMS IN THIS BATCH ===")
            # Get the raw batch items to trace the issue
            start_idx = batch_idx * 16
            items = [dataset[start_idx + i] for i in range(min(16, len(dataset) - start_idx))]
            
            cumulative_seq = 0
            cumulative_y_w = 0
            cumulative_mask = 0
            
            for i, (data, y_w, y_l, weight, node_mask, gid) in enumerate(items):
                graph_len = data.seq.numel()
                target_len = y_w.numel()
                mask_len = node_mask.numel() if node_mask is not None else target_len
                
                cumulative_seq += graph_len
                cumulative_y_w += target_len
                cumulative_mask += mask_len
                
                print(f"    Item {i} ({gid}): graph={graph_len}, target={target_len}, mask={mask_len}")
                
                if graph_len != target_len:
                    print(f"      ❌ GRAPH-TARGET mismatch!")
                if mask_len != target_len:
                    print(f"      ❌ MASK-TARGET mismatch!")
            
            print(f"\n  Cumulative totals:")
            print(f"    seq: {cumulative_seq} (expected: {seq_total})")
            print(f"    y_w: {cumulative_y_w} (expected: {y_w_total})")
            print(f"    mask: {cumulative_mask} (expected: {mask_total})")
            
            if cumulative_mask != mask_total:
                print(f"  ❌ COLLATE FUNCTION ERROR: Individual masks sum to {cumulative_mask} but collated mask has {mask_total}")
            
            break  # Stop at first problematic batch
        
        # Check first few batches only
        if batch_idx >= 3:
            break
    
    print(f"\nCompleted batch analysis")

if __name__ == "__main__":
    main()