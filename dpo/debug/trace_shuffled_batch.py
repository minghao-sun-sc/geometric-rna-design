#!/usr/bin/env python
"""
Trace batches with shuffling enabled to find the specific mismatch.
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

def detailed_item_analysis(dataset, indices):
    """Analyze specific items in detail"""
    print(f"\n=== DETAILED ANALYSIS OF ITEMS {indices} ===")
    
    items = []
    for i in indices:
        try:
            item = dataset[i]
            items.append((i, item))
            data, y_w, y_l, weight, node_mask, gid = item
            
            graph_len = data.seq.numel()
            target_len = y_w.numel()
            mask_len = node_mask.numel() if node_mask is not None else target_len
            
            print(f"Item {i} ({gid}):")
            print(f"  graph.seq: {graph_len}")
            print(f"  y_w: {target_len}")
            print(f"  mask: {mask_len} ({'None' if node_mask is None else 'Tensor'})")
            
            if graph_len != target_len or mask_len != target_len:
                print(f"  ❌ MISALIGNMENT DETECTED")
                
                # Check if this is a windowed case
                entry = dataset.pairs[i]
                if entry.get("_window") is not None:
                    start, window_len = entry["_window"]
                    print(f"    Windowed case: start={start}, window_len={window_len}")
                    print(f"    Graph length: {graph_len}")
                    print(f"    Target length should be: {target_len}")
                    print(f"    Mask should cover: positions {start} to {start + window_len}")
                
                # Look at the raw data
                raw_idx = entry["_local_index"]
                raw = dataset.raw_list[raw_idx]
                raw_seq_len = len(raw.get("sequence", ""))
                print(f"    Raw sequence length: {raw_seq_len}")
                
        except Exception as e:
            print(f"Error analyzing item {i}: {e}")
            
    return [item[1] for item in items]

def main():
    # Set a specific seed for reproducibility
    torch.manual_seed(42)
    
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
    
    # Use shuffling like in training
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,  # Enable shuffling like training
        collate_fn=collate_pairs,
        num_workers=0
    )
    
    print("=== CHECKING SHUFFLED BATCHES ===")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Graphs: {batch.num_graphs}")
        
        seq_total = batch.seq.numel()
        y_w_total = batch.y_w.numel()
        y_l_total = batch.y_l.numel()
        mask_total = batch.node_mask.numel()
        
        print(f"  seq: {seq_total}, y_w: {y_w_total}, y_l: {y_l_total}, mask: {mask_total}")
        
        if seq_total != y_w_total or seq_total != y_l_total or seq_total != mask_total:
            print(f"  ❌ MISMATCH FOUND IN SHUFFLED BATCH!")
            print(f"    This batch would cause the training error")
            
            # Try to manually trace what's happening
            print(f"\n  === MANUAL COLLATE TRACE ===")
            
            # Can't easily get the exact indices from shuffled batch,
            # but let's look for problematic items in the dataset
            print(f"  Searching for items with mismatches...")
            
            problematic_items = []
            for i in range(min(1000, len(dataset))):  # Check first 1000 items
                try:
                    data, y_w, y_l, weight, node_mask, gid = dataset[i]
                    graph_len = data.seq.numel()
                    target_len = y_w.numel()
                    mask_len = node_mask.numel() if node_mask is not None else target_len
                    
                    if graph_len != target_len or mask_len != target_len:
                        problematic_items.append(i)
                        if len(problematic_items) >= 5:  # Limit to first 5
                            break
                except Exception as e:
                    print(f"    Error checking item {i}: {e}")
            
            if problematic_items:
                print(f"  Found problematic items: {problematic_items}")
                detailed_item_analysis(dataset, problematic_items[:3])
            else:
                print(f"  No individual problematic items found in first 1000...")
            
            break
        else:
            print(f"  ✅ All aligned: {seq_total} elements")
        
        # Check first 10 batches
        if batch_idx >= 9:
            print(f"\nChecked 10 batches, no mismatches found")
            break

if __name__ == "__main__":
    main()