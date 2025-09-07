#!/usr/bin/env python
"""
Analyze the mismatch between graph structure and target sequences.
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
from hydra.utils import instantiate

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
    
    # Test the problematic case
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_pairs)
    
    device = torch.device("cuda")
    model = instantiate(cfg["model"]).to(device)
    
    print("Analyzing graph-sequence mismatches...")
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # Get model output
        logits = model(batch)
        
        print(f"\n=== BATCH ANALYSIS ===")
        print(f"Graphs: {batch.num_graphs}")
        print(f"Model input (batch.seq): shape={batch.seq.shape}, range={batch.seq.min()}-{batch.seq.max()}")
        print(f"Model output (logits): shape={logits.shape}")
        print(f"Target y_w: shape={batch.y_w.shape}, range={batch.y_w.min()}-{batch.y_w.max()}")
        print(f"Target y_l: shape={batch.y_l.shape}, range={batch.y_l.min()}-{batch.y_l.max()}")
        print(f"Node mask: shape={batch.node_mask.shape}")
        
        # The issue: logits/batch.seq have one length, targets have another
        seq_len = batch.seq.shape[0]  # What model actually processed
        target_len = batch.y_w.shape[0]  # What DPO expects to evaluate
        mask_len = batch.node_mask.shape[0]  # What the collator thinks we have
        
        print(f"\n=== SIZE ANALYSIS ===")
        print(f"Model processed: {seq_len} tokens")
        print(f"DPO targets: {target_len} tokens")  
        print(f"Node mask: {mask_len} tokens")
        
        if seq_len != target_len:
            print(f"❌ MISMATCH: Model processed {seq_len} but DPO expects {target_len}")
            
            # The correct approach: Use model's actual sequence space
            print(f"\n=== CORRECT APPROACH ===")
            print(f"1. Model outputs logits for positions it actually processed: {seq_len}")
            print(f"2. We should evaluate DPO loss on these {seq_len} positions")
            print(f"3. Target sequences need to be mapped to this {seq_len}-dimensional space")
            
            # Show the fundamental issue
            print(f"\n=== FUNDAMENTAL ISSUE ===")
            print("The model's forward pass uses batch.seq (graph-aligned sequence)")
            print("But DPO tries to evaluate y_w/y_l (raw target sequences)")
            print("These don't have the same correspondence to graph nodes!")
            
        break

if __name__ == "__main__":
    main()