#!/usr/bin/env python
"""
Test with exact training conditions to reproduce the CUDA error.
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
from dpo.losses import _compute_per_graph_logp

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    dc = cfg["data"]
    
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
        strict_length_check=bool(dc.get("strict_length_check", False)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    # Use training conditions: batch_size > 1, training mode
    dataloader = DataLoader(
        dataset, 
        batch_size=2,  # Same as sanity check
        shuffle=False, 
        collate_fn=collate_pairs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg["model"])
    model = model.to(device)
    model.train()  # Training mode like in actual training
    
    print("Testing with training conditions...")
    
    for batch in dataloader:
        batch = batch.to(device)
        
        print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
        print(f"batch.seq shape: {batch.seq.shape}")
        print(f"batch.y_w shape: {batch.y_w.shape}")  
        print(f"batch.y_l shape: {batch.y_l.shape}")
        print(f"batch.node_mask shape: {batch.node_mask.shape}")
        
        # Test the exact function that's failing
        print("\nTesting _compute_per_graph_logp with winner sequence...")
        try:
            logp_w = _compute_per_graph_logp(
                model, batch, batch.y_w, batch.node_mask, 
                length_norm=False, no_grad=False
            )
            print(f"SUCCESS! logp_w shape: {logp_w.shape}")
            print(f"logp_w values: {logp_w}")
            
        except Exception as e:
            print(f"ERROR in _compute_per_graph_logp: {e}")
            print(f"Error type: {type(e)}")
            
            # Additional debugging
            print(f"\nDEBUG INFO:")
            print(f"batch.seq range: {batch.seq.min()}-{batch.seq.max()}")
            print(f"batch.y_w range: {batch.y_w.min()}-{batch.y_w.max()}")
            print(f"Model W_s vocab size: {model.W_s.num_embeddings}")
            
            # Check if the issue is setting batch.seq
            print(f"\nOriginal batch.seq: {batch.seq}")
            print(f"Setting batch.seq to batch.y_w: {batch.y_w}")
            batch.seq = batch.y_w
            print(f"After setting: batch.seq == batch.y_w? {torch.equal(batch.seq, batch.y_w)}")
            
            import traceback
            traceback.print_exc()
        
        break

if __name__ == "__main__":
    main()