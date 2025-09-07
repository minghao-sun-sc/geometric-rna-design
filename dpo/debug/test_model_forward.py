#!/usr/bin/env python
"""
Debug script to test model forward pass with a single batch.
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
from hydra.utils import instantiate

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
    
    # Create dataloader with batch_size=1
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pairs,
    )
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg["model"])
    model = model.to(device)
    model.eval()
    
    print(f"Model device: {next(model.parameters()).device}")
    
    # Get one batch and test forward pass
    print("\nTesting forward pass...")
    for batch in dataloader:
        print(f"Batch type: {type(batch)}")
        if hasattr(batch, 'y_w'):
            print(f"y_w shape: {batch.y_w.shape}, device: {batch.y_w.device}")
            print(f"y_l shape: {batch.y_l.shape}, device: {batch.y_l.device}")
            print(f"node_mask shape: {batch.node_mask.shape}, device: {batch.node_mask.device}")
            print(f"y_w values: {batch.y_w}")
            print(f"y_l values: {batch.y_l}")
        
        # Move batch to device
        print(f"Moving batch to {device}...")
        batch = batch.to(device)
        
        print(f"After moving - y_w device: {batch.y_w.device}")
        print(f"y_w min/max: {batch.y_w.min()}/{batch.y_w.max()}")
        print(f"y_l min/max: {batch.y_l.min()}/{batch.y_l.max()}")
        
        # Check if batch has seq attribute
        print(f"Batch attributes: {[attr for attr in dir(batch) if not attr.startswith('_')]}")
        if hasattr(batch, 'seq'):
            print(f"batch.seq: {batch.seq}")
        else:
            print("batch.seq: NOT FOUND")
        
        # Try forward pass
        try:
            print("Calling model forward...")
            with torch.no_grad():
                # The model expects batch.seq to be set, let's try with winner sequence first
                batch.seq = batch.y_w
                logits = model(batch)
                print(f"Logits shape: {logits.shape}")
                print(f"Logits device: {logits.device}")
                print(f"Logits min/max: {logits.min()}/{logits.max()}")
                
            print("Forward pass successful!")
            
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
        
        break  # Only test first batch

if __name__ == "__main__":
    main()