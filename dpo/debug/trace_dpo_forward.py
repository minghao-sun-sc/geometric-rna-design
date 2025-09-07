#!/usr/bin/env python
"""
Debug script to trace exactly what happens in DPO loss computation.
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

def trace_model_forward(model, batch, seq_name, target_seq):
    """Trace what happens when we call model with different sequences."""
    print(f"\n=== Testing {seq_name} ===")
    print(f"Target sequence: {target_seq}")
    print(f"  Shape: {target_seq.shape}")
    print(f"  Min/Max: {target_seq.min()}/{target_seq.max()}")
    print(f"  Values: {target_seq.tolist()}")
    
    # Save original seq
    orig_seq = batch.seq.clone()
    print(f"Original batch.seq: {orig_seq}")
    print(f"  Shape: {orig_seq.shape}")
    print(f"  Min/Max: {orig_seq.min()}/{orig_seq.max()}")
    
    try:
        # Set target as batch.seq 
        batch.seq = target_seq
        print(f"Set batch.seq = target_seq")
        
        # Try forward pass
        print("Calling model(batch)...")
        logits = model(batch)
        print(f"SUCCESS! Logits shape: {logits.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original seq
        batch.seq = orig_seq

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
        strict_length_check=bool(dc.get("strict_length_check", False)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg["model"])
    model = model.to(device)
    model.eval()
    
    # Get a batch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)
    
    for batch in dataloader:
        print(f"Got batch with {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
        batch = batch.to(device)
        
        # Test 1: Original sequence (should work)
        success1 = trace_model_forward(model, batch, "ORIGINAL", batch.seq.clone())
        
        # Test 2: Winner sequence  
        success2 = trace_model_forward(model, batch, "WINNER", batch.y_w)
        
        # Test 3: Loser sequence
        success3 = trace_model_forward(model, batch, "LOSER", batch.y_l)
        
        print(f"\n=== SUMMARY ===")
        print(f"Original seq works: {success1}")
        print(f"Winner seq works: {success2}")  
        print(f"Loser seq works: {success3}")
        
        if not success2 or not success3:
            print("\n=== DIAGNOSIS ===")
            print("The issue is that winner/loser sequences don't match the graph structure")
            print("that the model expects. The model was trained with autoregressive masking")
            print("where seq[i] predicts logits for position i+1.")
            print("Setting batch.seq to arbitrary target sequences breaks this assumption.")
            
        break

if __name__ == "__main__":
    main()