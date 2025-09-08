#!/usr/bin/env python
"""
Trace the vocabulary issue causing CUDA index out of bounds.
"""

import sys
import torch
import yaml
from pathlib import Path
import numpy as np

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
    
    print("=== VOCABULARY ANALYSIS ===")
    
    # Check the dataset vocabulary
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
    
    print(f"\nDataset vocabulary mapping:")
    print(f"letter_to_num: {dataset.letter_to_num}")
    # Create reverse mapping
    num_to_letter = {v: k for k, v in dataset.letter_to_num.items()}
    print(f"num_to_letter: {num_to_letter}")
    
    # Check what tokens are in the data
    print(f"\n=== CHECKING TARGET TOKENS ===")
    
    unique_tokens = set()
    token_counts = {}
    
    for i in range(min(100, len(dataset))):
        data, y_w, y_l, weight, node_mask, gid = dataset[i]
        
        # Check winner tokens
        w_unique = torch.unique(y_w)
        l_unique = torch.unique(y_l)
        
        for tok in w_unique.tolist():
            unique_tokens.add(tok)
            token_counts[tok] = token_counts.get(tok, 0) + 1
            
        for tok in l_unique.tolist():
            unique_tokens.add(tok)
            token_counts[tok] = token_counts.get(tok, 0) + 1
        
        # Also check graph sequence tokens
        seq_unique = torch.unique(data.seq)
        for tok in seq_unique.tolist():
            unique_tokens.add(tok)
    
    print(f"Unique tokens found in first 100 items: {sorted(unique_tokens)}")
    print(f"Token counts: {dict(sorted(token_counts.items()))}")
    
    # Check if token 4 appears
    if 4 in unique_tokens:
        print(f"\n⚠️ WARNING: Token 4 found in data!")
        print(f"  This corresponds to: {dataset.num_to_letter.get(4, 'UNKNOWN')}")
        
        # Find which items have token 4
        items_with_4 = []
        for i in range(min(100, len(dataset))):
            data, y_w, y_l, weight, node_mask, gid = dataset[i]
            if 4 in y_w or 4 in y_l:
                items_with_4.append(i)
                if len(items_with_4) <= 3:
                    print(f"\n  Item {i} ({gid}) has token 4:")
                    if 4 in y_w:
                        positions = (y_w == 4).nonzero(as_tuple=True)[0]
                        print(f"    Winner has token 4 at positions: {positions.tolist()}")
                        # Show the actual sequence around those positions
                        winner_seq = dataset.pairs[i]["winner_seq"]
                        for pos in positions[:3]:
                            if pos < len(winner_seq):
                                print(f"      Position {pos}: '{winner_seq[pos]}' (in sequence: '{winner_seq[max(0,pos-2):min(len(winner_seq),pos+3)]}')")
                    if 4 in y_l:
                        positions = (y_l == 4).nonzero(as_tuple=True)[0]
                        print(f"    Loser has token 4 at positions: {positions.tolist()}")
    
    # Check model output dimension
    print(f"\n=== MODEL OUTPUT DIMENSION ===")
    model = instantiate(cfg["model"])
    print(f"Model output dimension: {model.out_dim}")
    print(f"Expected to handle tokens: 0 to {model.out_dim - 1}")
    
    # Create a small batch to test
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    
    print(f"\n=== BATCH ANALYSIS ===")
    print(f"batch.seq unique values: {torch.unique(batch.seq).tolist()}")
    print(f"batch.y_w unique values: {torch.unique(batch.y_w).tolist()}")
    print(f"batch.y_l unique values: {torch.unique(batch.y_l).tolist()}")
    
    # Check if any values are >= model.out_dim
    max_token = max(torch.max(batch.y_w).item(), torch.max(batch.y_l).item())
    if max_token >= model.out_dim:
        print(f"\n❌ ERROR: Max token {max_token} >= model output dim {model.out_dim}")
        print(f"   This will cause index out of bounds!")
    
    # Test model forward
    print(f"\n=== TESTING MODEL FORWARD ===")
    model = model.to("cpu")
    batch = batch.to("cpu")
    
    try:
        with torch.no_grad():
            logits = model(batch)
        print(f"✅ Model forward successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Expected: [batch_size * seq_len, {model.out_dim}]")
        
        # Check if we can gather with y_w
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Ensure targets are in valid range
        y_w_safe = torch.clamp(batch.y_w, 0, model.out_dim - 1)
        
        if not torch.equal(y_w_safe, batch.y_w):
            n_clamped = (y_w_safe != batch.y_w).sum().item()
            print(f"\n⚠️ Had to clamp {n_clamped} tokens from y_w")
            original_vals = batch.y_w[y_w_safe != batch.y_w].unique()
            print(f"   Original values that were clamped: {original_vals.tolist()}")
        
        gathered = torch.gather(log_probs, dim=-1, index=y_w_safe.unsqueeze(-1))
        print(f"✅ Gather operation successful with clamped targets")
        
    except Exception as e:
        print(f"❌ Error during model forward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()