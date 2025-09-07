#!/usr/bin/env python
"""
Debug script to check model embedding vocabulary vs dataset tokens.
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
        strict_length_check=bool(dc.get("strict_length_check", False)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    print(f"Dataset vocab: {dataset.letter_to_num}")
    print(f"Max token in dataset: {max(dataset.letter_to_num.values())}")
    
    # Load model
    print("Loading model...")
    model = instantiate(cfg["model"])
    
    # Check model embedding
    print(f"Model W_s (sequence embedding): {model.W_s}")
    print(f"Model embedding vocab size: {model.W_s.num_embeddings}")
    print(f"Model embedding dim: {model.W_s.embedding_dim}")
    
    # Get a sample and check token values
    print("\nChecking sample tokens...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)
    
    for batch in dataloader:
        print(f"Original batch.seq: {batch.seq}")
        print(f"  seq min/max: {batch.seq.min()}/{batch.seq.max()}")
        print(f"  seq shape: {batch.seq.shape}")
        
        print(f"y_w: {batch.y_w}")
        print(f"  y_w min/max: {batch.y_w.min()}/{batch.y_w.max()}")
        print(f"  y_w shape: {batch.y_w.shape}")
        
        print(f"y_l: {batch.y_l}")
        print(f"  y_l min/max: {batch.y_l.min()}/{batch.y_l.max()}")
        print(f"  y_l shape: {batch.y_l.shape}")
        
        # Check if any tokens are out of bounds
        embedding_vocab_size = model.W_s.num_embeddings
        
        for name, tensor in [("seq", batch.seq), ("y_w", batch.y_w), ("y_l", batch.y_l)]:
            max_token = tensor.max().item()
            if max_token >= embedding_vocab_size:
                print(f"ERROR: {name} has token {max_token} >= embedding vocab size {embedding_vocab_size}")
            else:
                print(f"OK: {name} max token {max_token} < embedding vocab size {embedding_vocab_size}")
        
        break

if __name__ == "__main__":
    main()