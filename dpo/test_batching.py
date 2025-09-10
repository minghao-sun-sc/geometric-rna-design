#!/usr/bin/env python3
"""
Test script to verify batching implementation works correctly.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import torch
import yaml
from types import SimpleNamespace as SN

from dpo.data import build_dataloaders
from dpo.ref_manager import build_policy_and_reference
from dpo.losses import dpo_step_losses


def _to_sn(o):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o


def test_batching():
    """Test that batching works correctly with different batch sizes."""
    
    print("Testing batching implementation...")
    print("=" * 60)
    
    # Load config
    with open("dpo/configs/defaults.yaml", "r") as f:
        cfg = _to_sn(yaml.safe_load(f))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        print(f"\n--- Testing batch_size={batch_size} ---")
        
        # Update config
        cfg.training.batch_size = batch_size
        
        # Build dataloaders
        train_loader, _, _, meta = build_dataloaders(cfg, device=device)
        print(f"Dataset size: {meta['n_train']} pairs")
        
        # Build models
        policy, reference = build_policy_and_reference(cfg, device)
        
        # Test a few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            
            print(f"\n  Batch {i+1}:")
            
            # Check batch structure
            if hasattr(batch, 'graph'):
                if hasattr(batch.graph, 'num_graphs'):
                    # Batched graph
                    num_graphs = batch.graph.num_graphs
                    print(f"    Number of graphs in batch: {num_graphs}")
                else:
                    # Single graph
                    print(f"    Single graph (no batching)")
                    num_graphs = 1
                
                # Check sequence dimensions
                if batch.winner_seq.dim() == 2:
                    print(f"    Winner seq shape: {batch.winner_seq.shape} (batched)")
                    print(f"    Loser seq shape: {batch.loser_seq.shape} (batched)")
                else:
                    print(f"    Winner seq shape: {batch.winner_seq.shape} (single)")
                    print(f"    Loser seq shape: {batch.loser_seq.shape} (single)")
                
                # Check CIDs
                if isinstance(batch.cid, list):
                    print(f"    CIDs: {len(batch.cid)} items")
                else:
                    print(f"    CID: {batch.cid}")
            
            # Test forward pass
            try:
                out = dpo_step_losses(
                    model=policy,
                    ref_model=reference,
                    batch=batch,
                    beta=cfg.dpo.beta,
                    label_smoothing=cfg.dpo.label_smoothing,
                    max_len=cfg.dpo.max_len
                )
                
                loss = out["loss_dpo"]
                pref_acc = out["pref_acc"]
                
                print(f"    Loss: {loss.item():.4f}")
                print(f"    Pref acc: {pref_acc.item():.4f}")
                
                # Test backward pass
                loss.backward()
                print(f"    Backward pass: SUCCESS")
                
            except Exception as e:
                print(f"    ERROR in forward/backward pass: {e}")
                raise
    
    print("\n" + "=" * 60)
    print("All tests passed! Batching implementation works correctly.")


if __name__ == "__main__":
    test_batching()