#!/usr/bin/env python
"""
Test script to verify CUDA multiprocessing fix for DPO training.
This script tests that DataLoader with num_workers > 0 works correctly
after moving CUDA operations from collate functions to trainer.
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_dataloader_multiprocessing():
    """Test that DataLoader with num_workers > 0 works without CUDA errors."""
    
    print("Testing CUDA multiprocessing fix...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Mock config for testing
    from types import SimpleNamespace as SN
    cfg = SN(
        paths=SN(
            pairs=SN(
                train="data/pairs_margin125/by_das/clean/train.clean.jsonl",
                val="data/pairs_margin125/by_das/clean/val.clean.jsonl",
                test="data/pairs_margin125/by_das/clean/test.clean.jsonl"
            ),
            processed_pt="data/processed.pt"
        ),
        featurizer=SN(
            split="train",
            radius=0.0,
            top_k=32,
            num_rbf=32,
            num_posenc=32,
            max_num_conformers=1,
            noise_scale=0.1,
            distance_eps=0.001,
            device="cpu"
        ),
        training=SN(
            batch_size=4,
            num_workers=2,  # Test with workers
            pin_memory=True,
            drop_last=False
        ),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    try:
        # Import data module
        from dpo.data import DPOPairDataset, collate_batch_pairs, _collate_identity
        
        # Create dataset (should stay on CPU)
        print("\n1. Creating dataset (should be on CPU)...")
        dataset = DPOPairDataset(
            pairs_path=cfg.paths.pairs.train,
            processed_pt_path=cfg.paths.processed_pt,
            featurizer_cfg=cfg.featurizer,
            split_name="train",
            device="cpu"  # Force CPU
        )
        print(f"   Dataset device: {dataset.device}")
        assert dataset.device == "cpu", "Dataset should be on CPU!"
        
        # Create DataLoader with workers
        print("\n2. Creating DataLoader with num_workers=2...")
        collate_fn = collate_batch_pairs if cfg.training.batch_size > 1 else _collate_identity
        loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            collate_fn=collate_fn,
            drop_last=cfg.training.drop_last
        )
        print("   DataLoader created successfully")
        
        # Test fetching a batch
        print("\n3. Fetching a batch from DataLoader...")
        batch = next(iter(loader))
        print(f"   Batch fetched successfully!")
        print(f"   Batch graph device: {batch.graph.x.device if hasattr(batch.graph, 'x') else 'N/A'}")
        print(f"   Winner seq device: {batch.winner_seq.device}")
        print(f"   Loser seq device: {batch.loser_seq.device}")
        
        # Verify batch is on CPU
        assert str(batch.winner_seq.device) == "cpu", "Batch should be on CPU from DataLoader!"
        assert str(batch.loser_seq.device) == "cpu", "Batch should be on CPU from DataLoader!"
        
        # Simulate trainer moving to device
        if torch.cuda.is_available():
            print("\n4. Simulating trainer moving batch to CUDA...")
            device = torch.device("cuda")
            batch.graph = batch.graph.to(device)
            batch.winner_seq = batch.winner_seq.to(device)
            batch.loser_seq = batch.loser_seq.to(device)
            print(f"   Winner seq device after transfer: {batch.winner_seq.device}")
            print(f"   Loser seq device after transfer: {batch.loser_seq.device}")
            assert str(batch.winner_seq.device).startswith("cuda"), "Batch should be on CUDA after transfer!"
        
        print("\n✅ SUCCESS: CUDA multiprocessing fix is working correctly!")
        print("   - Dataset stays on CPU")
        print("   - DataLoader with workers doesn't trigger CUDA errors")
        print("   - Trainer can move batch to GPU in main process")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set multiprocessing start method (important for CUDA)
    mp.set_start_method('spawn', force=True)
    
    success = test_dataloader_multiprocessing()
    sys.exit(0 if success else 1)