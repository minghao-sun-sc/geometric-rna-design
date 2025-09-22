#!/usr/bin/env python3
"""
Debug script to reproduce and fix the DataLoader issue at index 7183.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import json
import torch
from types import SimpleNamespace as SN
from pathlib import Path

def test_specific_indices():
    """Test specific indices around 7183 to identify the issue."""
    print("🔍 Testing DataLoader issue around index 7183...")
    
    # Load problematic region from preference pairs
    pair_file = "data/pairs_margin25/by_das/clean/train.clean.jsonl"
    print(f"📂 Loading pairs from: {pair_file}")
    
    # Read lines around the problematic index
    start_idx = max(0, 7183 - 5)
    end_idx = 7183 + 15
    
    pairs = []
    with open(pair_file, 'r') as f:
        for i, line in enumerate(f):
            if start_idx <= i <= end_idx:
                pairs.append((i, json.loads(line.strip())))
                
    print(f"📊 Loaded {len(pairs)} pairs from indices {start_idx} to {end_idx}")
    
    # Test each pair for potential issues
    for line_idx, pair in pairs:
        pdb_file = pair["pdb_file"]
        winner_seq = pair["winner_seq"]
        loser_seq = pair["loser_seq"]
        
        print(f"\n--- Testing index {line_idx} ---")
        print(f"PDB: {pdb_file}")
        print(f"Winner length: {len(winner_seq)}")
        print(f"Loser length: {len(loser_seq)}")
        
        # Check if PDB file exists
        pdb_path = Path(pdb_file)
        if not pdb_path.exists():
            print(f"❌ PDB file missing: {pdb_file}")
            continue
            
        # Try to load the structure and check sequence length
        try:
            from src.data.data_utils import pdb_to_tensor
            from dpo.featurizer import GraphDatasetFeaturizer
            
            # Load structure 
            seq, coords, _ = pdb_to_tensor(str(pdb_path), return_sec_struct=False)
            print(f"Native sequence length: {len(seq)}")
            
            # Check for length mismatches
            if len(winner_seq) != len(seq):
                print(f"⚠️ Winner sequence length mismatch: {len(winner_seq)} vs {len(seq)}")
            if len(loser_seq) != len(seq):
                print(f"⚠️ Loser sequence length mismatch: {len(loser_seq)} vs {len(seq)}")
                
            # Try featurization
            print("🧪 Testing featurization...")
            
            # Create minimal config for featurizer
            feat_cfg = SN(
                split="train",
                radius=0.0,
                top_k=32,
                num_rbf=32,
                num_posenc=32,
                max_num_conformers=1,
                noise_scale=0.1,
                distance_eps=0.001,
                device="cpu"
            )
            
            # Create featurizer and test
            featurizer = GraphDatasetFeaturizer(feat_cfg, device="cpu")
            
            # Create a mock raw data entry for testing
            raw_data = {
                'id_list': [pdb_path.stem],
                'sequence': seq,
                'coords_list': [coords],
                'sec_struct_list': ['.'] * len(seq),  # Dummy secondary structure
                'sasa_list': [[0.5] * len(seq)],  # Dummy SASA values
                'rfam_list': ['unknown'],
                'eq_class_list': ['unknown'],
                'cluster_structsim0.45': 'unknown'
            }
            
            graph = featurizer(raw_data)
            print(f"✅ Featurization successful, graph sequence length: {len(graph.seq)}")
            
            # Final length check
            if len(winner_seq) != len(graph.seq):
                print(f"❌ CRITICAL: Winner sequence vs graph mismatch: {len(winner_seq)} vs {len(graph.seq)}")
            if len(loser_seq) != len(graph.seq):
                print(f"❌ CRITICAL: Loser sequence vs graph mismatch: {len(loser_seq)} vs {len(graph.seq)}")
                
        except Exception as e:
            print(f"❌ Error processing {pdb_file}: {e}")
            import traceback
            traceback.print_exc()

def test_dataloader_creation():
    """Test creating actual DPOPairDataset to reproduce the error."""
    print("\n🔄 Testing DPOPairDataset creation...")
    
    try:
        from dpo.data import DPOPairDataset
        
        # Create minimal config
        cfg = SN(
            device="cpu",
            paths=SN(
                processed_pt="data/processed.pt",
                split_pt="data/das_split.pt",
                pairs=SN(
                    train="data/pairs_margin25/by_das/clean/train.clean.jsonl",
                    val="data/pairs_margin25/by_das/clean/val.clean.jsonl"
                )
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
            )
        )
        
        print("📦 Creating DPOPairDataset...")
        dataset = DPOPairDataset(cfg, split='train')
        print(f"✅ Dataset created with {len(dataset)} pairs")
        
        # Test accessing the problematic index
        print(f"🎯 Testing access to index 7183...")
        try:
            sample = dataset[7183]
            print(f"✅ Index 7183 accessed successfully")
            print(f"   Structure: {sample.cid}")
            print(f"   Winner seq length: {len(sample.winner_seq)}")
            print(f"   Loser seq length: {len(sample.loser_seq)}")
            print(f"   Graph seq length: {len(sample.graph.seq)}")
        except Exception as e:
            print(f"❌ Error accessing index 7183: {e}")
            import traceback
            traceback.print_exc()
            
        # Test a few indices around it
        for test_idx in range(7180, 7190):
            try:
                sample = dataset[test_idx]
                print(f"✅ Index {test_idx}: {sample.cid}")
            except Exception as e:
                print(f"❌ Index {test_idx}: {e}")
                
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 DataLoader Debug Script")
    print("=" * 50)
    
    test_specific_indices()
    test_dataloader_creation()
    
    print("\n" + "=" * 50)
    print("🏁 Debug script completed")