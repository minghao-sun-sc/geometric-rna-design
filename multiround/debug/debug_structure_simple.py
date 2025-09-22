#!/usr/bin/env python3
"""
Simple debug script to check the problematic structure 1H4S_1_T.pdb
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import json

def check_structure():
    """Check the structure 1H4S_1_T.pdb for issues."""
    pdb_file = "./data/raw/1H4S_1_T.pdb"
    
    print(f"🔍 Analyzing problematic structure: {pdb_file}")
    
    try:
        from src.data.data_utils import pdb_to_tensor
        
        # Load the structure
        print("📂 Loading PDB structure...")
        seq, coords, _ = pdb_to_tensor(pdb_file, return_sec_struct=False)
        print(f"✅ Structure loaded successfully")
        print(f"   Native sequence length: {len(seq)}")
        print(f"   Native sequence: {seq}")
        
        # Check a few preference pairs with this structure
        pair_file = "data/pairs_margin25/by_das/clean/train.clean.jsonl"
        problematic_pairs = []
        
        print(f"\n📋 Checking preference pairs for {pdb_file}...")
        with open(pair_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if data["pdb_file"] == pdb_file:
                    problematic_pairs.append((i, data))
                    if len(problematic_pairs) >= 5:  # Check first 5 pairs
                        break
        
        print(f"   Found {len(problematic_pairs)} pairs with this structure")
        
        for idx, pair in problematic_pairs:
            winner_seq = pair["winner_seq"]
            loser_seq = pair["loser_seq"]
            
            print(f"\n   Pair {idx}:")
            print(f"     Winner length: {len(winner_seq)} {'✅' if len(winner_seq) == len(seq) else '❌'}")
            print(f"     Loser length: {len(loser_seq)} {'✅' if len(loser_seq) == len(seq) else '❌'}")
            
            if len(winner_seq) != len(seq):
                print(f"     Winner seq: {winner_seq}")
                print(f"     Native seq: {seq}")
                print(f"     Length diff: {len(winner_seq) - len(seq)}")
                
            if len(loser_seq) != len(seq):
                print(f"     Loser seq:  {loser_seq}")
                print(f"     Native seq: {seq}")
                print(f"     Length diff: {len(loser_seq) - len(seq)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def check_dataset_loading():
    """Test loading the actual dataset around index 7183."""
    print(f"\n🔄 Testing dataset loading around index 7183...")
    
    try:
        from dpo.data import DPOPairDataset
        
        # Create dataset
        dataset = DPOPairDataset(
            pairs_path="data/pairs_margin25/by_das/clean/train.clean.jsonl",
            processed_pt_path="data/processed.pt",
            featurizer_cfg={
                "split": "train",
                "radius": 0.0,
                "top_k": 32,
                "num_rbf": 32,
                "num_posenc": 32,
                "max_num_conformers": 1,
                "noise_scale": 0.1,
                "distance_eps": 0.001,
                "device": "cpu"
            },
            split_name="train",
            device="cpu"
        )
        
        print(f"✅ Dataset created with {len(dataset)} pairs")
        
        # Test around the problematic index
        test_indices = [7183 - 2, 7183 - 1, 7183, 7183 + 1, 7183 + 2]
        
        for idx in test_indices:
            try:
                print(f"\n🎯 Testing index {idx}...")
                sample = dataset[idx]
                print(f"   ✅ Success: {sample.cid}")
                print(f"   Winner seq length: {len(sample.winner_seq)}")
                print(f"   Loser seq length: {len(sample.loser_seq)}")
                print(f"   Graph seq length: {len(sample.graph.seq)}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 Simple Structure Debug Script")
    print("=" * 50)
    
    check_structure()
    check_dataset_loading()
    
    print("\n" + "=" * 50)
    print("🏁 Debug completed")