#!/usr/bin/env python
"""
Check the vocabulary mapping and identify why we have token 4.
"""

import sys
import torch
import pickle
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    print("=== CHECKING VOCABULARY MAPPINGS ===")
    
    # Load the processed data directly
    processed_path = "data/processed.pt"
    print(f"\nLoading processed data from: {processed_path}")
    
    try:
        processed = torch.load(processed_path, map_location="cpu")
        print(f"✅ Loaded processed data")
        
        # Check what's in the processed data
        if isinstance(processed, dict):
            print(f"Keys in processed data: {list(processed.keys())}")
        elif isinstance(processed, list):
            print(f"Processed data is a list with {len(processed)} items")
            if len(processed) > 0:
                first = processed[0]
                print(f"First item type: {type(first)}")
                if hasattr(first, '__dict__'):
                    print(f"First item attributes: {list(vars(first).keys())}")
                    
                    # Check sequence
                    if hasattr(first, 'seq'):
                        seq = first.seq
                        print(f"\nSequence tensor info:")
                        print(f"  Shape: {seq.shape}")
                        print(f"  Unique values: {torch.unique(seq).tolist()}")
                        print(f"  Min: {seq.min()}, Max: {seq.max()}")
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
    
    # Check the featurizer vocabulary
    print(f"\n=== CHECKING FEATURIZER VOCABULARY ===")
    
    try:
        from src.data.featurizer import RNAGraphFeaturizer
        
        featurizer = RNAGraphFeaturizer()
        print(f"\nFeaturizer vocabulary:")
        print(f"  letter_to_num: {featurizer.letter_to_num}")
        print(f"  num_to_letter: {featurizer.num_to_letter}")
        
        # Check if underscore is in vocabulary
        if "_" in featurizer.letter_to_num:
            print(f"\n⚠️ Underscore '_' is in vocabulary with value: {featurizer.letter_to_num['_']}")
            print(f"   This is used for unknown/non-standard nucleotides")
            
    except Exception as e:
        print(f"❌ Error checking featurizer: {e}")
    
    # Check constants
    print(f"\n=== CHECKING CONSTANTS ===")
    
    try:
        from src.constants import LETTER_TO_NUM, NUM_TO_LETTER
        
        print(f"Constants LETTER_TO_NUM: {LETTER_TO_NUM}")
        print(f"Constants NUM_TO_LETTER: {NUM_TO_LETTER}")
        
        # Check for any value >= 4
        for letter, num in LETTER_TO_NUM.items():
            if num >= 4:
                print(f"\n⚠️ Letter '{letter}' maps to {num} which is >= 4!")
                print(f"   This will cause issues with a 4-class model!")
                
    except ImportError:
        print("No constants module found")
    
    # Check the model's expected vocabulary
    print(f"\n=== MODEL EXPECTATIONS ===")
    print(f"gRNAde model expects:")
    print(f"  0: A (Adenine)")
    print(f"  1: C (Cytosine)")  
    print(f"  2: G (Guanine)")
    print(f"  3: U/T (Uracil/Thymine)")
    print(f"  Model outputs 4 classes (indices 0-3)")
    
    # Check what's in our DPO data module
    print(f"\n=== DPO DATA MODULE VOCABULARY ===")
    
    from dpo.data import PreferencePairDataset
    import yaml
    
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
    
    print(f"Dataset letter_to_num: {dataset.letter_to_num}")
    print(f"Dataset num_to_letter: {dataset.num_to_letter}")
    
    # Check if the dataset is using a different vocabulary than expected
    if "_" in dataset.letter_to_num:
        underscore_val = dataset.letter_to_num["_"]
        print(f"\n❌ PROBLEM IDENTIFIED!")
        print(f"   '_' (unknown/padding) maps to {underscore_val}")
        print(f"   But model only outputs 4 classes (0-3)")
        print(f"   This causes index out of bounds when '_' tokens appear!")
        
        # Check where '_' tokens come from
        print(f"\n   Checking where '_' tokens appear...")
        
        # Look at a few pairs
        import json
        with open(dc["pairs_path_train"], 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                pair = json.loads(line)
                winner = pair["winner_seq"]
                loser = pair["loser_seq"]
                
                if "_" in winner or "_" in loser:
                    print(f"\n   Pair {i} has '_' tokens:")
                    print(f"     Winner: {'_' in winner}")
                    print(f"     Loser: {'_' in loser}")

if __name__ == "__main__":
    main()