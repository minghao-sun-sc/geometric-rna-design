#!/usr/bin/env python3
"""
Find which test structure corresponds to sample0 and get its native PDB.
"""

import os
import sys
sys.path.insert(0, '/mnt/rna01/smh/projects/ribopo')
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import torch
from dpo.utils import load_processed_pt

def find_test_structure_info():
    """Find the native PDB for sample0 in the test set."""
    
    # Load the data split
    tr, va, te = torch.load('data/das_split.pt', map_location="cpu")
    test_indices = list(map(int, te))
    
    # Load processed data  
    all_items = load_processed_pt('data/processed.pt')
    
    print(f"Test set has {len(test_indices)} structures")
    print(f"Looking at first 8 test structures (sample0 = first one):")
    
    for i, idx in enumerate(test_indices[:8]):
        item = all_items[idx]
        print(f"\nSample {i} (idx {idx}):")
        print(f"  Sequence: {item['sequence'][:50]}...")
        print(f"  ID list: {item.get('id_list', 'N/A')}")
        print(f"  Length: {len(item['sequence'])}")
        
        if i == 0:  # This is sample0
            native_ids = item.get('id_list', [])
            print(f"\n=== SAMPLE 0 (our target) ===")
            print(f"Native structure IDs: {native_ids}")
            
            # Check if native PDBs exist
            for nid in native_ids:
                native_path = f"data/raw/{nid}.pdb"
                exists = os.path.exists(native_path)
                print(f"  {nid}.pdb: {'✅ EXISTS' if exists else '❌ MISSING'}")
                if exists:
                    return native_path
    
    return None

if __name__ == "__main__":
    native_pdb = find_test_structure_info()
    if native_pdb:
        print(f"\n🎯 Found native PDB for sample0: {native_pdb}")
    else:
        print("\n❌ No native PDB found for sample0")