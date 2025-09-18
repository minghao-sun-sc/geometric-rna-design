#!/usr/bin/env python3
"""
Test lDDT calculation with actual DAS test set data
"""

import sys
import os
import json

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_lddt_with_das_test():
    print("🔬 Testing lDDT with DAS Test Set Data")
    print("=" * 60)
    
    # Load the test set structure IDs
    test_ids_file = "/mnt/rna01/smh/projects/ribopo/data/das_split_raw_data/test_set_structure_ids.txt"
    with open(test_ids_file, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Found {len(test_ids)} test structures")
    
    # Check how many have matching PDB files
    data_path = "/mnt/rna01/smh/projects/ribopo/data"
    raw_path = os.path.join(data_path, "raw")
    
    found_files = []
    missing_files = []
    alternative_files = []
    
    for test_id in test_ids[:10]:  # Check first 10
        # Direct match
        pdb_path = os.path.join(raw_path, f"{test_id}.pdb")
        if os.path.exists(pdb_path):
            found_files.append((test_id, pdb_path))
        else:
            # Check for alternative naming patterns
            import glob
            pattern = os.path.join(raw_path, f"{test_id}*.pdb")
            alternatives = glob.glob(pattern)
            if alternatives:
                alternative_files.append((test_id, alternatives))
            else:
                missing_files.append(test_id)
    
    print(f"\n📁 File Check Results:")
    print(f"   ✅ Direct matches: {len(found_files)}")
    print(f"   ⚠️ Alternative names: {len(alternative_files)}")
    print(f"   ❌ Missing: {len(missing_files)}")
    
    if alternative_files:
        print(f"\n🔍 Alternative file patterns found:")
        for test_id, alts in alternative_files[:5]:
            print(f"   {test_id} → {[os.path.basename(a) for a in alts]}")
    
    # Test lDDT calculation with a working example
    print(f"\n🧪 Testing lDDT Calculation:")
    
    # Use one of the found files
    if found_files:
        test_id, pdb_path = found_files[0]
        print(f"   Using: {test_id} ({pdb_path})")
        
        from src.evaluator import get_lddt
        
        # Self-comparison should give 1.0
        lddt_self = get_lddt(pdb_path, pdb_path)
        print(f"   Self-comparison lDDT: {lddt_self:.4f}")
        
        if lddt_self == 1.0:
            print(f"   ✅ lDDT calculation working!")
        else:
            print(f"   ⚠️ Unexpected self-comparison score")
    
    # Check the actual raw data structure
    print(f"\n📋 Analyzing raw_data structure:")
    import torch
    
    # Load the processed data to see structure
    processed_pt = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    if os.path.exists(processed_pt):
        data = torch.load(processed_pt, map_location='cpu')
        
        # Check first item's id_list
        if data and len(data) > 0:
            item = data[0]
            if 'id_list' in item:
                print(f"   Example id_list: {item['id_list'][:3]}")
            else:
                print(f"   ❌ No id_list in data items")
                print(f"   Available keys: {list(item.keys())}")
    
    print(f"\n🎯 Diagnosis:")
    if len(alternative_files) > len(found_files):
        print(f"   ⚠️ Most files have alternative naming (e.g., -A, -B suffixes)")
        print(f"   💡 Need to handle flexible PDB file naming in lDDT calculation")
    
    return found_files, alternative_files, missing_files

if __name__ == "__main__":
    found, alt, missing = test_lddt_with_das_test()