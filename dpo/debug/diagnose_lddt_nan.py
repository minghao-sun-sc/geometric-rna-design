#!/usr/bin/env python3
"""
Diagnose why lDDT returns NaN in evaluation
"""

import sys
import os
import glob

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def diagnose_lddt_nan():
    print("🔬 Diagnosing lDDT NaN Issues")
    print("=" * 70)
    
    # Load test dataset info
    import torch
    
    # Get test structure IDs
    processed_pt = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    split_pt = "/mnt/rna01/smh/projects/ribopo/data/das_split.pt"
    
    all_items = torch.load(processed_pt, map_location='cpu')
    _, _, test_indices = torch.load(split_pt, map_location='cpu')
    test_indices = list(map(int, test_indices))
    
    print(f"📊 Analyzing {len(test_indices)} test structures")
    
    # Check first 5 test items
    data_path = "/mnt/rna01/smh/projects/ribopo/data"
    raw_path = os.path.join(data_path, "raw")
    
    exact_matches = 0
    alt_matches = 0
    no_matches = 0
    
    print("\n🔍 Checking PDB file availability for test structures:")
    print("-" * 60)
    
    for i, idx in enumerate(test_indices[:10]):  # Check first 10
        # all_items is a list, access by position not index
        if isinstance(all_items, dict):
            item = all_items[idx]
        else:
            # Find item by iterating
            item = None
            for potential_item in all_items:
                if potential_item.get('index', -1) == idx or all_items.index(potential_item) == idx:
                    item = potential_item
                    break
            if item is None and idx < len(all_items):
                item = all_items[idx]
        id_list = item.get('id_list', [])
        
        if id_list:
            struct_id = id_list[0]
            print(f"\n   Structure {idx}: {struct_id}")
            
            # Check exact match
            exact_pdb = os.path.join(raw_path, f"{struct_id}.pdb")
            if os.path.exists(exact_pdb):
                print(f"      ✅ Exact match: {struct_id}.pdb")
                exact_matches += 1
            else:
                # Check alternatives
                pattern = os.path.join(raw_path, f"{struct_id}*.pdb")
                alternatives = glob.glob(pattern)
                
                if alternatives:
                    alt_names = [os.path.basename(a) for a in alternatives]
                    print(f"      ⚠️ Alternative: {alt_names[0]}")
                    alt_matches += 1
                    
                    # Show why it doesn't match
                    print(f"         Expected: {struct_id}.pdb")
                    print(f"         Found:    {alt_names[0]}")
                else:
                    print(f"      ❌ No PDB file found")
                    no_matches += 1
    
    print("\n" + "=" * 70)
    print("📈 Summary (first 10 test structures):")
    print(f"   ✅ Exact matches:      {exact_matches}/10")
    print(f"   ⚠️ Alternative names:  {alt_matches}/10") 
    print(f"   ❌ Missing:            {no_matches}/10")
    
    if alt_matches > exact_matches:
        print("\n⚠️ ISSUE IDENTIFIED:")
        print("   Most PDB files have non-standard naming (e.g., -A, -B suffixes)")
        print("   This causes lDDT to return NaN because files aren't found")
        
        print("\n💡 SOLUTION:")
        print("   We need to update the lDDT lookup logic to handle alternative naming")
        print("   Options:")
        print("   1. Create symlinks with expected names")
        print("   2. Update get_lddt to search for alternative patterns")
        print("   3. Fix the id_list in processed data to match actual files")
    
    # Check a specific problematic case
    print("\n🔍 Deep dive on a specific case:")
    test_item = all_items[test_indices[0]]
    print(f"   First test item ID: {test_item.get('id_list', ['N/A'])[0]}")
    print(f"   Full id_list: {test_item.get('id_list', [])[:3]}")
    
    # Try to match the pattern
    if test_item.get('id_list'):
        for struct_id in test_item['id_list'][:3]:
            exact = os.path.join(raw_path, f"{struct_id}.pdb")
            pattern = os.path.join(raw_path, f"{struct_id}*.pdb")
            alts = glob.glob(pattern)
            
            print(f"\n   {struct_id}:")
            print(f"      Looking for: {exact}")
            print(f"      Exists: {os.path.exists(exact)}")
            if alts:
                print(f"      Alternatives found: {[os.path.basename(a) for a in alts]}")

if __name__ == "__main__":
    diagnose_lddt_nan()