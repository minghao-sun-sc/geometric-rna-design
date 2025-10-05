#!/usr/bin/env python3
"""
Script to inspect the actual structure of the processed dataset.
"""

import torch
import pickle
from pprint import pprint

def inspect_dataset_structure(filepath):
    """Inspect the structure of the processed dataset."""
    print(f"Loading dataset from: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        print(f"Successfully loaded dataset")
        print(f"Type: {type(data)}")
        print(f"Length: {len(data)}")
        
        # Look at first few items to understand structure
        print("\n" + "="*50)
        print("INSPECTING FIRST 3 ITEMS:")
        print("="*50)
        
        for i in range(min(3, len(data))):
            print(f"\nItem {i}:")
            item = data[i]
            print(f"  Type: {type(item)}")
            
            if hasattr(item, '__dict__'):
                print(f"  Attributes: {dir(item)}")
            elif isinstance(item, dict):
                print(f"  Keys: {list(item.keys())}")
                for key, value in item.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        print(f"    {key}: {type(value)} with {len(value)} items, first item type: {type(value[0])}")
                    else:
                        print(f"    {key}: {type(value)} = {value}")
            elif hasattr(item, 'keys'):  # torch geometric data object
                try:
                    print(f"  Keys: {item.keys}")
                    for key in item.keys:
                        attr = getattr(item, key)
                        if hasattr(attr, 'shape'):
                            print(f"    {key}: {type(attr)} shape {attr.shape}")
                        else:
                            print(f"    {key}: {type(attr)} = {attr}")
                except:
                    print(f"  Could not access keys")
            else:
                print(f"  Value: {item}")
        
        # Look for any items with id information
        print("\n" + "="*50)
        print("SEARCHING FOR ID INFORMATION:")
        print("="*50)
        
        found_ids = set()
        for i, item in enumerate(data[:100]):  # Check first 100 items
            if hasattr(item, 'keys'):
                try:
                    for key in item.keys:
                        if 'id' in key.lower():
                            attr = getattr(item, key)
                            print(f"Item {i}: {key} = {attr}")
                            if isinstance(attr, (list, tuple)):
                                found_ids.update(attr)
                            else:
                                found_ids.add(attr)
                except:
                    pass
            elif isinstance(item, dict):
                for key, value in item.items():
                    if 'id' in key.lower():
                        print(f"Item {i}: {key} = {value}")
                        if isinstance(value, (list, tuple)):
                            found_ids.update(value)
                        else:
                            found_ids.add(value)
        
        if found_ids:
            print(f"\nFound {len(found_ids)} unique IDs:")
            for id_val in sorted(found_ids):
                print(f"  {id_val}")
        
        # Try to find any attributes that might contain the problematic IDs
        problematic_ids = ['5HCQ_1_2x', '8AGW_1_x', '2DR2_1_B', '4V99_1_Ac', '7M57_1_h-S']
        
        print("\n" + "="*50)
        print("SEARCHING FOR PROBLEMATIC IDs IN DATASET:")
        print("="*50)
        
        for target_id in problematic_ids:
            print(f"\nSearching for: {target_id}")
            found = False
            
            for i, item in enumerate(data):
                item_str = str(item)
                if target_id in item_str:
                    print(f"  Found in item {i}")
                    found = True
                    break
            
            if not found:
                # Check for partial matches
                base_pdb = target_id.split('_')[0]
                print(f"  Looking for partial matches with {base_pdb}...")
                for i, item in enumerate(data[:50]):  # Check first 50 items
                    item_str = str(item)
                    if base_pdb in item_str:
                        print(f"    Partial match in item {i}: {item_str[:200]}...")
                        break
        
        return data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    dataset_path = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/processed.pt"
    inspect_dataset_structure(dataset_path)

if __name__ == "__main__":
    main()