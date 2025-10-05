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
        
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys")
            print(f"Keys: {list(data.keys())[:10]}...")  # Show first 10 keys
            
            # Look at first few items to understand structure
            print("\n" + "="*50)
            print("INSPECTING FIRST 3 DICTIONARY ITEMS:")
            print("="*50)
            
            keys = list(data.keys())
            for i in range(min(3, len(keys))):
                key = keys[i]
                item = data[key]
                print(f"\nKey: {key}")
                print(f"  Value type: {type(item)}")
                
                if hasattr(item, '__dict__'):
                    print(f"  Attributes: {dir(item)}")
                elif isinstance(item, dict):
                    print(f"  Keys: {list(item.keys())}")
                    for sub_key, sub_value in item.items():
                        if isinstance(sub_value, (list, tuple)) and len(sub_value) > 0:
                            print(f"    {sub_key}: {type(sub_value)} with {len(sub_value)} items")
                            if hasattr(sub_value[0], '__class__'):
                                print(f"      First item type: {type(sub_value[0])}")
                        else:
                            print(f"    {sub_key}: {type(sub_value)}")
                            if isinstance(sub_value, str) and len(sub_value) < 100:
                                print(f"      Value: {sub_value}")
                elif hasattr(item, 'keys'):  # torch geometric data object
                    try:
                        print(f"  Keys: {item.keys}")
                        for sub_key in item.keys:
                            attr = getattr(item, sub_key)
                            if hasattr(attr, 'shape'):
                                print(f"    {sub_key}: {type(attr)} shape {attr.shape}")
                            else:
                                print(f"    {sub_key}: {type(attr)} = {attr}")
                    except:
                        print(f"  Could not access keys")
                else:
                    print(f"  Value: {str(item)[:200]}...")
        
        # Check if keys themselves might be the IDs we're looking for
        print("\n" + "="*50)
        print("CHECKING IF KEYS ARE STRUCTURE IDs:")
        print("="*50)
        
        keys = list(data.keys())
        print(f"Sample keys: {keys[:10]}")
        
        # Check for problematic IDs in keys
        problematic_ids = ['5HCQ_1_2x', '8AGW_1_x', '2DR2_1_B', '4V99_1_Ac', '7M57_1_h-S']
        
        print("\nChecking for problematic IDs in keys:")
        for target_id in problematic_ids:
            if target_id in keys:
                print(f"✅ FOUND: {target_id}")
            else:
                print(f"❌ MISSING: {target_id}")
                # Look for similar keys
                base_pdb = target_id.split('_')[0]
                similar_keys = [k for k in keys if isinstance(k, str) and k.startswith(base_pdb)]
                if similar_keys:
                    print(f"  Similar keys: {similar_keys}")
        
        # Look for ID fields in the values
        print("\n" + "="*50)
        print("SEARCHING FOR ID FIELDS IN VALUES:")
        print("="*50)
        
        found_id_fields = set()
        sample_keys = keys[:10]  # Check first 10 items
        
        for key in sample_keys:
            item = data[key]
            if isinstance(item, dict):
                for sub_key, sub_value in item.items():
                    if 'id' in sub_key.lower():
                        found_id_fields.add(sub_key)
                        print(f"Found ID field '{sub_key}' in key '{key}': {sub_value}")
        
        if found_id_fields:
            print(f"\nFound ID fields: {found_id_fields}")
            
            # Check these fields for problematic IDs
            print("\nChecking ID fields for problematic IDs:")
            for target_id in problematic_ids:
                found = False
                for key in keys:
                    item = data[key]
                    if isinstance(item, dict):
                        for id_field in found_id_fields:
                            if id_field in item:
                                id_value = item[id_field]
                                if isinstance(id_value, (list, tuple)):
                                    if target_id in id_value:
                                        print(f"  ✅ {target_id} found in {key}['{id_field}']")
                                        found = True
                                        break
                                elif target_id == id_value:
                                    print(f"  ✅ {target_id} found in {key}['{id_field}']")
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    print(f"  ❌ {target_id} not found in any ID fields")
        
        return data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    dataset_path = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/processed.pt"
    data = inspect_dataset_structure(dataset_path)
    
    if data and isinstance(data, dict):
        print("\n" + "="*50)
        print("FINAL SUMMARY:")
        print("="*50)
        print(f"Dataset is a dictionary with {len(data)} entries")
        print(f"Keys appear to be: {type(list(data.keys())[0])}")
        print(f"Values appear to be: {type(list(data.values())[0])}")

if __name__ == "__main__":
    main()