#!/usr/bin/env python3
"""
Inspect a single entry from the processed dataset to understand its structure.
"""

import torch

def inspect_single_entry():
    """Inspect a single entry from the processed dataset."""
    dataset_path = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/processed.pt"
    
    print("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        data = torch.load(f, map_location='cpu')
    
    print(f"Dataset type: {type(data)}")
    print(f"Dataset length: {len(data)}")
    
    # Get the first key-value pair
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    
    print(f"\nFirst key (sequence): {first_key}")
    print(f"First value type: {type(first_value)}")
    
    if isinstance(first_value, dict):
        print(f"First value keys: {list(first_value.keys())}")
        for k, v in first_value.items():
            print(f"  {k}: {type(v)} - {v if not torch.is_tensor(v) else f'tensor shape {v.shape}'}")
    elif hasattr(first_value, '__dict__'):
        print(f"First value attributes: {list(first_value.__dict__.keys())}")
        for attr in first_value.__dict__.keys():
            val = getattr(first_value, attr)
            print(f"  {attr}: {type(val)} - {val if not torch.is_tensor(val) else f'tensor shape {val.shape}'}")
    else:
        print(f"First value: {first_value}")
    
    # Check a few more entries
    print(f"\nChecking structure of first 3 entries...")
    for i, (key, value) in enumerate(list(data.items())[:3]):
        print(f"\nEntry {i+1}:")
        print(f"  Key (sequence): {key[:50]}..." if len(key) > 50 else f"  Key (sequence): {key}")
        print(f"  Value type: {type(value)}")
        
        if isinstance(value, dict):
            print(f"  Value keys: {list(value.keys())}")
            # Look for ID-related fields
            for k, v in value.items():
                if 'id' in k.lower():
                    print(f"    ID field '{k}': {v}")
        elif hasattr(value, '__dict__'):
            for attr in value.__dict__.keys():
                if 'id' in attr.lower():
                    val = getattr(value, attr)
                    print(f"    ID attribute '{attr}': {val}")

if __name__ == "__main__":
    inspect_single_entry()