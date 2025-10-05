#!/usr/bin/env python3
"""
Inspect the actual structure of the processed dataset to understand the data format.
"""

import torch
import pickle

def inspect_dataset_structure(filepath):
    """Inspect the structure of the processed dataset."""
    print(f"Loading dataset from: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        print(f"Successfully loaded dataset with {len(data)} items")
        print(f"Dataset type: {type(data)}")
        
        # If the data is a dict, inspect its structure
        if isinstance(data, dict):
            print(f"\nDataset is a dictionary with keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)} - length {len(value) if hasattr(value, '__len__') else 'N/A'}")
            
            # If there's a list-like structure, inspect the first few items
            for key, value in data.items():
                if hasattr(value, '__len__') and len(value) > 0:
                    print(f"\nInspecting {key} content:")
                    if hasattr(value, '__getitem__'):
                        try:
                            first_item = value[0]
                            print(f"  First item type: {type(first_item)}")
                            if isinstance(first_item, dict):
                                print(f"  First item keys: {list(first_item.keys())}")
                            elif hasattr(first_item, '__dict__'):
                                print(f"  First item attributes: {list(first_item.__dict__.keys())}")
                            else:
                                print(f"  First item: {first_item}")
                        except (IndexError, KeyError, TypeError) as e:
                            print(f"  Could not access first item: {e}")
        
        # If the data is a list, use original logic
        elif hasattr(data, '__len__') and len(data) > 0:
            print(f"\nFirst item type: {type(data[0])}")
            
            # If it's a dict, show keys
            if isinstance(data[0], dict):
                print(f"First item keys: {list(data[0].keys())}")
                print(f"First item structure:")
                for key, value in data[0].items():
                    print(f"  {key}: {type(value)} - {value if not torch.is_tensor(value) else f'tensor shape {value.shape}'}")
            
            # If it's a tensor or other object, inspect further
            elif hasattr(data[0], '__dict__'):
                print(f"First item attributes: {list(data[0].__dict__.keys())}")
            else:
                print(f"First item: {data[0]}")
            
            # Check a few more items to see if structure is consistent
            print(f"\nChecking structure consistency across items...")
            for i in range(min(5, len(data))):
                item = data[i]
                if isinstance(item, dict):
                    print(f"Item {i} keys: {list(item.keys())}")
                else:
                    print(f"Item {i} type: {type(item)}")
        
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_id_fields(data, max_items=10):
    """Search for fields that might contain structure IDs."""
    print(f"\nSearching for ID fields...")
    
    potential_id_fields = set()
    
    # Handle dictionary dataset structure
    if isinstance(data, dict):
        print("Dataset is a dictionary, inspecting top-level keys...")
        for key, value in data.items():
            if 'id' in key.lower():
                potential_id_fields.add(key)
                print(f"Top-level potential ID field '{key}': {type(value)}")
                if hasattr(value, '__len__'):
                    print(f"  Length: {len(value)}")
                    if hasattr(value, '__getitem__') and len(value) > 0:
                        print(f"  First few items: {value[:min(5, len(value))]}")
        
        # Also check if any values are lists of items with ID fields
        for key, value in data.items():
            if hasattr(value, '__getitem__') and hasattr(value, '__len__') and len(value) > 0:
                try:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        print(f"\nInspecting items in '{key}' for ID fields...")
                        for item_key, item_value in first_item.items():
                            if 'id' in item_key.lower():
                                potential_id_fields.add(f"{key}.{item_key}")
                                print(f"  Item field '{item_key}': {item_value}")
                except (IndexError, KeyError, TypeError):
                    pass
    
    # Handle list dataset structure (original logic)
    elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
        for i, item in enumerate(data[:max_items]):
            if isinstance(item, dict):
                for key, value in item.items():
                    if 'id' in key.lower():
                        potential_id_fields.add(key)
                        print(f"Item {i} - potential ID field '{key}': {value}")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        # Check if it looks like a list of IDs
                        first_val = value[0]
                        if isinstance(first_val, str) and len(first_val) > 3:
                            potential_id_fields.add(key)
                            print(f"Item {i} - potential ID list '{key}': {value}")
    
    return potential_id_fields

def main():
    dataset_path = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/processed.pt"
    
    print("="*60)
    print("INSPECTING DATASET STRUCTURE")
    print("="*60)
    
    # Load and inspect the dataset
    data = inspect_dataset_structure(dataset_path)
    if data is None:
        return
    
    # Search for ID fields
    potential_id_fields = find_id_fields(data)
    
    print(f"\nPotential ID fields found: {potential_id_fields}")
    
    # If we find potential ID fields, extract all IDs
    if potential_id_fields:
        all_ids = set()
        print(f"\nExtracting IDs from dataset...")
        
        # Handle dictionary dataset structure
        if isinstance(data, dict):
            for field in potential_id_fields:
                if '.' in field:  # nested field like "data_list.id_list"
                    parts = field.split('.')
                    if parts[0] in data:
                        items = data[parts[0]]
                        if hasattr(items, '__getitem__') and hasattr(items, '__len__'):
                            for item in items:
                                if isinstance(item, dict) and parts[1] in item:
                                    value = item[parts[1]]
                                    if isinstance(value, str):
                                        all_ids.add(value)
                                    elif isinstance(value, (list, tuple)):
                                        for v in value:
                                            if isinstance(v, str):
                                                all_ids.add(v)
                elif field in data:  # top-level field
                    value = data[field]
                    if isinstance(value, str):
                        all_ids.add(value)
                    elif isinstance(value, (list, tuple)):
                        for v in value:
                            if isinstance(v, str):
                                all_ids.add(v)
        
        # Handle list dataset structure
        elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            for item in data:
                if isinstance(item, dict):
                    for field in potential_id_fields:
                        if field in item:
                            value = item[field]
                            if isinstance(value, str):
                                all_ids.add(value)
                            elif isinstance(value, (list, tuple)):
                                for v in value:
                                    if isinstance(v, str):
                                        all_ids.add(v)
        
        print(f"Total unique IDs found: {len(all_ids)}")
        print(f"First 20 IDs: {sorted(list(all_ids))[:20]}")
        
        # Save the IDs to a file for analysis
        output_file = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/available_structure_ids.txt"
        with open(output_file, 'w') as f:
            for id_val in sorted(all_ids):
                f.write(f"{id_val}\n")
        print(f"\nAll IDs saved to: {output_file}")
        
        return all_ids
    
    return set()

if __name__ == "__main__":
    main()