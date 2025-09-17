#!/usr/bin/env python3
"""Extract test set structure IDs from DAS split using dataset indices"""

import sys
import os
import json
import torch
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

def load_dataset_and_split():
    """Load both the dataset and the split indices"""
    print("🔍 Loading dataset and split data...")
    
    # Load the processed dataset
    data_path = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    split_path = "/mnt/rna01/smh/projects/ribopo/data/das_split.pt"
    
    try:
        print(f"Loading dataset from: {data_path}")
        dataset = torch.load(data_path, map_location='cpu')
        print(f"✅ Dataset loaded: {type(dataset)}")
        
        print(f"Loading split from: {split_path}")
        split_data = torch.load(split_path, map_location='cpu')
        print(f"✅ Split loaded: {type(split_data)}")
        
        return dataset, split_data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def examine_dataset_structure(dataset, max_items=3):
    """Examine the structure of the dataset"""
    print(f"\n📊 Examining dataset structure:")
    print(f"Dataset type: {type(dataset)}")
    
    if hasattr(dataset, '__len__'):
        print(f"Dataset length: {len(dataset)}")
    
    if isinstance(dataset, dict):
        print(f"Dataset keys: {list(dataset.keys())}")
        for key, value in dataset.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, '__len__'):
                print(f"    Length: {len(value)}")
            if isinstance(value, (list, tuple)) and len(value) > 0:
                print(f"    Sample item type: {type(value[0])}")
                if hasattr(value[0], '__dict__'):
                    attrs = [attr for attr in dir(value[0]) if not attr.startswith('_')]
                    print(f"    Sample attributes: {attrs[:5]}...")
    
    elif hasattr(dataset, 'data_list') or hasattr(dataset, 'data'):
        data_list = getattr(dataset, 'data_list', getattr(dataset, 'data', None))
        if data_list:
            print(f"Data list length: {len(data_list)}")
            
            for i, item in enumerate(data_list[:max_items]):
                print(f"\n--- Dataset Item {i} ---")
                print(f"Type: {type(item)}")
                
                if hasattr(item, '__dict__'):
                    attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                    print(f"Attributes: {attrs[:10]}...")  # Show first 10 attrs
                    
                    # Check for ID fields
                    for attr in ['gid', 'id_list', 'pdb_id', 'structure_id', 'id', 'name']:
                        if hasattr(item, attr):
                            val = getattr(item, attr)
                            print(f"  {attr}: {val}")
                
                elif isinstance(item, dict):
                    print(f"Keys: {list(item.keys())[:10]}...")
                    for key in ['gid', 'id_list', 'pdb_id', 'structure_id', 'id', 'name']:
                        if key in item:
                            print(f"  {key}: {item[key]}")
    
    elif isinstance(dataset, (list, tuple)):
        print(f"Dataset is a {type(dataset).__name__} with {len(dataset)} items")
        for i, item in enumerate(dataset[:max_items]):
            print(f"\n--- Dataset Item {i} ---")
            print(f"Type: {type(item)}")
            if hasattr(item, '__dict__'):
                attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                print(f"Attributes: {attrs[:10]}...")
    
    return dataset

def examine_split_structure(split_data):
    """Examine the split structure"""
    print(f"\n📊 Examining split structure:")
    print(f"Split type: {type(split_data)}")
    
    if isinstance(split_data, (list, tuple)):
        print(f"Split has {len(split_data)} elements:")
        for i, element in enumerate(split_data):
            print(f"  Element {i}: {type(element)} with {len(element) if hasattr(element, '__len__') else 'N/A'} items")
            if hasattr(element, '__len__') and len(element) > 0:
                sample_items = element[:5] if len(element) >= 5 else element
                print(f"    Sample items: {sample_items}")
    
    return split_data

def extract_test_structures(dataset, split_data, split_index=2):
    """Extract test structures using split indices"""
    print(f"\n🔍 Extracting test structures using split index {split_index}...")
    
    # Get test indices
    if not isinstance(split_data, (list, tuple)) or split_index >= len(split_data):
        print(f"❌ Invalid split index {split_index}")
        return []
    
    test_indices = split_data[split_index]
    print(f"Test set has {len(test_indices)} indices")
    print(f"Sample indices: {test_indices[:10]}")
    
    # Get dataset items
    if hasattr(dataset, 'data_list'):
        data_list = dataset.data_list
    elif hasattr(dataset, 'data'):
        data_list = dataset.data
    elif isinstance(dataset, (list, tuple)):
        data_list = dataset
    elif isinstance(dataset, dict):
        # Try to find the data list in dictionary
        possible_keys = ['data_list', 'data', 'items', 'structures']
        data_list = None
        for key in possible_keys:
            if key in dataset:
                data_list = dataset[key]
                print(f"Found data in key: {key}")
                break
        if data_list is None:
            # Use the largest list in the dataset
            largest_key = max(dataset.keys(), key=lambda k: len(dataset[k]) if hasattr(dataset[k], '__len__') else 0)
            data_list = dataset[largest_key]
            print(f"Using largest dataset key: {largest_key}")
    else:
        print(f"❌ Cannot access dataset items from {type(dataset)}")
        return []
    
    print(f"Dataset has {len(data_list)} total items")
    
    # Extract test structures
    test_structures = []
    missing_indices = []
    
    for idx in test_indices:
        if idx < len(data_list):
            test_structures.append(data_list[idx])
        else:
            missing_indices.append(idx)
    
    print(f"✅ Successfully extracted {len(test_structures)} test structures")
    if missing_indices:
        print(f"⚠️ Missing indices: {missing_indices[:10]}... (total: {len(missing_indices)})")
    
    return test_structures

def extract_structure_ids(test_structures):
    """Extract structure IDs from test structures"""
    print(f"\n🔍 Extracting structure IDs from {len(test_structures)} test structures...")
    
    structure_ids = []
    id_sources = Counter()
    
    for i, item in enumerate(test_structures):
        item_ids = []
        
        # Try different ways to get IDs based on item type
        if hasattr(item, 'gid'):
            if item.gid:
                item_ids.append(('gid', item.gid))
                id_sources['gid'] += 1
        
        if hasattr(item, 'id_list'):
            if item.id_list:
                for id_val in item.id_list:
                    item_ids.append(('id_list', id_val))
                    id_sources['id_list'] += 1
        
        if hasattr(item, 'pdb_id'):
            if item.pdb_id:
                item_ids.append(('pdb_id', item.pdb_id))
                id_sources['pdb_id'] += 1
        
        if hasattr(item, 'id'):
            if item.id:
                item_ids.append(('id', item.id))
                id_sources['id'] += 1
        
        # Check if it's a dictionary
        if isinstance(item, dict):
            for key in ['gid', 'id_list', 'pdb_id', 'structure_id', 'id', 'name']:
                if key in item and item[key]:
                    if isinstance(item[key], list):
                        for id_val in item[key]:
                            item_ids.append((key, id_val))
                            id_sources[key] += 1
                    else:
                        item_ids.append((key, item[key]))
                        id_sources[key] += 1
        
        if item_ids:
            structure_ids.extend(item_ids)
        else:
            # Show details for first few items without IDs
            if i < 5:
                print(f"⚠️ No IDs found for item {i}: {type(item)}")
                if hasattr(item, '__dict__'):
                    attrs = [attr for attr in dir(item) if not attr.startswith('_')][:5]
                    print(f"    Available attributes: {attrs}")
                elif isinstance(item, dict):
                    keys = list(item.keys())[:5]
                    print(f"    Available keys: {keys}")
    
    print(f"\n📈 ID source statistics:")
    for source, count in id_sources.most_common():
        print(f"  {source}: {count} IDs")
    
    return structure_ids

def map_to_raw_files(structure_ids):
    """Map structure IDs to raw PDB files"""
    raw_dir = Path("/mnt/rna01/smh/projects/ribopo/data/raw")
    print(f"\n🗂️ Mapping IDs to raw files in: {raw_dir}")
    
    # Get all raw PDB files
    raw_files = list(raw_dir.glob("*.pdb"))
    print(f"Found {len(raw_files)} raw PDB files")
    
    # Create mapping from filenames (without extension)
    raw_basenames = {f.stem: f.name for f in raw_files}
    
    # Sample raw file patterns
    sample_files = list(raw_basenames.keys())[:10]
    print(f"Sample raw file basenames: {sample_files}")
    
    # Try to map structure IDs to raw files
    mapping = {}
    unmapped_ids = []
    id_formats = Counter()
    
    for source, struct_id in structure_ids:
        id_formats[f"{source}_{type(struct_id).__name__}"] += 1
        
        # Convert to string and try different matching strategies
        id_str = str(struct_id)
        
        # Strategy 1: Direct match
        if id_str in raw_basenames:
            mapping[struct_id] = raw_basenames[id_str]
            continue
        
        # Strategy 2: Try without chain/model suffixes
        base_id = id_str.split('_')[0] if '_' in id_str else id_str
        matching_files = [name for basename, name in raw_basenames.items() if basename.startswith(base_id)]
        
        if matching_files:
            mapping[struct_id] = matching_files[0]
            continue
        
        # Strategy 3: Case insensitive match
        id_lower = id_str.lower()
        for basename, filename in raw_basenames.items():
            if basename.lower() == id_lower:
                mapping[struct_id] = filename
                break
        else:
            unmapped_ids.append((source, struct_id))
    
    print(f"\n📊 Mapping Results:")
    print(f"  Successfully mapped: {len(mapping)} IDs")
    print(f"  Unmapped IDs: {len(unmapped_ids)}")
    if len(mapping) + len(unmapped_ids) > 0:
        success_rate = len(mapping)/(len(mapping)+len(unmapped_ids))*100
        print(f"  Mapping success rate: {success_rate:.1f}%")
    
    print(f"\n📋 ID Format Distribution:")
    for fmt, count in id_formats.most_common():
        print(f"  {fmt}: {count}")
    
    if unmapped_ids:
        print(f"\n⚠️ Sample unmapped IDs:")
        for source, uid in unmapped_ids[:10]:
            print(f"  {source}: {uid}")
    
    return mapping, unmapped_ids

def save_results(mapping, unmapped_ids, test_structures, test_indices):
    """Save the mapping results to files"""
    output_dir = Path("/mnt/rna01/smh/projects/ribopo/dpo/debug")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_dir}")
    
    # 1. Test set indices
    indices_file = output_dir / "test_set_indices.txt"
    with open(indices_file, 'w') as f:
        for idx in sorted(test_indices):
            f.write(f"{idx}\n")
    print(f"✅ Saved {len(test_indices)} test indices to: {indices_file}")
    
    # 2. Test set IDs (unique structure IDs)
    unique_ids = list(set(mapping.keys()))
    ids_file = output_dir / "test_set_ids.txt"
    with open(ids_file, 'w') as f:
        for struct_id in sorted(unique_ids, key=str):
            f.write(f"{struct_id}\n")
    print(f"✅ Saved {len(unique_ids)} unique test IDs to: {ids_file}")
    
    # 3. Corresponding PDB files
    unique_pdbs = list(set(mapping.values()))
    pdbs_file = output_dir / "test_set_pdbs.txt"
    with open(pdbs_file, 'w') as f:
        for pdb_file in sorted(unique_pdbs):
            f.write(f"{pdb_file}\n")
    print(f"✅ Saved {len(unique_pdbs)} unique PDB files to: {pdbs_file}")
    
    # 4. Full mapping with metadata
    mapping_data = {
        "test_set_indices": test_indices,
        "test_set_mapping": {str(k): v for k, v in mapping.items()},
        "unmapped_ids": [(source, str(uid)) for source, uid in unmapped_ids],
        "statistics": {
            "total_test_items": len(test_structures),
            "test_indices": len(test_indices),
            "total_structure_ids": len(mapping) + len(unmapped_ids),
            "mapped_ids": len(mapping),
            "unmapped_ids": len(unmapped_ids),
            "unique_structures": len(unique_ids),
            "unique_pdb_files": len(unique_pdbs),
            "mapping_success_rate": len(mapping)/(len(mapping)+len(unmapped_ids)) if (len(mapping)+len(unmapped_ids)) > 0 else 0
        }
    }
    
    mapping_file = output_dir / "test_set_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"✅ Saved full mapping data to: {mapping_file}")
    
    # 5. Summary report
    report_file = output_dir / "test_set_report.txt"
    with open(report_file, 'w') as f:
        f.write("DAS TEST SET ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset split: DAS (train/val/test)\n")
        f.write(f"Test set indices: {len(test_indices)}\n")
        f.write(f"Test set items: {len(test_structures)}\n")
        f.write(f"Total structure IDs extracted: {len(mapping) + len(unmapped_ids)}\n")
        f.write(f"Successfully mapped: {len(mapping)} IDs\n")
        f.write(f"Unmapped IDs: {len(unmapped_ids)}\n")
        f.write(f"Unique structures: {len(unique_ids)}\n")
        f.write(f"Unique PDB files: {len(unique_pdbs)}\n")
        if (len(mapping)+len(unmapped_ids)) > 0:
            success_rate = len(mapping)/(len(mapping)+len(unmapped_ids))*100
            f.write(f"Mapping success rate: {success_rate:.1f}%\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("- test_set_indices.txt: Dataset indices for test set\n")
        f.write("- test_set_ids.txt: Structure IDs for test set\n")
        f.write("- test_set_pdbs.txt: Raw PDB filenames for test set\n")
        f.write("- test_set_mapping.json: Complete mapping data\n\n")
        
        f.write("USAGE FOR BASELINE MODELS:\n")
        f.write("Use test_set_pdbs.txt to evaluate other inverse folding models\n")
        f.write("on the same test structures used for RiboPO evaluation.\n\n")
        
        if unmapped_ids:
            f.write("UNMAPPED IDs (first 20):\n")
            for source, uid in unmapped_ids[:20]:
                f.write(f"  {source}: {uid}\n")
    
    print(f"✅ Saved summary report to: {report_file}")
    
    return mapping_data

def main():
    print("🧬 DAS Test Set ID Extraction Tool v2")
    print("=" * 50)
    
    # Step 1: Load dataset and split
    dataset, split_data = load_dataset_and_split()
    if not dataset or not split_data:
        return
    
    # Step 2: Examine structures
    dataset = examine_dataset_structure(dataset)
    split_data = examine_split_structure(split_data)
    
    # Step 3: Extract test structures using indices
    test_structures = extract_test_structures(dataset, split_data)
    if not test_structures:
        return
    
    # Step 4: Extract structure IDs from test structures
    structure_ids = extract_structure_ids(test_structures)
    if not structure_ids:
        print("❌ No structure IDs found")
        return
    
    # Step 5: Map to raw files
    mapping, unmapped_ids = map_to_raw_files(structure_ids)
    
    # Step 6: Save results
    test_indices = split_data[2]  # Test split index
    results = save_results(mapping, unmapped_ids, test_structures, test_indices)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Final Statistics:")
    stats = results["statistics"]
    for key, value in stats.items():
        if key.endswith("_rate"):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()