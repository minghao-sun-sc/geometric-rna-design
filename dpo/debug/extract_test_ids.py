#!/usr/bin/env python3
"""Extract test set structure IDs from DAS split for baseline comparisons"""

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

def load_das_split():
    """Load and examine the DAS split data"""
    split_path = "/mnt/rna01/smh/projects/ribopo/data/das_split.pt"
    print(f"Loading DAS split from: {split_path}")
    
    try:
        split_data = torch.load(split_path, map_location='cpu')
        print(f"✅ Successfully loaded split data")
        print(f"Split data type: {type(split_data)}")
        
        if isinstance(split_data, dict):
            print(f"Split keys: {list(split_data.keys())}")
            for split_name, split_items in split_data.items():
                print(f"  {split_name}: {len(split_items)} items (type: {type(split_items)})")
        elif isinstance(split_data, (list, tuple)):
            print(f"Split is a {type(split_data).__name__} with {len(split_data)} elements")
            for i, element in enumerate(split_data):
                print(f"  Element {i}: {type(element)} with length {len(element) if hasattr(element, '__len__') else 'N/A'}")
        else:
            print(f"Unknown split data format: {type(split_data)}")
        
        return split_data
    except Exception as e:
        print(f"❌ Error loading split data: {e}")
        return None

def examine_data_structure(split_data, split_index=2, max_items=3):
    """Examine the structure of split data items"""
    if isinstance(split_data, (list, tuple)):
        if split_index >= len(split_data):
            print(f"❌ Split index {split_index} out of range (max: {len(split_data)-1})")
            return None
        items = split_data[split_index]
        print(f"📊 Examining split element {split_index} (assumed to be test set)")
    elif isinstance(split_data, dict):
        split_name = "test"
        if split_name not in split_data:
            print(f"❌ Split '{split_name}' not found in data")
            return None
        items = split_data[split_name]
    else:
        print(f"❌ Unknown split data format: {type(split_data)}")
        return None
    print(f"\nStructure ({len(items)} total items):")
    
    for i, item in enumerate(items[:max_items]):
        print(f"\n--- Item {i} ---")
        print(f"Type: {type(item)}")
        
        if hasattr(item, '__dict__'):
            # Object with attributes
            attrs = [attr for attr in dir(item) if not attr.startswith('_')]
            print(f"Attributes: {attrs}")
            
            for attr in ['gid', 'id_list', 'pdb_id', 'structure_id'][:5]:  # Check common ID fields
                if hasattr(item, attr):
                    val = getattr(item, attr)
                    print(f"  {attr}: {val} (type: {type(val)})")
        
        elif isinstance(item, dict):
            # Dictionary
            print(f"Keys: {list(item.keys())}")
            for key in ['gid', 'id_list', 'pdb_id', 'structure_id'][:5]:
                if key in item:
                    print(f"  {key}: {item[key]}")
        
        elif isinstance(item, (list, tuple)):
            # List/tuple
            print(f"Length: {len(item)}")
            if len(item) > 0:
                print(f"First element: {item[0]} (type: {type(item[0])})")
        
        else:
            # Other types
            print(f"Value: {item}")
        
        if i >= max_items - 1:
            break
    
    return items

def extract_test_ids(test_items):
    """Extract all structure IDs from test items"""
    print(f"\n🔍 Extracting structure IDs from {len(test_items)} test items...")
    
    structure_ids = []
    id_sources = defaultdict(int)
    
    for i, item in enumerate(test_items):
        item_ids = []
        
        # Try different ways to get IDs
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
        
        # Check if it's a dictionary
        if isinstance(item, dict):
            for key in ['gid', 'id_list', 'pdb_id', 'structure_id']:
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
            print(f"⚠️ No IDs found for item {i}: {type(item)}")
    
    print(f"\n📈 ID source statistics:")
    for source, count in id_sources.items():
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
    print(f"Sample raw basenames: {list(raw_basenames.keys())[:5]}")
    
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
            # If multiple matches, take the first (or implement more sophisticated logic)
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
    print(f"  Mapping success rate: {len(mapping)/(len(mapping)+len(unmapped_ids))*100:.1f}%")
    
    print(f"\n📋 ID Format Distribution:")
    for fmt, count in id_formats.most_common():
        print(f"  {fmt}: {count}")
    
    if unmapped_ids:
        print(f"\n⚠️ Sample unmapped IDs:")
        for source, uid in unmapped_ids[:10]:
            print(f"  {source}: {uid}")
    
    return mapping, unmapped_ids

def save_results(mapping, unmapped_ids, test_items):
    """Save the mapping results to files"""
    output_dir = Path("/mnt/rna01/smh/projects/ribopo/dpo/debug")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_dir}")
    
    # 1. Test set IDs (unique structure IDs)
    unique_ids = list(set(mapping.keys()))
    ids_file = output_dir / "test_set_ids.txt"
    with open(ids_file, 'w') as f:
        for struct_id in sorted(unique_ids):
            f.write(f"{struct_id}\n")
    print(f"✅ Saved {len(unique_ids)} unique test IDs to: {ids_file}")
    
    # 2. Corresponding PDB files
    unique_pdbs = list(set(mapping.values()))
    pdbs_file = output_dir / "test_set_pdbs.txt"
    with open(pdbs_file, 'w') as f:
        for pdb_file in sorted(unique_pdbs):
            f.write(f"{pdb_file}\n")
    print(f"✅ Saved {len(unique_pdbs)} unique PDB files to: {pdbs_file}")
    
    # 3. Full mapping with metadata
    mapping_data = {
        "test_set_mapping": {str(k): v for k, v in mapping.items()},
        "unmapped_ids": [(source, str(uid)) for source, uid in unmapped_ids],
        "statistics": {
            "total_test_items": len(test_items),
            "total_structure_ids": len(mapping) + len(unmapped_ids),
            "mapped_ids": len(mapping),
            "unmapped_ids": len(unmapped_ids),
            "unique_structures": len(unique_ids),
            "unique_pdb_files": len(unique_pdbs),
            "mapping_success_rate": len(mapping)/(len(mapping)+len(unmapped_ids))
        }
    }
    
    mapping_file = output_dir / "test_set_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"✅ Saved full mapping data to: {mapping_file}")
    
    # 4. Summary report
    report_file = output_dir / "test_set_report.txt"
    with open(report_file, 'w') as f:
        f.write("DAS TEST SET ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set size: {len(test_items)} items\n")
        f.write(f"Total structure IDs extracted: {len(mapping) + len(unmapped_ids)}\n")
        f.write(f"Successfully mapped: {len(mapping)} IDs\n")
        f.write(f"Unmapped IDs: {len(unmapped_ids)}\n")
        f.write(f"Unique structures: {len(unique_ids)}\n")
        f.write(f"Unique PDB files: {len(unique_pdbs)}\n")
        f.write(f"Mapping success rate: {len(mapping)/(len(mapping)+len(unmapped_ids))*100:.1f}%\n\n")
        
        f.write("USAGE:\n")
        f.write("- Use test_set_ids.txt for structure IDs\n")
        f.write("- Use test_set_pdbs.txt for raw PDB filenames\n")
        f.write("- Use test_set_mapping.json for programmatic access\n\n")
        
        if unmapped_ids:
            f.write("UNMAPPED IDs (first 20):\n")
            for source, uid in unmapped_ids[:20]:
                f.write(f"  {source}: {uid}\n")
    
    print(f"✅ Saved summary report to: {report_file}")
    
    return mapping_data

def main():
    print("🧬 DAS Test Set ID Extraction Tool")
    print("=" * 50)
    
    # Step 1: Load DAS split data
    split_data = load_das_split()
    if not split_data:
        return
    
    # Step 2: Examine data structure
    test_items = examine_data_structure(split_data)
    if not test_items:
        return
    
    # Step 3: Extract structure IDs
    structure_ids = extract_test_ids(test_items)
    if not structure_ids:
        print("❌ No structure IDs found")
        return
    
    # Step 4: Map to raw files
    mapping, unmapped_ids = map_to_raw_files(structure_ids)
    
    # Step 5: Save results
    results = save_results(mapping, unmapped_ids, test_items)
    
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