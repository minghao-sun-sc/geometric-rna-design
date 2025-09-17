#!/usr/bin/env python3
"""
Extract test set structure IDs from DAS split, reusing proven scripts workflow.
Based on dpo/scripts/filter_pairs_by_split.py approach.
"""

import sys
import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

# Import proven utility functions
from dpo.common_id import canonicalize_id, canonical_from_id_list

def load_processed_data(processed_pt: str) -> List[dict]:
    """Load processed data, following filter_pairs_by_split.py approach"""
    print(f"📂 Loading processed data from: {processed_pt}")
    data = torch.load(processed_pt, map_location='cpu')
    
    if isinstance(data, dict):
        print(f"  Data is dict with {len(data)} entries")
        # Convert dict values to list (sequences as keys, structures as values)
        raw_list = list(data.values())
    else:
        print(f"  Data is list with {len(data)} entries")
        raw_list = list(data)
    
    print(f"  Converted to list of {len(raw_list)} raw data items")
    return raw_list

def load_das_split(split_pt: str):
    """Load DAS split indices, following filter_pairs_by_split.py approach"""
    print(f"📂 Loading DAS split from: {split_pt}")
    train_idx, val_idx, test_idx = torch.load(split_pt, map_location='cpu')
    
    train_list = list(map(int, train_idx))
    val_list = list(map(int, val_idx))
    test_list = list(map(int, test_idx))
    
    print(f"  Train: {len(train_list)} items")
    print(f"  Val:   {len(val_list)} items") 
    print(f"  Test:  {len(test_list)} items")
    
    return train_list, val_list, test_list

def extract_test_structure_ids(all_raws: List[dict], test_indices: List[int]) -> Dict:
    """Extract structure IDs from test set using proven ID extraction logic"""
    print(f"\n🔍 Extracting structure IDs from {len(test_indices)} test items...")
    
    # Get test raw data items
    test_raws = [all_raws[i] for i in test_indices]
    
    # Collect all structure IDs using the proven canonicalization approach
    structure_ids = []
    id_sources = defaultdict(int)
    
    for idx, raw_item in enumerate(test_raws):
        global_idx = test_indices[idx]
        item_ids = []
        
        # Extract IDs from id_list (primary source)
        if 'id_list' in raw_item and raw_item['id_list']:
            for raw_id in raw_item['id_list']:
                canonical_id = canonicalize_id(str(raw_id))
                if canonical_id:
                    item_ids.append(('id_list', canonical_id))
                    id_sources['id_list'] += 1
        
        # Extract canonical ID using the proven approach
        if 'id_list' in raw_item:
            canonical_main = canonical_from_id_list(raw_item['id_list'])
            if canonical_main:
                item_ids.append(('canonical', canonical_main))
                id_sources['canonical'] += 1
        
        # Store with metadata
        for source, struct_id in item_ids:
            structure_ids.append({
                'test_index': idx,
                'global_index': global_idx,
                'source': source,
                'structure_id': struct_id,
                'sequence': raw_item.get('sequence', '')[:50] + '...' if raw_item.get('sequence', '') else '',
            })
        
        if not item_ids:
            print(f"⚠️  No IDs found for test item {idx} (global: {global_idx})")
    
    print(f"\n📊 ID extraction statistics:")
    for source, count in id_sources.items():
        print(f"  {source}: {count} IDs")
    
    return {
        'structure_ids': structure_ids,
        'statistics': dict(id_sources),
        'test_count': len(test_indices),
        'total_ids': len(structure_ids)
    }

def map_to_raw_files(structure_ids: List[dict], data_path: str = "/mnt/rna01/smh/projects/ribopo/data") -> Dict:
    """Map structure IDs to raw PDB files using proven canonicalization"""
    raw_dir = Path(data_path) / "raw"
    print(f"\n🗂️  Mapping IDs to raw files in: {raw_dir}")
    
    if not raw_dir.exists():
        print(f"❌ Raw directory not found: {raw_dir}")
        return {'mapped': {}, 'unmapped': structure_ids, 'raw_files': []}
    
    # Get all raw files (PDB and CIF)
    raw_files = list(raw_dir.glob("*.pdb")) + list(raw_dir.glob("*.cif"))
    print(f"  Found {len(raw_files)} raw files")
    
    # Create canonical name mapping
    raw_canonical_map = {}
    for file_path in raw_files:
        canonical_name = canonicalize_id(file_path.stem)
        raw_canonical_map[canonical_name] = file_path.name
    
    print(f"  Canonical mapping created for {len(raw_canonical_map)} files")
    print(f"  Sample canonical names: {list(raw_canonical_map.keys())[:5]}")
    
    # Map structure IDs to files
    mapped = {}
    unmapped = []
    
    for item in structure_ids:
        struct_id = item['structure_id']
        canonical_struct_id = canonicalize_id(struct_id)
        
        if canonical_struct_id in raw_canonical_map:
            raw_file = raw_canonical_map[canonical_struct_id]
            mapped[struct_id] = {
                'raw_file': raw_file,
                'test_index': item['test_index'],
                'global_index': item['global_index'],
                'source': item['source']
            }
        else:
            unmapped.append(item)
    
    print(f"\n📊 Mapping results:")
    print(f"  Successfully mapped: {len(mapped)} IDs")
    print(f"  Unmapped IDs: {len(unmapped)}")
    if len(mapped) + len(unmapped) > 0:
        success_rate = len(mapped) / (len(mapped) + len(unmapped)) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    return {
        'mapped': mapped,
        'unmapped': unmapped,
        'raw_files': list(raw_canonical_map.values())
    }

def save_results(extraction_data: Dict, mapping_data: Dict, output_dir: str):
    """Save all results to organized output files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_path}")
    
    # 1. Test set structure IDs (unique)
    unique_ids = list(set(item['structure_id'] for item in extraction_data['structure_ids']))
    ids_file = output_path / "test_set_structure_ids.txt"
    with open(ids_file, 'w') as f:
        for struct_id in sorted(unique_ids):
            f.write(f"{struct_id}\n")
    print(f"✅ Saved {len(unique_ids)} unique structure IDs to: {ids_file}")
    
    # 2. Successfully mapped PDB files
    mapped_files = list(set(item['raw_file'] for item in mapping_data['mapped'].values()))
    files_file = output_path / "test_set_raw_files.txt"
    with open(files_file, 'w') as f:
        for filename in sorted(mapped_files):
            f.write(f"{filename}\n")
    print(f"✅ Saved {len(mapped_files)} mapped raw files to: {files_file}")
    
    # 3. Detailed mapping with metadata
    mapping_detail = {
        'extraction_statistics': extraction_data['statistics'],
        'mapping_statistics': {
            'total_extracted_ids': len(extraction_data['structure_ids']),
            'unique_structure_ids': len(unique_ids),
            'successfully_mapped': len(mapping_data['mapped']),
            'unmapped_ids': len(mapping_data['unmapped']),
            'unique_raw_files': len(mapped_files),
            'mapping_success_rate': len(mapping_data['mapped']) / len(extraction_data['structure_ids']) if extraction_data['structure_ids'] else 0
        },
        'test_indices': [item['global_index'] for item in extraction_data['structure_ids']],
        'mapped_ids': {k: v for k, v in mapping_data['mapped'].items()},
        'unmapped_details': mapping_data['unmapped'][:20]  # First 20 for debugging
    }
    
    mapping_file = output_path / "test_set_detailed_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping_detail, f, indent=2)
    print(f"✅ Saved detailed mapping to: {mapping_file}")
    
    # 4. Summary report
    report_file = output_path / "test_set_extraction_report.txt"
    with open(report_file, 'w') as f:
        f.write("DAS TEST SET EXTRACTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set size: {extraction_data['test_count']} items\n")
        f.write(f"Total structure IDs extracted: {len(extraction_data['structure_ids'])}\n")
        f.write(f"Unique structure IDs: {len(unique_ids)}\n")
        f.write(f"Successfully mapped to raw files: {len(mapping_data['mapped'])}\n")
        f.write(f"Unmapped IDs: {len(mapping_data['unmapped'])}\n")
        f.write(f"Unique raw files found: {len(mapped_files)}\n")
        
        if extraction_data['structure_ids']:
            success_rate = len(mapping_data['mapped']) / len(extraction_data['structure_ids']) * 100
            f.write(f"Overall mapping success rate: {success_rate:.1f}%\n\n")
        
        f.write("USAGE:\n")
        f.write("- Use test_set_structure_ids.txt for canonical structure IDs\n")
        f.write("- Use test_set_raw_files.txt for corresponding raw file names\n")
        f.write("- Use test_set_detailed_mapping.json for programmatic access\n\n")
        
        f.write("ID SOURCE STATISTICS:\n")
        for source, count in extraction_data['statistics'].items():
            f.write(f"  {source}: {count} IDs\n")
        
        if mapping_data['unmapped']:
            f.write(f"\nUNMAPPED IDs (first 10):\n")
            for item in mapping_data['unmapped'][:10]:
                f.write(f"  {item['structure_id']} (source: {item['source']})\n")
    
    print(f"✅ Saved summary report to: {report_file}")
    
    return mapping_detail

def main():
    print("🧬 DAS Test Set ID Extraction Tool (v3 - Using Proven Scripts)")
    print("=" * 60)
    
    # Configuration
    processed_pt = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    split_pt = "/mnt/rna01/smh/projects/ribopo/data/das_split.pt"
    data_path = "/mnt/rna01/smh/projects/ribopo/data"
    output_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug"
    
    try:
        # Step 1: Load all processed data
        all_raws = load_processed_data(processed_pt)
        
        # Step 2: Load DAS split indices
        train_idx, val_idx, test_idx = load_das_split(split_pt)
        
        # Step 3: Extract test set structure IDs
        extraction_data = extract_test_structure_ids(all_raws, test_idx)
        
        # Step 4: Map to raw files
        mapping_data = map_to_raw_files(extraction_data['structure_ids'], data_path)
        
        # Step 5: Save results
        results = save_results(extraction_data, mapping_data, output_dir)
        
        # Final summary
        print(f"\n🎉 Extraction complete!")
        print(f"📊 Final Statistics:")
        stats = results['mapping_statistics']
        for key, value in stats.items():
            if key.endswith('_rate'):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())