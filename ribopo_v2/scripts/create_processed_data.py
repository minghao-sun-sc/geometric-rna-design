#!/usr/bin/env python3
"""
Create processed.pt and das_split.pt for ribopo_v2 from filtered PDB files
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_utils import pdb_to_tensor, get_c4p_coords

def extract_sequence_from_pdb(pdb_path):
    """Extract RNA sequence from PDB file"""
    try:
        seq, coords, sec_struct, sasa = pdb_to_tensor(
            pdb_path,
            return_sec_struct=True,
            return_sasa=True,
            keep_insertions=False,
        )
        return seq, coords, sec_struct, sasa
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None, None, None, None

def process_split_directory(split_dir, split_name):
    """Process all PDB files in a split directory"""
    print(f"\n📂 Processing {split_name} split: {split_dir}")
    
    pdb_files = list(Path(split_dir).glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")
    
    processed_data = {}
    indices = []
    
    for i, pdb_path in enumerate(tqdm(pdb_files, desc=f"Processing {split_name}")):
        # Extract ID from filename (e.g., "1CSL_1_B-A.pdb" -> "1CSL_1_B-A")
        structure_id = pdb_path.stem
        
        # Process PDB
        seq, coords, sec_struct, sasa = extract_sequence_from_pdb(str(pdb_path))
        
        if seq is None:
            print(f"❌ Failed to process {pdb_path}")
            continue
        
        # Convert sequence tensor to string
        from src.constants import NUM_TO_LETTER
        seq_str = "".join([NUM_TO_LETTER[int(n)] for n in seq])
        
        # Create data entry similar to original processed.pt format
        data_entry = {
            'sequence': seq_str,
            'id_list': [structure_id],
            'coords_list': [coords],
            'sec_struct_list': [sec_struct] if sec_struct else ["."] * len(seq_str),
            'sasa_list': [sasa] if sasa is not None else [np.zeros(len(seq_str))],
            'rfam_list': ['Unknown'],  # We don't have RFAM annotations for filtered data
            'eq_class_list': ['filtered'],
            'type_list': ['RNA'],
            'rmsds_list': [0.0],  # Single structure, so RMSD is 0
            'cluster_seqid0.8': f'{split_name}_cluster_{i}',
            'cluster_structsim0.45': f'{split_name}_structsim_{i}'
        }
        
        # Use sequence as key (like original format)
        if seq_str in processed_data:
            # Merge with existing entry (same sequence, different ID)
            existing = processed_data[seq_str]
            existing['id_list'].append(structure_id)
            existing['coords_list'].append(coords)
            existing['sec_struct_list'].append(sec_struct if sec_struct else "." * len(seq_str))
            if sasa is not None:
                existing['sasa_list'].append(sasa)
        else:
            processed_data[seq_str] = data_entry
            indices.append(len(processed_data) - 1)  # Will be corrected later
    
    return processed_data, len(processed_data)

def create_ribopo_v2_processed_data():
    """Create processed.pt and das_split.pt for ribopo_v2"""
    
    data_dir = Path("/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered")
    output_dir = data_dir
    
    print("🔄 Creating processed.pt and das_split.pt for RiboPO v2")
    print(f"📁 Source directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Process each split
    all_processed_data = {}
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Process training data
    train_data, train_count = process_split_directory(data_dir / "train", "train")
    all_processed_data.update(train_data)
    
    # Process validation data  
    val_data, val_count = process_split_directory(data_dir / "val", "val")
    all_processed_data.update(val_data)
    
    # Process test data
    test_data, test_count = process_split_directory(data_dir / "test", "test")
    all_processed_data.update(test_data)
    
    # Create split indices
    # The keys of all_processed_data are in the order: train, val, test
    total_sequences = len(all_processed_data)
    sequence_keys = list(all_processed_data.keys())
    
    # Calculate split boundaries based on actual data counts
    train_end = train_count
    val_end = train_end + val_count
    test_end = val_end + test_count
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, test_end))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {total_sequences}")
    print(f"   Train: {len(train_indices)} sequences")
    print(f"   Val: {len(val_indices)} sequences")
    print(f"   Test: {len(test_indices)} sequences")
    
    # Save processed.pt
    processed_path = output_dir / "processed.pt"
    print(f"\n💾 Saving processed data: {processed_path}")
    torch.save(all_processed_data, processed_path)
    
    # Save das_split.pt (as tuple like original)
    split_path = output_dir / "das_split.pt" 
    split_data = (train_indices, val_indices, test_indices)
    print(f"💾 Saving split data: {split_path}")
    torch.save(split_data, split_path)
    
    # Save summary information
    summary = {
        'total_sequences': total_sequences,
        'train_count': len(train_indices),
        'val_count': len(val_indices), 
        'test_count': len(test_indices),
        'sequence_keys_sample': sequence_keys[:5],
        'split_boundaries': {
            'train_end': train_end,
            'val_end': val_end,
            'test_end': test_end
        }
    }
    
    summary_path = output_dir / "data_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saving summary: {summary_path}")
    
    print(f"\n✅ Successfully created ribopo_v2 processed data files!")
    print(f"   📄 processed.pt: {processed_path}")
    print(f"   📄 das_split.pt: {split_path}")
    print(f"   📄 data_summary.json: {summary_path}")
    
    return all_processed_data, split_data

if __name__ == "__main__":
    create_ribopo_v2_processed_data()