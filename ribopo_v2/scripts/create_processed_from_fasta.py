#!/usr/bin/env python3
"""
Create processed.pt and das_split.pt for ribopo_v2 from existing FASTA files
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
import json

def read_fasta_sequences(fasta_path):
    """Read sequences from FASTA file"""
    sequences = {}
    sequence_order = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract PDB ID from header (e.g., "7PKS_1_P Extracted from 7PKS_1_P.pdb")
        pdb_id = record.id
        sequence = str(record.seq)
        
        # Create data entry similar to original processed.pt format
        # Note: coords_list should have shape [sequence_length, 27, 3] (27 atoms per residue)
        data_entry = {
            'sequence': sequence,
            'id_list': [pdb_id],
            'coords_list': [torch.zeros((len(sequence), 27, 3))],  # Fixed: 27 atoms per residue
            'sec_struct_list': ["."] * len(sequence),  # Placeholder secondary structure
            'sasa_list': [np.zeros(len(sequence))],  # Placeholder SASA
            'rfam_list': ['Unknown'],
            'eq_class_list': ['filtered'],
            'type_list': ['RNA'],
            'rmsds_list': [0.0],
            'cluster_seqid0.8': f'ribopo_v2_cluster_{pdb_id}',
            'cluster_structsim0.45': f'ribopo_v2_structsim_{pdb_id}'
        }
        
        # Use sequence as key (like original format)
        if sequence in sequences:
            # Merge with existing entry (same sequence, different ID)
            existing = sequences[sequence]
            existing['id_list'].append(pdb_id)
            existing['coords_list'].append(torch.zeros((len(sequence), 27, 3)))  # Fixed: 27 atoms per residue
            existing['sec_struct_list'].append("." * len(sequence))
            existing['sasa_list'].append(np.zeros(len(sequence)))
        else:
            sequences[sequence] = data_entry
            sequence_order.append(sequence)
    
    return sequences, sequence_order

def create_ribopo_v2_processed_from_fasta():
    """Create processed.pt and das_split.pt for ribopo_v2 from FASTA files"""
    
    data_dir = Path("/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered")
    native_seq_dir = data_dir / "native_seq"
    output_dir = data_dir
    
    print("🔄 Creating processed.pt and das_split.pt for RiboPO v2 from FASTA files")
    print(f"📁 Source directory: {native_seq_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Read sequences from each split
    print("📖 Reading training sequences...")
    train_sequences, train_order = read_fasta_sequences(
        native_seq_dir / "train" / "train_native_sequences.fasta"
    )
    
    print("📖 Reading validation sequences...")
    val_sequences, val_order = read_fasta_sequences(
        native_seq_dir / "val" / "val_native_sequences.fasta"
    )
    
    print("📖 Reading test sequences...")
    test_sequences, test_order = read_fasta_sequences(
        native_seq_dir / "test" / "test_native_sequences.fasta"
    )
    
    # Combine all sequences in order: train, val, test
    all_processed_data = {}
    all_sequence_order = []
    
    # Add train sequences
    for seq in train_order:
        all_processed_data[seq] = train_sequences[seq]
        all_sequence_order.append(seq)
    
    # Add validation sequences
    for seq in val_order:
        all_processed_data[seq] = val_sequences[seq]
        all_sequence_order.append(seq)
    
    # Add test sequences
    for seq in test_order:
        all_processed_data[seq] = test_sequences[seq]
        all_sequence_order.append(seq)
    
    # Create split indices based on sequence counts
    train_count = len(train_order)
    val_count = len(val_order)
    test_count = len(test_order)
    total_count = len(all_sequence_order)
    
    train_indices = list(range(0, train_count))
    val_indices = list(range(train_count, train_count + val_count))
    test_indices = list(range(train_count + val_count, total_count))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {total_count}")
    print(f"   Train: {len(train_indices)} sequences (indices {train_indices[0]}-{train_indices[-1]})")
    print(f"   Val: {len(val_indices)} sequences (indices {val_indices[0]}-{val_indices[-1]})")
    print(f"   Test: {len(test_indices)} sequences (indices {test_indices[0]}-{test_indices[-1]})")
    
    # Validate split indices
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_count
    assert max(train_indices + val_indices + test_indices) == total_count - 1
    assert min(train_indices + val_indices + test_indices) == 0
    
    # Save processed.pt
    processed_path = output_dir / "processed.pt"
    print(f"\n💾 Saving processed data: {processed_path}")
    torch.save(all_processed_data, processed_path)
    print(f"   Size: {processed_path.stat().st_size / (1024**2):.1f} MB")
    
    # Save das_split.pt (as tuple like original)
    split_path = output_dir / "das_split.pt"
    split_data = (train_indices, val_indices, test_indices)
    print(f"💾 Saving split data: {split_path}")
    torch.save(split_data, split_path)
    
    # Save detailed summary information
    summary = {
        'total_sequences': total_count,
        'train_count': len(train_indices),
        'val_count': len(val_indices),
        'test_count': len(test_indices),
        'sample_sequences': {
            'train_samples': list(train_order[:3]),
            'val_samples': list(val_order[:2]),
            'test_samples': list(test_order[:2])
        },
        'split_indices': {
            'train_start': 0,
            'train_end': train_count,
            'val_start': train_count,
            'val_end': train_count + val_count,
            'test_start': train_count + val_count,
            'test_end': total_count
        },
        'sequence_length_stats': {
            'min_length': min(len(seq) for seq in all_sequence_order),
            'max_length': max(len(seq) for seq in all_sequence_order),
            'mean_length': sum(len(seq) for seq in all_sequence_order) / len(all_sequence_order)
        }
    }
    
    summary_path = output_dir / "ribopo_v2_data_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saving detailed summary: {summary_path}")
    
    # Verify the data structure matches original format
    print(f"\n🔍 Verifying data structure...")
    sample_key = list(all_processed_data.keys())[0]
    sample_data = all_processed_data[sample_key]
    print(f"   Sample sequence length: {len(sample_key)}")
    print(f"   Sample data keys: {list(sample_data.keys())}")
    print(f"   Sample ID: {sample_data['id_list'][0]}")
    
    print(f"\n✅ Successfully created ribopo_v2 processed data files!")
    print(f"   📄 processed.pt: {processed_path}")
    print(f"   📄 das_split.pt: {split_path}")
    print(f"   📄 ribopo_v2_data_summary.json: {summary_path}")
    print(f"\n🎯 Ready for RiboPO v2 training pipeline!")
    
    return all_processed_data, split_data

if __name__ == "__main__":
    create_ribopo_v2_processed_from_fasta()