#!/usr/bin/env python
"""
Filter preference pairs to only include those with exact length matches to graph structures.
This creates a cleaner dataset for simplified training without windowing/masking.
"""

import json
import torch
import os
from typing import Dict, List, Set
from collections import defaultdict

def extract_pdb_name(pdb_path: str) -> str:
    """Extract PDB name from file path."""
    return os.path.basename(pdb_path).replace('.pdb', '')

def find_exact_matches(
    pairs_path: str,
    processed_path: str,
    split_path: str,
    split: str = "train",
    target_count: int = 20000
) -> List[dict]:
    """
    Find pairs that exactly match graph structure lengths.
    
    Args:
        pairs_path: Path to pairs JSONL file
        processed_path: Path to processed.pt
        split_path: Path to split file (das_split.pt)
        split: Which split to process (train/val/test)
        target_count: Target number of pairs to collect
        
    Returns:
        List of filtered pairs with exact length matches
    """
    
    # Load processed data and splits
    print(f"Loading processed data from {processed_path}...")
    data_dict = torch.load(processed_path)
    all_raws = list(data_dict.values())
    
    print(f"Loading splits from {split_path}...")
    train_idx, val_idx, test_idx = torch.load(split_path)
    
    # Select appropriate indices
    if split == "train":
        split_indices = set(train_idx if isinstance(train_idx, list) else train_idx.tolist())
    elif split == "val":
        split_indices = set(val_idx if isinstance(val_idx, list) else val_idx.tolist())
    else:
        split_indices = set(test_idx if isinstance(test_idx, list) else test_idx.tolist())
    
    # Build a map from PDB names to (local_idx, sequence_length)
    pdb_to_info = defaultdict(list)
    for idx in split_indices:
        raw = all_raws[idx]
        seq_len = len(raw.get('sequence', ''))
        
        # Extract PDB identifiers from id_list
        for id_item in raw.get('id_list', []):
            # Handle different ID formats
            if isinstance(id_item, str):
                # Extract PDB code from strings like "1A34_1_B-C" or full paths
                if '/' in id_item:
                    pdb_name = os.path.basename(id_item).replace('.pdb', '')
                else:
                    pdb_name = id_item.split('.')[0]  # Remove extensions
                pdb_to_info[pdb_name].append((idx, seq_len))
    
    print(f"Built index for {len(pdb_to_info)} PDB structures in {split} split")
    
    # Load and filter pairs
    print(f"Loading pairs from {pairs_path}...")
    filtered_pairs = []
    total_pairs = 0
    length_mismatches = 0
    not_in_split = 0
    
    with open(pairs_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            pair = json.loads(line)
            total_pairs += 1
            
            # Extract PDB name from pair
            pdb_name = extract_pdb_name(pair['pdb_file'])
            
            # Check if this PDB is in our split
            if pdb_name not in pdb_to_info:
                not_in_split += 1
                continue
            
            # Get sequence lengths
            pair_len = len(pair['winner_seq'])
            assert len(pair['loser_seq']) == pair_len, "Winner/loser length mismatch!"
            
            # Find matching graph structures
            found_match = False
            for idx, graph_len in pdb_to_info[pdb_name]:
                if graph_len == pair_len:
                    # Exact match found!
                    filtered_pair = {
                        "pdb_file": pair['pdb_file'],
                        "winner_seq": pair['winner_seq'],
                        "loser_seq": pair['loser_seq'],
                        "weight": pair.get('weight', 1.0),
                        "_graph_idx": idx,  # Store for verification
                        "_length": pair_len
                    }
                    
                    # Include metrics if available
                    if 'winner_metrics' in pair:
                        filtered_pair['winner_metrics'] = pair['winner_metrics']
                    if 'loser_metrics' in pair:
                        filtered_pair['loser_metrics'] = pair['loser_metrics']
                    
                    filtered_pairs.append(filtered_pair)
                    found_match = True
                    break
            
            if not found_match:
                length_mismatches += 1
            
            # Stop if we have enough pairs
            if len(filtered_pairs) >= target_count:
                break
            
            # Progress update
            if total_pairs % 5000 == 0:
                print(f"  Processed {total_pairs} pairs, found {len(filtered_pairs)} exact matches")
    
    print(f"\nFiltering complete:")
    print(f"  Total pairs processed: {total_pairs}")
    print(f"  Exact matches found: {len(filtered_pairs)}")
    print(f"  Length mismatches: {length_mismatches}")
    print(f"  Not in split: {not_in_split}")
    
    # Analyze length distribution
    length_dist = defaultdict(int)
    for pair in filtered_pairs:
        length_dist[pair['_length']] += 1
    
    print(f"\nLength distribution of filtered pairs:")
    for length in sorted(length_dist.keys())[:10]:
        print(f"  Length {length}: {length_dist[length]} pairs")
    if len(length_dist) > 10:
        print(f"  ... and {len(length_dist) - 10} more lengths")
    
    return filtered_pairs

def save_filtered_pairs(pairs: List[dict], output_path: str):
    """Save filtered pairs to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pair in pairs:
            # Remove internal fields before saving
            save_pair = {k: v for k, v in pair.items() if not k.startswith('_')}
            f.write(json.dumps(save_pair) + '\n')
    
    print(f"Saved {len(pairs)} pairs to {output_path}")

def main():
    """Filter all splits to create exact-match datasets."""
    
    base_dir = "/mnt/rna01/smh/projects/offline-dpo"
    
    # Process each split
    for split in ["train", "val"]:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        input_path = f"{base_dir}/data/pairs_margin125/split_by_das/{split}.jsonl"
        output_path = f"{base_dir}/data/pairs_exact_match/{split}.jsonl"
        
        # Use different target counts for different splits
        target_count = 20000 if split == "train" else 2000
        
        filtered_pairs = find_exact_matches(
            pairs_path=input_path,
            processed_path=f"{base_dir}/data/processed.pt",
            split_path=f"{base_dir}/data/das_split.pt",
            split=split,
            target_count=target_count
        )
        
        save_filtered_pairs(filtered_pairs, output_path)
    
    print(f"\n{'='*60}")
    print("Filtering complete! Exact-match datasets created.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()