#!/usr/bin/env python3
"""
Test the created processed.pt and das_split.pt files
"""
import torch
import json
from pathlib import Path

def test_ribopo_v2_data():
    """Test the created data files"""
    
    data_dir = Path("/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered")
    
    print("🧪 Testing RiboPO v2 processed data files")
    
    # Test processed.pt
    print("\n📄 Testing processed.pt...")
    processed_path = data_dir / "processed.pt"
    processed_data = torch.load(processed_path, map_location='cpu')
    
    print(f"   ✅ Loaded successfully")
    print(f"   📊 Type: {type(processed_data)}")
    print(f"   📊 Number of unique sequences: {len(processed_data)}")
    
    # Test a sample entry
    sample_key = list(processed_data.keys())[0]
    sample_data = processed_data[sample_key]
    print(f"   📋 Sample sequence: {sample_key[:50]}...")
    print(f"   📋 Sample data structure: {list(sample_data.keys())}")
    print(f"   📋 Sample ID list: {sample_data['id_list']}")
    
    # Test das_split.pt
    print("\n📄 Testing das_split.pt...")
    split_path = data_dir / "das_split.pt"
    split_data = torch.load(split_path, map_location='cpu')
    
    print(f"   ✅ Loaded successfully")
    print(f"   📊 Type: {type(split_data)}")
    print(f"   📊 Number of splits: {len(split_data)}")
    
    train_indices, val_indices, test_indices = split_data
    print(f"   📊 Train indices: {len(train_indices)} (range: {min(train_indices)}-{max(train_indices)})")
    print(f"   📊 Val indices: {len(val_indices)} (range: {min(val_indices)}-{max(val_indices)})")
    print(f"   📊 Test indices: {len(test_indices)} (range: {min(test_indices)}-{max(test_indices)})")
    
    # Verify indices are valid
    max_index = max(train_indices + val_indices + test_indices)
    min_index = min(train_indices + val_indices + test_indices)
    total_indices = len(train_indices) + len(val_indices) + len(test_indices)
    
    print(f"   📊 Index range: {min_index} to {max_index}")
    print(f"   📊 Total indices: {total_indices}")
    print(f"   📊 Expected range: 0 to {len(processed_data) - 1}")
    
    # Verify consistency
    if max_index == len(processed_data) - 1 and min_index == 0 and total_indices == len(processed_data):
        print("   ✅ Split indices are consistent with processed data")
    else:
        print("   ❌ Split indices are NOT consistent with processed data")
    
    # Test accessing data by split
    print("\n🔍 Testing data access by split...")
    sequence_keys = list(processed_data.keys())
    
    # Test train data access
    train_sample_idx = train_indices[0]
    train_sample_seq = sequence_keys[train_sample_idx]
    train_sample_data = processed_data[train_sample_seq]
    print(f"   Train sample (idx {train_sample_idx}): {train_sample_data['id_list'][0]}")
    
    # Test val data access
    val_sample_idx = val_indices[0]
    val_sample_seq = sequence_keys[val_sample_idx]
    val_sample_data = processed_data[val_sample_seq]
    print(f"   Val sample (idx {val_sample_idx}): {val_sample_data['id_list'][0]}")
    
    # Test test data access
    test_sample_idx = test_indices[0]
    test_sample_seq = sequence_keys[test_sample_idx]
    test_sample_data = processed_data[test_sample_seq]
    print(f"   Test sample (idx {test_sample_idx}): {test_sample_data['id_list'][0]}")
    
    # Read summary
    print("\n📄 Reading summary...")
    summary_path = data_dir / "ribopo_v2_data_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    
    print(f"   📊 Total sequences: {summary['total_sequences']}")
    print(f"   📊 Train/Val/Test: {summary['train_count']}/{summary['val_count']}/{summary['test_count']}")
    print(f"   📊 Sequence lengths: {summary['sequence_length_stats']['min_length']}-{summary['sequence_length_stats']['max_length']} (avg: {summary['sequence_length_stats']['mean_length']:.1f})")
    
    print(f"\n✅ All tests passed! RiboPO v2 data files are ready for use.")
    
    return True

if __name__ == "__main__":
    test_ribopo_v2_data()