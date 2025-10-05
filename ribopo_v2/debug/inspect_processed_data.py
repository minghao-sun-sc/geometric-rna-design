#!/usr/bin/env python3
"""
Inspect processed.pt structure to understand data format for ribopo_v2 adaptation
"""
import torch
import os
import sys
from pathlib import Path

def inspect_processed_data():
    """Inspect the structure of original processed.pt"""
    
    original_processed_path = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    
    if not os.path.exists(original_processed_path):
        print(f"❌ File not found: {original_processed_path}")
        return
    
    print(f"🔍 Inspecting: {original_processed_path}")
    print(f"📊 File size: {os.path.getsize(original_processed_path) / (1024**3):.2f} GB")
    
    # Load data carefully
    print("Loading data...")
    data = torch.load(original_processed_path, map_location='cpu')
    
    print(f"📊 Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📦 Dictionary with {len(data)} keys")
        print(f"🔑 Sample keys: {list(data.keys())[:10]}")
        
        # Examine first few entries
        sample_key = list(data.keys())[0]
        sample_data = data[sample_key]
        print(f"\n📋 Sample entry structure (key: {sample_key}):")
        print(f"   Type: {type(sample_data)}")
        
        if isinstance(sample_data, dict):
            print(f"   Keys: {list(sample_data.keys())}")
            for key, value in list(sample_data.items())[:5]:
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: Tensor {value.shape}")
                elif isinstance(value, list):
                    print(f"      {key}: List with {len(value)} items")
                    if len(value) > 0:
                        print(f"         First item type: {type(value[0])}")
                        if isinstance(value[0], str):
                            print(f"         First item: {value[0]}")
                else:
                    print(f"      {key}: {type(value)} - {str(value)[:100]}")
    
    elif isinstance(data, list):
        print(f"📦 List with {len(data)} items")
        if len(data) > 0:
            print(f"   First item type: {type(data[0])}")
            sample_data = data[0]
            if isinstance(sample_data, dict):
                print(f"   Sample keys: {list(sample_data.keys())}")
    
    print(f"\n✅ Analysis complete")

def check_ribopo_v2_filtered_data():
    """Check what data we have in ribopo_v2 filtered directory"""
    
    filtered_dir = Path("/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered")
    
    print(f"\n🔍 Checking ribopo_v2 filtered data directory:")
    print(f"📁 Directory: {filtered_dir}")
    
    if not filtered_dir.exists():
        print("❌ Directory doesn't exist")
        return
    
    print("📂 Contents:")
    for item in sorted(filtered_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024**2)
            print(f"   📄 {item.name}: {size_mb:.2f} MB")
        elif item.is_dir():
            count = len(list(item.iterdir()))
            print(f"   📁 {item.name}/: {count} items")

if __name__ == "__main__":
    inspect_processed_data()
    check_ribopo_v2_filtered_data()