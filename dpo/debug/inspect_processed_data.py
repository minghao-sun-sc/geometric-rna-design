#!/usr/bin/env python3
"""
Quick inspection of processed.pt structure
"""

import torch
import os

def inspect_processed_data():
    processed_path = "data/processed.pt"
    
    if not os.path.exists(processed_path):
        print(f"❌ File not found: {processed_path}")
        return
    
    print(f"🔍 Inspecting: {processed_path}")
    
    data = torch.load(processed_path, map_location='cpu')
    
    print(f"📊 Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"🔑 Keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"   {key}: list with {len(value)} items")
                if len(value) > 0:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        print(f"      First item keys: {list(first_item.keys())}")
                        if 'id_list' in first_item:
                            print(f"      First ID: {first_item['id_list'][0] if first_item['id_list'] else 'None'}")
            else:
                print(f"   {key}: {type(value)} - {str(value)[:100]}...")
    elif isinstance(data, list):
        print(f"📝 List with {len(data)} items")
        if len(data) > 0:
            first_item = data[0]
            print(f"   First item type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"   First item keys: {list(first_item.keys())}")
    else:
        print(f"   Content: {str(data)[:200]}...")

if __name__ == "__main__":
    inspect_processed_data()