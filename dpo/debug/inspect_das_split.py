#!/usr/bin/env python3
"""
Quick inspection of das_split.pt structure
"""

import torch
import os

def inspect_das_split():
    split_path = "data/das_split.pt"
    
    if not os.path.exists(split_path):
        print(f"❌ File not found: {split_path}")
        return
    
    print(f"🔍 Inspecting: {split_path}")
    
    data = torch.load(split_path, map_location='cpu')
    
    print(f"📊 Data type: {type(data)}")
    
    if isinstance(data, tuple):
        print(f"📦 Tuple with {len(data)} elements")
        for i, item in enumerate(data):
            print(f"   Element {i}: {type(item)}")
            if isinstance(item, dict):
                print(f"      Keys: {list(item.keys())}")
                for key, value in item.items():
                    if isinstance(value, list):
                        print(f"         {key}: list with {len(value)} items")
                        if len(value) > 0 and isinstance(value[0], dict):
                            print(f"            First item keys: {list(value[0].keys())}")
                            if 'id_list' in value[0]:
                                print(f"            First ID: {value[0]['id_list'][0] if value[0]['id_list'] else 'None'}")
                    else:
                        print(f"         {key}: {type(value)}")
            elif isinstance(item, list):
                print(f"      List with {len(item)} items")
                if len(item) > 0:
                    print(f"         First item type: {type(item[0])}")
                    if isinstance(item[0], dict) and 'id_list' in item[0]:
                        print(f"         First ID: {item[0]['id_list'][0] if item[0]['id_list'] else 'None'}")

if __name__ == "__main__":
    inspect_das_split()
