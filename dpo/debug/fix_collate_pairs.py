#!/usr/bin/env python
"""
Fixed collate_pairs function that attaches all data to the PyG batch.
This file shows the complete fix for dpo/data.py
"""

import torch
from torch_geometric.data import Batch
from typing import List, Tuple, Any


def collate_pairs_fixed(batch: List[Tuple[Any, ...]]):
    """
    Fixed version of collate_pairs that returns a single PyG batch 
    with all necessary attributes attached.
    
    This ensures compatibility with Lightning's batch handling and
    the dpo_sft_step function's expectations.
    """
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    
    # Create the base PyG batch from graph data
    data_batch = Batch.from_data_list(datas)
    
    # Attach DPO-specific tensors as attributes
    data_batch.y_w = torch.cat(ys_w, dim=0)
    data_batch.y_l = torch.cat(ys_l, dim=0)
    data_batch.weight = torch.stack(ws)
    
    # Handle node mask
    if masks[0] is not None:
        data_batch.node_mask = torch.cat(masks, dim=0)
    else:
        # Create all-ones mask if not using windowing
        data_batch.node_mask = torch.ones(data_batch.y_w.shape[0], dtype=torch.float32)
    
    # Store graph IDs for reference
    data_batch.gids = list(gids)
    
    # Return single batch object with all attributes
    # This will work correctly with _unwrap_batch and dpo_sft_step
    return data_batch


# Here's what needs to be changed in dpo/data.py:
# Replace the existing collate_pairs function (lines 304-316) with:

REPLACEMENT_CODE = '''
def collate_pairs(batch: List[Tuple[Any, ...]]):
    """Collate function that attaches all data to the PyG batch for DPO training."""
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    
    # Create base PyG batch
    data_batch = Batch.from_data_list(datas)
    
    # Attach DPO-specific data as batch attributes
    data_batch.y_w = torch.cat(ys_w, dim=0)
    data_batch.y_l = torch.cat(ys_l, dim=0) 
    data_batch.weight = torch.stack(ws)
    
    if masks[0] is not None:
        data_batch.node_mask = torch.cat(masks, dim=0)
    else:
        data_batch.node_mask = torch.ones(data_batch.y_w.shape[0], dtype=torch.float32)
    
    data_batch.gids = list(gids)
    
    return data_batch
'''

print("Fix for dpo/data.py:")
print("=" * 60)
print(REPLACEMENT_CODE)
print("=" * 60)
print("\nThis fix ensures:")
print("1. All data stays together in a single PyG batch object")
print("2. Lightning's device transfers work correctly") 
print("3. _unwrap_batch won't lose any data")
print("4. dpo_sft_step can access y_w, y_l, node_mask via batch attributes")