#!/usr/bin/env python3
"""
Debug the masking issue in Vienna RNA evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import torch
import numpy as np
from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def debug_mask_issue():
    """Debug the mask_coords vs sequence length mismatch."""
    print("🔧 Debugging Mask Issue")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print(f"✅ Config loaded: {config_path}")
        
        # Create evaluator
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        
        print(f"✅ Evaluator created successfully")
        
        # Test the first few items to understand the dimensions
        for i in range(min(3, len(evaluator.eval_dataset))):
            print(f"\n📊 Item {i}:")
            item = evaluator.eval_dataset[i]
            print(f"  ID: {item.get('id_list', ['unknown'])[0]}")
            print(f"  Sequence length: {len(item['sequence'])}")
            print(f"  Coords list length: {len(item['coords_list'])}")
            
            coords = item['coords_list'][0]
            print(f"  Coords shape: {coords.shape}")
            
            # Test featurizer to see what mask_coords it produces
            try:
                graph = evaluator.eval_dataset.featurizer(item)
                print(f"  Graph seq length: {len(graph.seq)}")
                print(f"  Graph mask_coords length: {len(graph.mask_coords)}")
                print(f"  Graph mask_coords sum: {graph.mask_coords.sum()}")
                print(f"  Graph mask_coords: {graph.mask_coords}")
                
                # This is the critical check - does mask_coords length match sequence length?
                if len(graph.mask_coords) == len(item['sequence']):
                    print(f"  ✅ Mask length matches sequence length")
                else:
                    print(f"  ❌ MISMATCH: mask_coords length ({len(graph.mask_coords)}) != sequence length ({len(item['sequence'])})")
                
                # Check if mask_coords indices are valid for the sequence
                mask_indices = np.where(graph.mask_coords.cpu().numpy())[0]
                max_seq_idx = len(item['sequence']) - 1
                invalid_indices = mask_indices[mask_indices > max_seq_idx]
                if len(invalid_indices) > 0:
                    print(f"  ❌ Invalid mask indices: {invalid_indices} (max valid: {max_seq_idx})")
                else:
                    print(f"  ✅ All mask indices are valid")
                    
            except Exception as e:
                print(f"  ❌ Featurizer failed: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mask_issue()