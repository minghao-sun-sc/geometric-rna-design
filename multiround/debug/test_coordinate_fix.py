#!/usr/bin/env python3
"""
Test the coordinate fix to ensure backbone extraction works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import torch
from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def test_coordinate_fix():
    """Test that coordinate extraction works correctly."""
    print("🔧 Testing Coordinate Fix")
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
        print(f"  eval_dataset length: {len(evaluator.eval_dataset)}")
        
        # Test first item coordinate processing
        item = evaluator.eval_dataset[0]
        print(f"\n📊 Testing first item:")
        print(f"  ID: {item.get('id_list', ['unknown'])[0]}")
        print(f"  Sequence length: {len(item['sequence'])}")
        print(f"  Coords list length: {len(item['coords_list'])}")
        
        coords = item['coords_list'][0]
        print(f"  Fixed coords shape: {coords.shape}")
        
        if coords.shape[1] == 3:
            print(f"  ✅ Coordinates correctly extracted to backbone atoms (P, C4', N1)")
        else:
            print(f"  ❌ Unexpected coordinate shape: {coords.shape}")
            return False
        
        # Test featurizer on the fixed data
        print(f"\n🧪 Testing featurizer...")
        try:
            graph = evaluator.eval_dataset.featurizer(item)
            print(f"  ✅ Featurizer works correctly!")
            print(f"     Graph seq shape: {graph.seq.shape}")
            print(f"     Graph coords shape: {graph.coords.shape}")
            print(f"     Graph mask_coords sum: {graph.mask_coords.sum()}")
            return True
        except Exception as e:
            print(f"  ❌ Featurizer failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Coordinate Fix Test")
    print("Testing backbone atom extraction for featurizer compatibility")
    
    success = test_coordinate_fix()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"   Coordinate extraction and featurizer work correctly.")
    else:
        print(f"\n❌ FAILED")
        print(f"   Coordinate extraction or featurizer not working.")

if __name__ == "__main__":
    main()