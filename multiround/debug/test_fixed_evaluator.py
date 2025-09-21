#!/usr/bin/env python3
"""
Test script for the fixed MultiRoundEvaluator.
This script tests that the evaluation dataset loading now works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import yaml
from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def test_fixed_evaluator():
    """Test the fixed MultiRoundEvaluator class."""
    print("🔧 Testing Fixed MultiRoundEvaluator")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print(f"✅ Config loaded: {config_path}")
        
        # Import and create evaluator
        from multiround.evaluator import MultiRoundEvaluator
        
        print("Creating MultiRoundEvaluator...")
        evaluator = MultiRoundEvaluator(cfg)
        
        print(f"✅ Evaluator created successfully")
        print(f"  eval_dataset is None: {evaluator.eval_dataset is None}")
        
        if evaluator.eval_dataset is not None:
            print(f"  eval_dataset length: {len(evaluator.eval_dataset)}")
            print(f"  eval_dataset type: {type(evaluator.eval_dataset)}")
            
            # Test accessing a few items
            print(f"  Testing access to first 3 items...")
            for i in range(min(3, len(evaluator.eval_dataset))):
                item = evaluator.eval_dataset[i]
                print(f"    Item {i}: {len(item['sequence'])} nucleotides, {item['id_list'][0] if item.get('id_list') else 'no_id'}")
            
            # Test featurizer
            print(f"  Testing featurizer...")
            item = evaluator.eval_dataset[0]
            graph = evaluator.eval_dataset.featurizer(item)
            print(f"    Featurized graph: seq_len={len(graph.seq)}, mask_coords_sum={graph.mask_coords.sum() if hasattr(graph, 'mask_coords') else 'N/A'}")
            
            print(f"🎉 SUCCESS: Evaluator and dataset loading work correctly!")
            return evaluator
        else:
            print(f"❌ FAILED: eval_dataset is still None")
            return None
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_evaluation_call():
    """Test that evaluation can be called without errors."""
    print("\n" + "=" * 60)
    print("🔍 Testing Evaluation Call")
    print("=" * 60)
    
    evaluator = test_fixed_evaluator()
    if evaluator is None:
        print("❌ Cannot test evaluation - evaluator creation failed")
        return
    
    try:
        # Create a mock model for testing
        from dpo.ref_manager import build_model_from_cfg
        import torch
        
        # Load model from config
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Build model
        model = build_model_from_cfg(cfg.model).to(device)
        
        # Load checkpoint (use the base checkpoint for testing)
        if hasattr(cfg.paths, 'base_checkpoint') and os.path.exists(cfg.paths.base_checkpoint):
            print(f"Loading base checkpoint: {cfg.paths.base_checkpoint}")
            checkpoint = torch.load(cfg.paths.base_checkpoint, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Model loaded successfully")
        else:
            print(f"⚠️ No base checkpoint found, using random weights")
        
        model.eval()
        
        # Test evaluation call (with minimal samples for speed)
        print(f"Testing evaluation call...")
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,
                output_dir=temp_dir,
                is_final_round=False
            )
            
            if 'error' in results:
                print(f"❌ Evaluation failed: {results['error']}")
            else:
                print(f"🎉 SUCCESS: Evaluation completed!")
                print(f"   Sample results keys: {list(results.keys())}")
                
                # Show a few key metrics
                for key in ['recovery', 'perplexity', 'sc_score_eternafold']:
                    if key in results:
                        print(f"   {key}: {results[key]}")
                
                return results
    
    except Exception as e:
        print(f"❌ Evaluation call failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 MultiRound Evaluator Test Suite")
    print("Testing the fixed evaluation dataset loading and evaluation process.")
    
    # Test 1: Evaluator creation and dataset loading
    evaluator = test_fixed_evaluator()
    
    # Test 2: Evaluation call (if dataset loading worked)
    if evaluator and evaluator.eval_dataset is not None:
        results = test_evaluation_call()
        
        if results and 'error' not in results:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   The fixed evaluator successfully loads the dataset and runs evaluation.")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            print(f"   Dataset loading works, but evaluation call failed.")
    else:
        print(f"\n❌ TESTS FAILED")
        print(f"   Dataset loading still doesn't work.")

if __name__ == "__main__":
    main()