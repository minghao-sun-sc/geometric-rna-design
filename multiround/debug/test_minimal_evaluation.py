#!/usr/bin/env python3
"""
Minimal test of evaluation functionality without testing the featurizer directly.
This tests the full evaluation pipeline to see if the multiround evaluator works.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import yaml
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

def test_minimal_evaluation():
    """Test just the evaluation call with minimal samples."""
    print("🔧 Minimal Evaluation Test")
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
        
        # Build and load model
        from dpo.ref_manager import build_model_from_cfg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = build_model_from_cfg(cfg.model).to(device)
        
        # Load base checkpoint for testing
        if hasattr(cfg.paths, 'base_checkpoint') and os.path.exists(cfg.paths.base_checkpoint):
            print(f"Loading base checkpoint: {cfg.paths.base_checkpoint}")
            checkpoint = torch.load(cfg.paths.base_checkpoint, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Model loaded successfully")
        else:
            print(f"⚠️ No base checkpoint found, using random weights")
        
        model.eval()
        
        # Test evaluation with minimal configuration for speed
        print(f"Testing evaluation (minimal config for speed)...")
        
        # Temporarily reduce sample sizes for speed
        original_eval_samples = evaluator.eval_samples
        original_final_samples = evaluator.final_eval_samples
        
        # Use small sample size for testing
        evaluator.eval_samples = 2  # Minimal samples for speed
        evaluator.final_eval_samples = 4  # Still minimal
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running evaluation in: {temp_dir}")
            
            # Test round evaluation (non-final)
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,
                output_dir=temp_dir,
                is_final_round=False  # Start with simpler case
            )
            
            # Restore original settings
            evaluator.eval_samples = original_eval_samples
            evaluator.final_eval_samples = original_final_samples
            
            if 'error' in results:
                print(f"❌ Evaluation failed: {results['error']}")
                return False
            else:
                print(f"🎉 SUCCESS: Evaluation completed!")
                print(f"   Result keys: {list(results.keys())}")
                
                # Show a few key metrics if they exist
                key_metrics = ['recovery', 'perplexity', 'sc_score_eternafold', 'sc_score_rmsd', 'sc_score_tm']
                for key in key_metrics:
                    if key in results:
                        value = results[key]
                        print(f"   {key}: {value:.4f}")
                
                return True
    
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pass_at_k():
    """Test pass@k analysis if basic evaluation works."""
    print("\n" + "=" * 60)
    print("🎯 Pass@k Analysis Test")
    print("=" * 60)
    
    try:
        # Load config and create evaluator
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        
        # Build model (don't need to load weights for pass@k test)
        from dpo.ref_manager import build_model_from_cfg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model_from_cfg(cfg.model).to(device)
        model.eval()
        
        # Test with round 1 (should have pass@k)
        print(f"Testing pass@k for round 1...")
        print(f"  Pass@k rounds: {evaluator.passk_rounds}")
        print(f"  Round 1 in pass@k rounds: {1 in evaluator.passk_rounds}")
        
        # Reduce sample sizes drastically for testing
        evaluator.eval_samples = 2
        evaluator.final_eval_samples = 8  # Minimum for pass@8
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test round 1 evaluation (should include pass@k)
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,  # Round 1 should have pass@k
                output_dir=temp_dir,
                is_final_round=False
            )
            
            if 'error' in results:
                print(f"❌ Pass@k test failed: {results['error']}")
                return False
            else:
                # Look for pass@k results
                passk_keys = [k for k in results.keys() if 'pass' in k.lower()]
                if passk_keys:
                    print(f"✅ Pass@k analysis found!")
                    for key in passk_keys:
                        print(f"   {key}: {results[key]}")
                    return True
                else:
                    print(f"⚠️ No pass@k results found in output")
                    print(f"   Available keys: {list(results.keys())}")
                    return False
    
    except Exception as e:
        print(f"❌ Pass@k test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Minimal MultiRound Evaluation Test")
    print("Testing basic evaluation and pass@k functionality")
    
    # Test 1: Basic evaluation
    eval_success = test_minimal_evaluation()
    
    # Test 2: Pass@k analysis (if basic evaluation worked)
    if eval_success:
        passk_success = test_pass_at_k()
        
        if passk_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   Both evaluation and pass@k analysis are working.")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            print(f"   Basic evaluation works, but pass@k analysis failed.")
    else:
        print(f"\n❌ TESTS FAILED")
        print(f"   Basic evaluation is not working.")

if __name__ == "__main__":
    main()