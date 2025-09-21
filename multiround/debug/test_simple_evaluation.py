#!/usr/bin/env python3
"""
Simple test to verify evaluation pipeline works with coordinate fixes.
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

def test_simple_evaluation():
    """Test just the key parts without Vienna RNA features."""
    print("🔧 Simple Evaluation Test")
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
        
        # Test basic evaluation (without ViennaRNA features that cause issues)
        print(f"Testing basic evaluation...")
        
        # Temporarily modify the evaluation to use minimal metrics
        original_eval_samples = evaluator.eval_samples
        original_final_samples = evaluator.final_eval_samples
        
        # Use very small sample size for testing
        evaluator.eval_samples = 2  
        evaluator.final_eval_samples = 4  
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running basic evaluation in: {temp_dir}")
            
            # Test basic metrics only (no Vienna features)
            from src.evaluator import evaluate
            
            # Use just a subset of metrics to avoid Vienna RNA issues for now
            basic_metrics = [
                'recovery', 'perplexity', 'sc_eternafold'  
                # Remove Vienna and other complex metrics for this test
            ]
            
            try:
                eval_results = evaluate(
                    model=model,
                    dataset=evaluator.eval_dataset,
                    n_samples=2,  # Very small for testing
                    temperature=0.5,
                    device=device,
                    metrics=basic_metrics,
                    save_designs=False
                )
                
                print(f"🎉 SUCCESS: Basic evaluation completed!")
                print(f"   Result keys: {list(eval_results.keys())}")
                
                # Show key metrics
                key_metrics = ['recovery', 'perplexity', 'sc_score_eternafold']
                for key in key_metrics:
                    if key in eval_results:
                        values = eval_results[key]
                        if isinstance(values, list) and len(values) > 0:
                            print(f"   {key}: {values[0]:.4f}")
                        else:
                            print(f"   {key}: {values}")
                
                # Restore original settings
                evaluator.eval_samples = original_eval_samples
                evaluator.final_eval_samples = original_final_samples
                
                return True
                
            except Exception as e:
                print(f"❌ Basic evaluation failed: {e}")
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
    print("🧪 Simple Evaluation Test")
    print("Testing basic evaluation functionality with coordinate fixes.")
    
    success = test_simple_evaluation()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"   Basic evaluation pipeline is working correctly.")
    else:
        print(f"\n❌ FAILED")
        print(f"   Basic evaluation pipeline needs more debugging.")

if __name__ == "__main__":
    main()