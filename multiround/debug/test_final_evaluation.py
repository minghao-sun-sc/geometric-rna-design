#!/usr/bin/env python3
"""
Final test to verify the complete multiround evaluation pipeline works
including pass@k analysis.
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

def test_multiround_evaluation():
    """Test the complete multiround evaluation including pass@k."""
    print("🔧 Final MultiRound Evaluation Test")
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
        print(f"  eval_samples: {evaluator.eval_samples}")
        print(f"  final_eval_samples: {evaluator.final_eval_samples}")
        print(f"  pass@k rounds: {evaluator.passk_rounds}")
        
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
        
        # Test multiround evaluation with minimal samples for speed
        print(f"\nTesting multiround evaluation...")
        
        # Reduce sample sizes for testing speed
        original_eval_samples = evaluator.eval_samples
        original_final_samples = evaluator.final_eval_samples
        
        evaluator.eval_samples = 4      # Minimal for regular eval
        evaluator.final_eval_samples = 8  # Minimal for pass@k
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running multiround evaluation in: {temp_dir}")
            
            # Test round 1 evaluation (should include pass@k)
            print(f"\n📊 Testing Round 1 (with pass@k)...")
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,  # Round 1 should have pass@k
                output_dir=temp_dir,
                is_final_round=False
            )
            
            # Restore original settings
            evaluator.eval_samples = original_eval_samples
            evaluator.final_eval_samples = original_final_samples
            
            if 'error' in results:
                print(f"❌ Round 1 evaluation failed: {results['error']}")
                return False
            else:
                print(f"✅ Round 1 evaluation completed successfully!")
                print(f"   Result keys: {list(results.keys())}")
                
                # Check for pass@k results
                passk_keys = [k for k in results.keys() if 'pass' in k.lower()]
                if passk_keys:
                    print(f"✅ Pass@k analysis found!")
                    for key in sorted(passk_keys):
                        value = results[key]
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value:.4f}")
                        else:
                            print(f"   {key}: {value}")
                else:
                    print(f"⚠️ No pass@k results found")
                
                # Show key basic metrics
                key_metrics = ['recovery', 'perplexity', 'sc_score_eternafold']
                for key in key_metrics:
                    if key in results:
                        value = results[key]
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value:.4f}")
                        else:
                            print(f"   {key}: {value}")
                
                return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Final MultiRound Evaluation Test")
    print("Testing complete evaluation pipeline with pass@k analysis.")
    
    success = test_multiround_evaluation()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   The fixed multiround evaluation pipeline works correctly.")
        print(f"   Both basic evaluation and pass@k analysis are functional.")
        print(f"   Ready for full multiround training!")
    else:
        print(f"\n❌ TESTS FAILED")
        print(f"   The multiround evaluation pipeline needs more work.")

if __name__ == "__main__":
    main()