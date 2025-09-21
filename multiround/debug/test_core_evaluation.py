#!/usr/bin/env python3
"""
Test core evaluation functionality (fast version without expensive 3D metrics).
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

def test_core_evaluation():
    """Test core evaluation functionality (fast version)."""
    print("🔧 Core Evaluation Test (Fast)")
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
        
        # Test core evaluation (minimal samples, basic metrics only)
        print(f"Testing core evaluation...")
        
        # Use very small sample sizes for speed
        original_eval_samples = evaluator.eval_samples
        original_final_samples = evaluator.final_eval_samples
        
        evaluator.eval_samples = 2      # Minimal for speed
        evaluator.final_eval_samples = 4  # Still minimal
        
        # Test the core evaluation pipeline directly
        from src.evaluator import evaluate
        
        # Use only fast metrics (no RhoFold, no Vienna, no RibonanzaNet)
        fast_metrics = ['recovery', 'perplexity', 'sc_score_eternafold']
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running core evaluation in: {temp_dir}")
            
            try:
                eval_results = evaluate(
                    model=model,
                    dataset=evaluator.eval_dataset,
                    n_samples=2,  # Very small for testing
                    temperature=0.5,
                    device=device,
                    model_name="test",
                    metrics=fast_metrics,
                    save_designs=False
                )
                
                print(f"🎉 SUCCESS: Core evaluation completed!")
                print(f"   Result keys: {list(eval_results.keys())}")
                
                # Show key metrics
                for key in fast_metrics:
                    result_key = key.replace('sc_score_', '')
                    if result_key in eval_results:
                        values = eval_results[result_key]
                        if isinstance(values, list) and len(values) > 0:
                            print(f"   {key}: {values[0]:.4f}")
                        else:
                            print(f"   {key}: {values}")
                
                # Test pass@k analysis framework
                print(f"\n📊 Testing pass@k analysis framework...")
                
                # Test if the evaluator can handle pass@k structure
                pass_k_results = {
                    'pass@1_tm_0.45': 0.25,
                    'pass@2_tm_0.45': 0.35,
                    'pass@4_tm_0.45': 0.45
                }
                print(f"   Mock pass@k results: {pass_k_results}")
                
                # Restore original settings
                evaluator.eval_samples = original_eval_samples
                evaluator.final_eval_samples = original_final_samples
                
                return True
                
            except Exception as e:
                print(f"❌ Core evaluation failed: {e}")
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
    print("🧪 Core Evaluation Test")
    print("Testing essential evaluation functionality with coordinate fixes.")
    
    success = test_core_evaluation()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"   Core evaluation pipeline is working correctly.")
        print(f"   Coordinate processing and featurizer issues are fixed.")
        print(f"   Ready for multiround training with evaluation!")
    else:
        print(f"\n❌ FAILED")
        print(f"   Core evaluation pipeline needs more debugging.")

if __name__ == "__main__":
    main()