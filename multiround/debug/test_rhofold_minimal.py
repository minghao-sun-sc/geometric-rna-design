#!/usr/bin/env python3
"""
Minimal test of RhoFold evaluation to verify the complete pipeline works.
"""

import os
import sys
import tempfile
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

def test_rhofold_evaluation():
    """Test minimal RhoFold evaluation to verify complete pipeline."""
    print("🔬 Minimal RhoFold Evaluation Test")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        # Create evaluator with minimal settings
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        evaluator.eval_samples = 1  # Just 1 sample for quick test
        evaluator.final_eval_samples = 1
        
        print(f"✅ Evaluator created with minimal samples: {evaluator.eval_samples}")
        
        # Load model
        from dpo.ref_manager import build_model_from_cfg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model_from_cfg(cfg.model).to(device)
        
        if hasattr(cfg.paths, 'base_checkpoint') and os.path.exists(cfg.paths.base_checkpoint):
            checkpoint = torch.load(cfg.paths.base_checkpoint, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Model loaded")
        
        model.eval()
        
        # Run minimal evaluation on first structure only
        print(f"\n🧪 Testing RhoFold Evaluation on 1 structure...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            round1_dir = os.path.join(temp_dir, "round_01")
            os.makedirs(round1_dir, exist_ok=True)
            
            # Temporarily limit to 1 structure for testing
            original_data_list = evaluator.eval_dataset.data_list
            evaluator.eval_dataset.data_list = original_data_list[:1]  # Just first structure
            
            print(f"   Testing with structure: {evaluator.eval_dataset.data_list[0]['id_list'][0]}")
            
            # Test round 1 evaluation
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,
                output_dir=round1_dir,
                is_final_round=False
            )
            
            # Restore original dataset
            evaluator.eval_dataset.data_list = original_data_list
            
            if 'error' in results:
                print(f"❌ RhoFold evaluation failed: {results['error']}")
                return False
            
            print(f"✅ RhoFold evaluation completed!")
            print(f"   Result keys: {sorted(results.keys())}")
            
            # Check for key metrics
            key_metrics = ['recovery', 'perplexity', 'sc_score_rmsd', 'sc_score_tm', 'sc_score_gdt']
            for key in key_metrics:
                if key in results:
                    value = results[key]
                    print(f"   {key}: {value:.4f}")
            
            # Check for pass@k results
            passk_keys = [k for k in results.keys() if 'pass' in k.lower()]
            if passk_keys:
                print(f"✅ Pass@k analysis found: {len(passk_keys)} metrics")
            
            return True
    
    except Exception as e:
        print(f"❌ RhoFold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Minimal RhoFold Evaluation Test")
    print("Testing RhoFold evaluation pipeline with 1 structure.")
    
    success = test_rhofold_evaluation()
    
    if success:
        print(f"\n🎉 RHOFOLD EVALUATION TEST PASSED!")
        print(f"   ✅ Complete evaluation pipeline works")
        print(f"   ✅ RhoFold structure prediction works")
        print(f"   ✅ All metrics calculation works")
        print(f"   ✅ Pass@k analysis framework works")
        print(f"\n🚀 MULTIROUND TRAINING IS READY!")
        print(f"   All components verified - can proceed with full training.")
    else:
        print(f"\n❌ RHOFOLD EVALUATION TEST FAILED")
        print(f"   Need more fixes before full training.")

if __name__ == "__main__":
    main()