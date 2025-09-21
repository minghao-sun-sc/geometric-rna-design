#!/usr/bin/env python3
"""
Test fast debug configuration files to verify skip_intermediate_passk functionality.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def test_fast_config(config_path):
    """Test a fast configuration to verify skip_intermediate_passk works."""
    print(f"\n🚀 Testing {os.path.basename(config_path)}")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print(f"✅ Config loaded successfully")
        
        # Check fast training settings
        print(f"⚡ Fast Training Settings:")
        skip_passk = getattr(cfg.multiround, 'skip_intermediate_passk', False)
        print(f"   Skip intermediate pass@k: {skip_passk}")
        print(f"   Num rounds: {cfg.multiround.num_rounds}")
        print(f"   Final eval samples: {cfg.multiround.n_samples_final_eval}")
        
        # Test evaluator setup
        print(f"🔬 Testing Evaluator with Fast Settings:")
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        
        print(f"   Pass@k rounds: {evaluator.passk_rounds}")
        print(f"   Dataset: {len(evaluator.eval_dataset.data_list)} structures")
        
        # Verify pass@k behavior
        expected_passk = [cfg.multiround.num_rounds] if skip_passk else [1, 3, 5]
        # Ensure final round is included
        if cfg.multiround.num_rounds not in expected_passk:
            expected_passk.append(cfg.multiround.num_rounds)
        expected_passk = sorted(list(set(expected_passk)))
        
        actual_passk = sorted(evaluator.passk_rounds)
        
        if actual_passk == expected_passk:
            print(f"✅ Pass@k rounds correct: {actual_passk}")
        else:
            print(f"❌ Pass@k rounds mismatch:")
            print(f"   Expected: {expected_passk}")
            print(f"   Actual: {actual_passk}")
            return False
        
        # Calculate time savings
        if skip_passk:
            rounds_saved = cfg.multiround.num_rounds - 1  # All rounds except final
            time_saved_min = rounds_saved * 20  # ~20 min per pass@k analysis
            print(f"💰 Time Savings: ~{time_saved_min} minutes ({rounds_saved} rounds × 20 min/round)")
        
        # Test round evaluation logic
        print(f"🔍 Testing Round Evaluation Logic:")
        for round_num in range(1, cfg.multiround.num_rounds + 1):
            will_do_passk = round_num in evaluator.passk_rounds
            action = "PASS@K" if will_do_passk else "basic eval"
            print(f"   ✅ Round {round_num}: {action}")
            
        # Test that logic matches expected behavior
        if skip_passk:
            # Should only do pass@k on final round
            non_final_rounds = [r for r in range(1, cfg.multiround.num_rounds) if r in evaluator.passk_rounds]
            if non_final_rounds:
                print(f"   ❌ ERROR: Pass@k enabled on non-final rounds: {non_final_rounds}")
                return False
            else:
                print(f"   ✅ Pass@k correctly limited to final round only")
        
        return True
        
    except Exception as e:
        print(f"❌ {os.path.basename(config_path)} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test fast configuration functionality."""
    print("🚀 Fast Configuration Testing")
    print("Testing skip_intermediate_passk functionality for faster training.")
    
    configs_to_test = [
        "multiround/config/experiments/00_debug_a100_80g_fast.yaml",
        "multiround/config/experiments/00_debug_a40_fast.yaml"
    ]
    
    results = {}
    for config_path in configs_to_test:
        if os.path.exists(config_path):
            results[config_path] = test_fast_config(config_path)
        else:
            print(f"\n❌ {config_path} - FILE NOT FOUND")
            results[config_path] = False
    
    # Final summary
    print(f"\n🏁 FAST CONFIG SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for config_path, passed in results.items():
        config_name = os.path.basename(config_path)
        status = "✅ READY" if passed else "❌ NEEDS FIX"
        print(f"{status} {config_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🚀 ALL FAST CONFIGS READY!")
        print(f"   Fast training mode implemented and working.")
        print(f"   Significant time savings on intermediate rounds.")
        print(f"\n⚡ FAST TRAINING BENEFITS:")
        print(f"   • Skip pass@k on round 1 (saves ~20 minutes)")
        print(f"   • Only compute pass@k on final round")
        print(f"   • Still get basic evaluation metrics all rounds")
        print(f"   • Complete pass@k analysis on final results")
        print(f"\n💡 Fast Training Commands:")
        print(f"   # A100 Fast:")
        print(f"   python -m multiround.train --config multiround/config/experiments/00_debug_a100_80g_fast.yaml")
        print(f"   # A40 Fast:")
        print(f"   python -m multiround.train --config multiround/config/experiments/00_debug_a40_fast.yaml")
    else:
        print(f"\n⚠️ SOME FAST CONFIGS NEED FIXES")

if __name__ == "__main__":
    main()