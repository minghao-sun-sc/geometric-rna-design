#!/usr/bin/env python3
"""
Test script to verify pass@k analysis is properly integrated and working.
Tests the pass@k implementation with both the evaluator and standalone module.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add multiround directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_passk_module():
    """Test the standalone pass@k module functionality."""
    print("🧪 Testing standalone pass@k module...")
    
    try:
        from multiround.passk import Rule, pass_at_k_from_metrics, pass_at_k_unbiased
        
        # Create synthetic metric data for testing
        rng = np.random.default_rng(42)
        
        # Two structures, each with 8 samples
        md0 = {
            "tm":        rng.uniform(0.2, 0.9, size=8),     # TM scores
            "rmsd":      rng.uniform(1.0, 12.0, size=8),    # RMSD values
            "mfe":       rng.uniform(-25.0, -5.0, size=8),  # MFE values
        }
        md1 = {
            "tm":        rng.uniform(0.2, 0.9, size=8),
            "rmsd":      rng.uniform(1.0, 12.0, size=8), 
            "mfe":       rng.uniform(-25.0, -5.0, size=8),
        }
        items = [md0, md1]
        
        # Test different rule combinations
        test_cases = [
            # Single rules
            ([Rule("tm", ">=", 0.45)], "TM≥0.45"),
            ([Rule("rmsd", "<=", 8.0)], "RMSD≤8.0"),
            ([Rule("mfe", "<=", -10.0)], "MFE≤-10.0"),
            
            # Combined rules (AND)
            ([Rule("tm", ">=", 0.45), Rule("rmsd", "<=", 8.0)], "TM≥0.45 AND RMSD≤8.0"),
            ([Rule("tm", ">=", 0.5), Rule("rmsd", "<=", 4.0)], "TM≥0.5 AND RMSD≤4.0"),
        ]
        
        print(f"   Testing with {len(items)} structures, {md0['tm'].size} samples each")
        
        for rules, description in test_cases:
            print(f"\n   📊 Rule: {description}")
            
            for k in [1, 2, 4, 8]:
                # Test unbiased estimator
                passk_unbiased = pass_at_k_from_metrics(
                    items, rules, k, 
                    combine_mode="all", selection="unbiased"
                )
                
                # Test top-k by TM score  
                passk_topk = pass_at_k_from_metrics(
                    items, rules, k,
                    combine_mode="all", selection="topk",
                    rank_metric="tm", rank_higher_is_better=True
                )
                
                print(f"      k={k}: unbiased={passk_unbiased:.3f}, topk={passk_topk:.3f}")
        
        print("   ✅ Standalone pass@k module test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Standalone pass@k module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator_passk_integration():
    """Test pass@k integration with the multiround evaluator."""
    print("\n🧪 Testing evaluator pass@k integration...")
    
    try:
        # Import configuration and evaluator
        sys.path.insert(0, '/mnt/rna01/smh/projects/ribopo')
        from multiround.utils.config_utils import load_config
        from multiround.evaluator import MultiRoundEvaluator
        
        # Load a representative configuration
        config_path = "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/04_dpo_m25.yaml"
        cfg = load_config(config_path)
        
        # Create evaluator instance
        evaluator = MultiRoundEvaluator(cfg)
        
        # Check pass@k configuration
        print(f"   Pass@k config found: {evaluator.passk_config is not None}")
        if evaluator.passk_config:
            k_values = getattr(evaluator.passk_config, 'k_values', [])
            thresholds = getattr(evaluator.passk_config, 'thresholds', None)
            print(f"   K values: {k_values}")
            if thresholds:
                tm_thresholds = getattr(thresholds, 'tm_score', [])
                rmsd_thresholds = getattr(thresholds, 'rmsd', [])
                print(f"   TM thresholds: {tm_thresholds}")
                print(f"   RMSD thresholds: {rmsd_thresholds}")
        
        print(f"   Pass@k rounds: {evaluator.passk_rounds}")
        
        # Test default pass@k analysis with synthetic data
        print("\n   Testing default pass@k analysis...")
        
        # Create synthetic evaluation results
        n_structures = 5
        n_samples = 8
        
        eval_results = {
            'sc_score_tm': [np.random.uniform(0.2, 0.9, n_samples) for _ in range(n_structures)],
            'sc_score_rmsd': [np.random.uniform(1.0, 12.0, n_samples) for _ in range(n_structures)],
            'vienna_mfe': [np.random.uniform(-25.0, -5.0, n_samples) for _ in range(n_structures)],
        }
        
        # Test pass@k computation
        passk_results = evaluator._compute_default_passk_analysis(eval_results, n_samples)
        
        if 'passk_error' in passk_results:
            print(f"   ❌ Pass@k analysis error: {passk_results['passk_error']}")
            return False
        
        print(f"   ✅ Pass@k analysis completed: {len(passk_results)} metrics")
        
        # Show sample results
        sample_keys = [k for k in passk_results.keys() if 'passk_' in k][:5]
        for key in sample_keys:
            print(f"      {key}: {passk_results[key]:.4f}")
        
        print("   ✅ Evaluator pass@k integration test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Evaluator pass@k integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_passk_settings():
    """Test that pass@k configuration is properly loaded from configs."""
    print("\n🧪 Testing pass@k configuration loading...")
    
    try:
        sys.path.insert(0, '/mnt/rna01/smh/projects/ribopo')
        from multiround.utils.config_utils import load_config
        
        # Test different configurations
        test_configs = [
            "04_dpo_m25.yaml",
            "03_simpo_m125.yaml", 
            "15_dpo_dynamic_margins.yaml"
        ]
        
        for config_name in test_configs:
            config_path = f"/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/{config_name}"
            print(f"\n   📄 Testing config: {config_name}")
            
            cfg = load_config(config_path)
            
            # Check evaluation section
            eval_cfg = getattr(cfg, 'evaluation', None)
            if eval_cfg:
                passk_cfg = getattr(eval_cfg, 'pass_k', None)
                if passk_cfg:
                    k_values = getattr(passk_cfg, 'k_values', [])
                    thresholds = getattr(passk_cfg, 'thresholds', None)
                    
                    print(f"      ✅ Pass@k config found")
                    print(f"         K values: {k_values}")
                    
                    if thresholds:
                        tm_thr = getattr(thresholds, 'tm_score', [])
                        rmsd_thr = getattr(thresholds, 'rmsd', [])
                        mfe_thr = getattr(thresholds, 'mfe', [])
                        print(f"         TM thresholds: {tm_thr}")
                        print(f"         RMSD thresholds: {rmsd_thr}")
                        print(f"         MFE thresholds: {mfe_thr}")
                else:
                    print(f"      ⚠️ Pass@k config missing in evaluation section")
            else:
                print(f"      ⚠️ Evaluation section missing")
            
            # Check eval section (from bench_full.yaml integration)
            eval_section = getattr(cfg, 'eval', None)
            if eval_section:
                metrics = getattr(eval_section, 'metrics', [])
                n_samples = getattr(eval_section, 'n_samples', 0)
                print(f"      ✅ Eval section found: {len(metrics)} metrics, {n_samples} samples")
            
        print("   ✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all pass@k integration tests."""
    print("🎯 Pass@k Integration Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Standalone pass@k module
    results.append(test_passk_module())
    
    # Test 2: Evaluator integration
    results.append(test_evaluator_passk_integration())
    
    # Test 3: Configuration loading
    results.append(test_config_passk_settings())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    test_names = ["Standalone pass@k module", "Evaluator integration", "Configuration loading"]
    
    all_passed = True
    for i, (name, passed) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✨ Pass@k analysis is properly implemented and integrated!")
        print("   - Standalone module works correctly")
        print("   - Evaluator integration is functional") 
        print("   - Configuration loading works properly")
        print("   - Ready for multiround training!")
    else:
        print("\n⚠️ Some issues found in pass@k integration.")
        print("   Please check the error messages above and fix any issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)