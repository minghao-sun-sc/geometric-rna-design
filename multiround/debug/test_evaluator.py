# multiround/debug/test_evaluator.py
"""
Test script for multiround evaluator functionality.
"""
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from multiround.train import load_multiround_config
from multiround.evaluator import MultiRoundEvaluator
from dpo.ref_manager import build_model_from_cfg


def test_evaluator_initialization():
    """Test evaluator initialization."""
    print("🧪 Testing MultiRoundEvaluator initialization...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        evaluator = MultiRoundEvaluator(cfg)
        
        print("✅ MultiRoundEvaluator initialized successfully")
        print(f"   Device: {evaluator.device}")
        print(f"   Eval samples: {evaluator.eval_samples}")
        print(f"   Final eval samples: {evaluator.final_eval_samples}")
        print(f"   Eval temperature: {evaluator.eval_temperature}")
        
        if evaluator.eval_dataset is not None:
            print(f"   Test dataset size: {len(evaluator.eval_dataset)} structures")
        else:
            print("   ⚠️ Test dataset not loaded")
            
        return True
        
    except Exception as e:
        print(f"❌ Evaluator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading the base model for evaluation."""
    print("\n🧪 Testing model loading...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        
        # Test loading base model
        model = build_model_from_cfg(cfg.model)
        model.eval()
        
        print("✅ Base model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Test model has expected attributes
        required_attrs = ['sample', 'out_dim']
        for attr in required_attrs:
            if hasattr(model, attr):
                print(f"   ✅ Model has {attr} attribute")
            else:
                print(f"   ⚠️ Model missing {attr} attribute")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_dry_run():
    """Test evaluation dry run with minimal settings."""
    print("\n🧪 Testing evaluation dry run...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        
        # Override settings for quick test
        cfg.multiround.eval_samples = 1  # Very small for quick test
        cfg.multiround.final_eval_samples = 2
        
        evaluator = MultiRoundEvaluator(cfg)
        
        if evaluator.eval_dataset is None:
            print("❌ Cannot run evaluation - dataset not loaded")
            return False
        
        # Load model
        model = build_model_from_cfg(cfg.model)
        model.eval()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   Using temp directory: {temp_dir}")
            
            # Test quick evaluation (round 1, not final)
            result = evaluator.evaluate_round(
                model=model,
                round_num=1,
                output_dir=temp_dir,
                is_final_round=False
            )
            
            if 'error' in result:
                print(f"❌ Evaluation failed: {result['error']}")
                return False
            
            print("✅ Evaluation dry run completed successfully")
            print(f"   Result keys: {list(result.keys())}")
            
            # Check for expected metrics
            expected_metrics = ['tm_mean', 'rmsd_mean', 'recovery']
            for metric in expected_metrics:
                if metric in result:
                    print(f"   ✅ {metric}: {result[metric]:.4f}")
                else:
                    print(f"   ⚠️ {metric}: not found")
            
            return True
        
    except Exception as e:
        print(f"❌ Evaluation dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_functions():
    """Test metric aggregation functions."""
    print("\n🧪 Testing metric aggregation...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_margin125.yaml"
        cfg = load_multiround_config(config_path)
        evaluator = MultiRoundEvaluator(cfg)
        
        # Create mock evaluation results
        mock_results = {
            'recovery_list': [0.5, 0.6, 0.55],
            'perplexity_list': [1.2, 1.3, 1.1],
            'sc_score_tm': [0.4, 0.5, 0.45],
            'sc_score_rmsd': [8.0, 7.5, 8.2],
            'samples_list': [[[1, 2, 3, 4]], [[2, 3, 4, 1]], [[3, 4, 1, 2]]]
        }
        
        # Test aggregation
        aggregated = evaluator._aggregate_evaluation_results(mock_results)
        
        print("✅ Metric aggregation completed")
        print(f"   Recovery mean: {aggregated.get('recovery', 'N/A')}")
        print(f"   TM mean: {aggregated.get('tm_mean', 'N/A')}")
        print(f"   RMSD mean: {aggregated.get('rmsd_mean', 'N/A')}")
        print(f"   N structures: {aggregated.get('n_structures', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all evaluator tests."""
    print("🚀 Starting multiround evaluator tests...\n")
    
    tests = [
        test_evaluator_initialization,
        test_model_loading,
        test_aggregation_functions,
        test_evaluation_dry_run,  # Most expensive test last
    ]
    
    results = []
    for test_func in tests:
        success = test_func()
        results.append(success)
        
        if not success:
            print(f"\n⚠️ Test {test_func.__name__} failed - stopping here")
            break
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All evaluator tests passed!")
    else:
        print("❌ Some evaluator tests failed")


if __name__ == "__main__":
    main()