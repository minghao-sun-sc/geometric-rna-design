#!/usr/bin/env python3
"""
Test script for pass@k integration with dpo.bench.eval_full.py

This script validates that:
1. Pass@k functions can be imported from multiround.passk
2. Configuration parsing works for eval_full.py 
3. The new pass@k integration functions correctly
4. Error handling works properly

Usage:
    cd /mnt/rna01/smh/projects/ribopo
    python multiround/debug/test_eval_full_passk.py
"""

import os
import sys
import tempfile
import json
import numpy as np
from types import SimpleNamespace as SN

def test_passk_imports():
    """Test that pass@k functions can be imported correctly."""
    print("🧪 Testing pass@k imports...")
    
    try:
        # Test core pass@k functions
        from multiround.passk import (
            Rule, pass_at_k_from_metrics, calculate_passk_metrics, 
            plot_metric_distributions
        )
        print("✅ Core pass@k functions imported successfully")
        
        # Test that eval_full.py can import the functions
        sys.path.insert(0, 'dpo/bench')
        import eval_full
        
        if hasattr(eval_full, 'PASSK_AVAILABLE'):
            print(f"✅ PASSK_AVAILABLE flag: {eval_full.PASSK_AVAILABLE}")
            if eval_full.PASSK_AVAILABLE:
                print("✅ Pass@k module is available for eval_full.py")
            else:
                print("❌ Pass@k module not available in eval_full.py")
                return False
        else:
            print("❌ PASSK_AVAILABLE flag not found in eval_full.py")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


def test_calculate_passk_metrics():
    """Test the calculate_passk_metrics function with synthetic data."""
    print("\n🧪 Testing calculate_passk_metrics function...")
    
    try:
        from multiround.passk import calculate_passk_metrics
        
        # Create synthetic data in the format expected by eval_full.py
        np.random.seed(42)
        
        metrics_by_structure = {}
        for struct_idx in range(3):  # 3 structures
            struct_id = f"test_struct_{struct_idx}"
            samples = []
            
            for sample_idx in range(8):  # 8 samples each
                sample_metrics = {
                    "structure_id": struct_id,
                    "sample_idx": sample_idx,
                    "sc_tm": np.random.uniform(0.1, 0.8),     # TM-score
                    "sc_rmsd": np.random.uniform(1.0, 15.0),  # RMSD
                    "sc_plddt": np.random.uniform(0.3, 0.9),  # pLDDT
                    "vienna_mfe": np.random.uniform(-25.0, -5.0),  # MFE
                    "recovery": np.random.uniform(0.2, 0.8),  # Recovery
                }
                samples.append(sample_metrics)
            
            metrics_by_structure[struct_id] = samples
        
        print(f"📊 Created test data: {len(metrics_by_structure)} structures")
        
        # Test pass@k calculation
        k_values = [1, 2, 4, 8]
        thresholds = {
            "tm_score": [0.4, 0.5],
            "rmsd": [8.0, 4.0],
            "mfe": [-15.0, -20.0]
        }
        
        results = calculate_passk_metrics(
            metrics_by_structure,
            k_values=k_values,
            thresholds=thresholds
        )
        
        # Validate results structure
        assert "n_structures" in results, "n_structures missing from results"
        assert "passk_results" in results, "passk_results missing from results"
        assert results["n_structures"] == 3, f"Expected 3 structures, got {results['n_structures']}"
        
        print(f"✅ Pass@k calculation successful")
        print(f"📈 Results structure: {list(results.keys())}")
        
        # Check that we have results for each threshold metric
        passk_results = results["passk_results"]
        for metric_name in thresholds.keys():
            if metric_name in passk_results:
                print(f"   {metric_name}: {len(passk_results[metric_name])} thresholds")
            else:
                print(f"   ⚠️ {metric_name}: no results (metric may not be available in test data)")
        
        return True
        
    except Exception as e:
        print(f"❌ calculate_passk_metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_metric_distributions():
    """Test the plot_metric_distributions function."""
    print("\n🧪 Testing plot_metric_distributions function...")
    
    try:
        from multiround.passk import plot_metric_distributions
        
        # Create synthetic individual metrics data
        np.random.seed(42)
        individual_metrics = []
        
        for struct_idx in range(3):
            for sample_idx in range(8):
                sample_metrics = {
                    "structure_id": f"test_struct_{struct_idx}",
                    "sample_idx": sample_idx,
                    "sc_plddt": np.random.uniform(0.3, 0.9),
                    "sc_rmsd": np.random.uniform(1.0, 15.0),
                    "sc_tm": np.random.uniform(0.1, 0.8),
                    "vienna_mfe": np.random.uniform(-25.0, -5.0),
                    "inf_all": np.random.uniform(0.2, 0.9),
                }
                individual_metrics.append(sample_metrics)
        
        print(f"📊 Created {len(individual_metrics)} individual metric samples")
        
        # Test plotting in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_metric_distributions(
                individual_metrics,
                output_dir=temp_dir,
                metrics_to_plot=["sc_plddt", "sc_rmsd", "sc_tm", "vienna_mfe"],
                checkpoint_name="test_checkpoint"
            )
            
            # Check if files were created
            expected_files = [
                "test_checkpoint_metric_distributions.png",
                "test_checkpoint_metric_statistics.json"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                if os.path.exists(filepath):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"✅ Created {filename} ({size_kb:.1f} KB)")
                else:
                    print(f"❌ Missing expected file: {filename}")
                    return False
        
        print("✅ plot_metric_distributions test passed")
        return True
        
    except Exception as e:
        print(f"❌ plot_metric_distributions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_parsing():
    """Test that pass@k configuration parsing works."""
    print("\n🧪 Testing configuration parsing...")
    
    try:
        # Create a test config in the format expected by eval_full.py
        config_data = {
            "eval": {
                "passk": {
                    "enable": True,
                    "k_values": [1, 2, 4, 8, 16, 32, 64],
                    "n_samples_passk": 64,
                    "thresholds": {
                        "tm_score": [0.4, 0.45, 0.5, 0.55],
                        "rmsd": [8.0, 6.0, 4.0, 2.0],
                        "mfe": [-10.0, -15.0, -20.0]
                    },
                    "collect_individual_metrics": True,
                    "plot_distributions": True,
                    "plot_metrics": ["plddt", "rmsd", "tm_score", "mfe", "inf_all"],
                    "save_passk_results": True,
                    "passk_output_dir": "passk_analysis"
                }
            }
        }
        
        # Convert to SimpleNamespace format (same as eval_full.py uses)
        def _to_sn(o):
            if isinstance(o, dict):
                return SN(**{k: _to_sn(v) for k, v in o.items()})
            if isinstance(o, list):
                return [_to_sn(x) for x in o]
            return o
        
        cfg = _to_sn(config_data)
        
        # Test configuration access (same as eval_full.py)
        passk_cfg = getattr(cfg.eval, 'passk', None) if hasattr(cfg, 'eval') else None
        
        assert passk_cfg is not None, "Pass@k config not found"
        assert getattr(passk_cfg, 'enable', False) == True, "Pass@k not enabled"
        assert len(getattr(passk_cfg, 'k_values', [])) == 7, "Wrong number of k values"
        assert getattr(passk_cfg, 'n_samples_passk', 0) == 64, "Wrong n_samples_passk"
        
        thresholds = getattr(passk_cfg, 'thresholds', {})
        assert hasattr(thresholds, 'tm_score'), "TM-score thresholds missing"
        assert len(thresholds.tm_score) == 4, "Wrong number of TM-score thresholds"
        
        print("✅ Configuration parsing successful")
        print(f"📝 Config: enable={passk_cfg.enable}, k_values={len(passk_cfg.k_values)}")
        print(f"         thresholds: tm={len(thresholds.tm_score)}, rmsd={len(thresholds.rmsd)}, mfe={len(thresholds.mfe)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_config_file():
    """Test that the template config file has pass@k settings."""
    print("\n🧪 Testing template configuration file...")
    
    template_path = "multiround/config/evaluation/eval_run/00_debug_cons_ckpt.yaml"
    
    if not os.path.exists(template_path):
        print(f"❌ Template config not found: {template_path}")
        return False
    
    try:
        import yaml
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if pass@k section exists
        if 'eval' in config and 'passk' in config['eval']:
            passk_config = config['eval']['passk']
            print("✅ Pass@k configuration found in template")
            
            # Check key fields
            required_fields = ['enable', 'k_values', 'thresholds']
            for field in required_fields:
                if field in passk_config:
                    print(f"✅ Required field '{field}' present")
                else:
                    print(f"❌ Required field '{field}' missing")
                    return False
            
            # Check thresholds structure
            thresholds = passk_config.get('thresholds', {})
            expected_metrics = ['tm_score', 'rmsd', 'mfe']
            for metric in expected_metrics:
                if metric in thresholds and isinstance(thresholds[metric], list):
                    print(f"✅ Thresholds for '{metric}': {len(thresholds[metric])} values")
                else:
                    print(f"❌ Invalid thresholds for '{metric}'")
                    return False
        
        else:
            print("❌ Pass@k configuration not found in template")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Template config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🎯 Pass@k Integration Tests for dpo.bench.eval_full.py")
    print("=" * 60)
    
    tests = [
        ("Pass@k Imports", test_passk_imports),
        ("calculate_passk_metrics", test_calculate_passk_metrics),
        ("plot_metric_distributions", test_plot_metric_distributions),
        ("Configuration Parsing", test_config_parsing),
        ("Template Config File", test_template_config_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! eval_full.py pass@k integration is ready!")
        print("\n✨ You can now use pass@k analysis in dpo.bench.eval_full.py by:")
        print("   1. Setting passk.enable: true in your config")
        print("   2. Running: python -m dpo.bench.eval_full --config your_config.yaml")
        print("   3. Check the output for pass@k results and distribution plots")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)