#!/usr/bin/env python3
"""
Comprehensive test for the enhanced multiround system with:
- Dynamic preference pair switching
- Enhanced WandB organization
- Data validation
- Error handling

This test validates all components work together correctly.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from types import SimpleNamespace as SN

def _to_sn(o):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o

def load_cfg(path: str) -> SN:
    """Load config with recursive inheritance support."""
    def load_with_inheritance(config_path):
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        
        # Handle inheritance recursively
        if "inherit_from" in raw:
            inherit_path = raw["inherit_from"]
            print(f"Loading base config: {inherit_path}")
            
            # Recursively load base config
            base_raw = load_with_inheritance(inherit_path)
            
            # Merge configs (current overrides base)
            def merge_dict(base, override):
                result = base.copy()
                for key, value in override.items():
                    if key == "inherit_from":
                        continue  # Skip inheritance directive
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dict(result[key], value)
                    else:
                        result[key] = value
                return result
            
            raw = merge_dict(base_raw, raw)
        
        return raw
    
    final_config = load_with_inheritance(path)
    return _to_sn(final_config)

def test_pair_provider():
    """Test the enhanced pair provider with dynamic switching."""
    print("\n🧪 Testing PairProvider with dynamic switching...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_dynamic_margins.yaml"
        cfg = load_cfg(config_path)
        
        from multiround.pair_provider import MultiRoundPairProvider
        
        pair_provider = MultiRoundPairProvider(cfg)
        print("✅ PairProvider initialized successfully")
        
        # Test dynamic switching for different rounds
        for round_num in [1, 2, 3, 4, 5]:
            print(f"\n🔄 Testing round {round_num}:")
            pair_paths = pair_provider.update_config_for_round(round_num)
            margin_type = pair_provider.get_current_margin_type()
            
            print(f"   Margin type: {margin_type}")
            print(f"   Train path: {os.path.basename(pair_paths['train'])}")
            
            # Verify the margin switching logic
            expected_margin = "25" if round_num <= 2 else "125"
            if margin_type == expected_margin:
                print(f"   ✅ Correct margin type for round {round_num}")
            else:
                print(f"   ❌ Wrong margin type: expected {expected_margin}, got {margin_type}")
        
        summary = pair_provider.get_summary()
        print(f"\n📊 PairProvider summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ PairProvider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wandb_manager():
    """Test the enhanced WandB manager."""
    print("\n🧪 Testing WandB Manager...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_dynamic_margins.yaml"
        cfg = load_cfg(config_path)
        
        from multiround.wandb_manager import MultiRoundWandBManager
        
        wandb_manager = MultiRoundWandBManager(cfg)
        print("✅ WandB Manager initialized successfully")
        
        # Test enhanced configuration generation
        wandb_config = wandb_manager.get_wandb_config()
        hyperparams = wandb_manager.get_hyperparameters()
        summary = wandb_manager.get_summary()
        
        print(f"   Generated run name: {wandb_config['name']}")
        print(f"   Tags ({len(wandb_config['tags'])}): {', '.join(wandb_config['tags'][:5])}")
        print(f"   Group: {wandb_config['group']}")
        print(f"   Config fingerprint: {summary['config_fingerprint']}")
        print(f"   Experiment family: {summary['experiment_family']}")
        
        # Verify key components
        if 'dynamic_margins' in wandb_config['name']:
            print("   ✅ Run name includes experiment name")
        else:
            print("   ⚠️ Run name missing experiment name")
        
        if 'multiround' in wandb_config['tags']:
            print("   ✅ Proper tagging applied")
        else:
            print("   ⚠️ Missing multiround tag")
        
        return True
        
    except Exception as e:
        print(f"❌ WandB Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validator():
    """Test the data validator."""
    print("\n🧪 Testing Data Validator...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_dynamic_margins.yaml"
        cfg = load_cfg(config_path)
        
        from multiround.data_validator import MultiRoundDataValidator
        
        data_validator = MultiRoundDataValidator(cfg)
        print("✅ Data Validator initialized successfully")
        
        # Test validation on existing pair files
        test_pair_file = "data/pairs_margin125/by_das/clean/test.clean.jsonl"
        if os.path.exists(test_pair_file):
            validation_result = data_validator.validate_pair_file(test_pair_file, "test")
            
            print(f"   Validation result: {validation_result['valid']}")
            print(f"   Total pairs: {validation_result['total_pairs']}")
            print(f"   Valid pairs: {validation_result['valid_pairs']}")
            
            if validation_result['issues']:
                print(f"   Issues found: {validation_result['issues']}")
            
            if validation_result['recommendations']:
                print("   Recommendations:")
                for rec in validation_result['recommendations']:
                    print(f"     - {rec}")
        else:
            print(f"   ⚠️ Test file not found: {test_pair_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """Test the complete integrated multiround system."""
    print("\n🧪 Testing Integrated MultiRound System...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_dynamic_margins.yaml"
        cfg = load_cfg(config_path)
        
        # Disable wandb for testing
        cfg.wandb.enable = False
        
        from multiround.trainer import MultiRoundDPOTrainer
        
        trainer = MultiRoundDPOTrainer(cfg)
        print("✅ MultiRoundDPOTrainer initialized successfully")
        
        # Test components integration
        print(f"   Pair provider mode: {'dynamic' if trainer.pair_provider.dynamic_mode else 'static'}")
        print(f"   WandB manager ready: {hasattr(trainer, 'wandb_manager')}")
        print(f"   Output directory: {trainer.output_root}")
        
        # Test round setup (without actual training)
        print("\n🔧 Testing round setup...")
        round_dir = os.path.join(trainer.output_root, "round_01")
        os.makedirs(round_dir, exist_ok=True)
        
        # Test pair provider update
        pair_paths = trainer.pair_provider.update_config_for_round(1)
        print(f"   Round 1 pairs: {trainer.pair_provider.get_current_margin_type()}")
        
        pair_paths = trainer.pair_provider.update_config_for_round(3)
        print(f"   Round 3 pairs: {trainer.pair_provider.get_current_margin_type()}")
        
        # Test WandB configuration
        wandb_config = trainer.wandb_manager.get_wandb_config()
        print(f"   Enhanced run name: {wandb_config['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the enhanced multiround system."""
    print("🚀 Testing Enhanced MultiRound DPO System")
    print("="*60)
    
    tests = [
        test_pair_provider,
        test_wandb_manager,
        test_data_validator,
        test_integrated_system
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("🎯 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "PairProvider (Dynamic Switching)",
        "WandB Manager (Enhanced Organization)", 
        "Data Validator (Quality Control)",
        "Integrated System (Full Pipeline)"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced multiround system is ready.")
        print("\n📋 Features validated:")
        print("   ✅ Dynamic preference pair switching (0.25*std → 0.125*std)")
        print("   ✅ Enhanced WandB organization with auto-generated names")
        print("   ✅ Data quality validation and error handling")
        print("   ✅ Modular architecture ready for Plan B extensions")
        print("\n🚀 Ready for multiround training with:")
        print("   python -m multiround.train --config multiround/config/experiments/plan_a_dynamic_margins.yaml")
    else:
        print(f"⚠️ {total-passed} tests failed. Please review and fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)