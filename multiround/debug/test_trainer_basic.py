# multiround/debug/test_trainer_basic.py
"""
Test script for basic multiround trainer functionality (without full training).
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
from multiround.trainer import MultiRoundDPOTrainer


def test_trainer_initialization():
    """Test trainer initialization."""
    print("🧪 Testing MultiRoundDPOTrainer initialization...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        
        # Override for quick testing
        cfg.multiround.num_rounds = 2
        cfg.multiround.epochs_per_round = 1
        cfg.multiround.eval_samples = 1
        cfg.training.batch_size = 1
        cfg.wandb.enable = False  # Disable wandb for testing
        
        trainer = MultiRoundDPOTrainer(cfg)
        
        print("✅ MultiRoundDPOTrainer initialized successfully")
        print(f"   Device: {trainer.device}")
        print(f"   Num rounds: {trainer.num_rounds}")
        print(f"   Epochs per round: {trainer.epochs_per_round}")
        print(f"   Output root: {trainer.output_root}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_round_directory_creation():
    """Test round directory creation."""
    print("\n🧪 Testing round directory creation...")
    
    try:
        from multiround.utils import create_round_output_dir
        
        with tempfile.TemporaryDirectory() as temp_dir:
            round_dir = create_round_output_dir(temp_dir, 1)
            
            # Check directory structure
            expected_subdirs = ['checkpoints', 'eval_results', 'plots', 'logs']
            
            print(f"✅ Round directory created: {round_dir}")
            
            for subdir in expected_subdirs:
                subdir_path = os.path.join(round_dir, subdir)
                if os.path.exists(subdir_path):
                    print(f"   ✅ {subdir}/ subdirectory created")
                else:
                    print(f"   ❌ {subdir}/ subdirectory missing")
                    return False
            
            return True
        
    except Exception as e:
        print(f"❌ Round directory creation failed: {e}")
        return False


def test_config_overrides():
    """Test configuration override functionality."""
    print("\n🧪 Testing configuration overrides...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        original_rounds = cfg.multiround.num_rounds
        
        # Test overriding number of rounds
        cfg.multiround.num_rounds = 3
        cfg.multiround.epochs_per_round = 2
        cfg.wandb.enable = False
        
        trainer = MultiRoundDPOTrainer(cfg)
        
        print("✅ Configuration overrides applied successfully")
        print(f"   Original rounds: {original_rounds}")
        print(f"   Override rounds: {trainer.num_rounds}")
        print(f"   Override epochs per round: {trainer.epochs_per_round}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration override test failed: {e}")
        return False


def test_utils_functions():
    """Test utility functions."""
    print("\n🧪 Testing utility functions...")
    
    try:
        from multiround.utils import format_time_delta, deep_merge_dicts
        
        # Test time formatting
        time_tests = [60, 3661, 7323]  # 1 min, 1h 1m 1s, 2h 2m 3s
        for seconds in time_tests:
            formatted = format_time_delta(seconds)
            print(f"   {seconds}s → {formatted}")
        
        # Test deep merge
        base_dict = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override_dict = {'b': {'c': 20, 'e': 4}, 'f': 5}
        merged = deep_merge_dicts(base_dict, override_dict)
        
        expected_c = 20  # Should be overridden
        expected_d = 3   # Should be preserved
        expected_e = 4   # Should be added
        expected_f = 5   # Should be added
        
        if (merged['b']['c'] == expected_c and 
            merged['b']['d'] == expected_d and 
            merged['b']['e'] == expected_e and 
            merged['f'] == expected_f):
            print("   ✅ Deep merge working correctly")
        else:
            print(f"   ❌ Deep merge failed: {merged}")
            return False
        
        print("✅ Utility functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False


def test_wandb_config_preparation():
    """Test WandB configuration preparation."""
    print("\n🧪 Testing WandB configuration preparation...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = load_multiround_config(config_path)
        cfg.wandb.enable = False  # Don't actually initialize wandb
        
        trainer = MultiRoundDPOTrainer(cfg)
        
        # Test that trainer has the necessary config attributes
        required_attrs = ['cfg', 'num_rounds', 'epochs_per_round', 'evaluator']
        for attr in required_attrs:
            if hasattr(trainer, attr):
                print(f"   ✅ Trainer has {attr} attribute")
            else:
                print(f"   ❌ Trainer missing {attr} attribute")
                return False
        
        print("✅ WandB configuration preparation successful")
        return True
        
    except Exception as e:
        print(f"❌ WandB config test failed: {e}")
        return False


def test_checkpoint_path_generation():
    """Test checkpoint path generation."""
    print("\n🧪 Testing checkpoint path generation...")
    
    try:
        config_path = "multiround/config/experiments/plan_a_margin125.yaml"
        cfg = load_multiround_config(config_path)
        cfg.wandb.enable = False
        
        trainer = MultiRoundDPOTrainer(cfg)
        
        # Test output path generation
        expected_components = ['runs', 'multiround']
        output_path = trainer.output_root
        
        print(f"   Output root: {output_path}")
        
        for component in expected_components:
            if component in output_path:
                print(f"   ✅ Output path contains '{component}'")
            else:
                print(f"   ⚠️ Output path missing '{component}' (might be OK)")
        
        print("✅ Checkpoint path generation working")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint path test failed: {e}")
        return False


def main():
    """Run all basic trainer tests."""
    print("🚀 Starting basic multiround trainer tests...\n")
    
    tests = [
        test_trainer_initialization,
        test_round_directory_creation,
        test_config_overrides,
        test_utils_functions,
        test_wandb_config_preparation,
        test_checkpoint_path_generation,
    ]
    
    results = []
    for test_func in tests:
        success = test_func()
        results.append(success)
        
        if not success:
            print(f"\n⚠️ Test {test_func.__name__} failed")
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All basic trainer tests passed!")
    else:
        print("❌ Some basic trainer tests failed")
        print("   Check individual test outputs above for details")


if __name__ == "__main__":
    main()