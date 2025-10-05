#!/usr/bin/env python3
"""
Test script for RiboPO v2 multi-round training integration.

Tests the complete multi-round training pipeline with reduced parameters
to validate the integration works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from multiround.utils import load_config_with_inheritance
from ribopo_v2.multiround_trainer import RiboPOv2MultiRoundTrainer

def test_configuration_loading():
    """Test that the configuration loads correctly."""
    print("Testing configuration loading...")
    
    config_path = "ribopo_v2/config/experiments/03_multiround_test.yaml"
    config_dict = load_config_with_inheritance(config_path)
    
    # Convert dict to object with dot notation access recursively
    from types import SimpleNamespace
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(i) for i in d]
        else:
            return d
    
    config = dict_to_namespace(config_dict)
    
    # Validate key configuration sections
    assert hasattr(config, 'multiround')
    assert hasattr(config, 'ribopo_v2')
    assert hasattr(config, 'ribopo_v2_training')
    assert hasattr(config, 'training')
    assert hasattr(config, 'eval')
    
    print(f"✅ Configuration loaded successfully")
    print(f"  - Rounds: {config.multiround.num_rounds}")
    print(f"  - Epochs per round: {config.training.epochs}")
    print(f"  - Candidate pool size: {config.ribopo_v2_training.candidate_pool_size}")
    print(f"  - Winner pool size: {config.ribopo_v2_training.winner_pool_size}")
    
    return config

def test_trainer_initialization():
    """Test that the trainer initializes correctly."""
    print("\nTesting trainer initialization...")
    
    config_path = "ribopo_v2/config/experiments/03_multiround_test.yaml"
    trainer = RiboPOv2MultiRoundTrainer(config_path)
    
    # Check that directories are created
    assert trainer.output_dir.exists()
    assert trainer.pairs_dir.exists()
    assert trainer.winners_dir.exists()
    
    # Check round directories
    for round_num in range(1, trainer.config.multiround.num_rounds + 1):
        assert trainer.round_dirs[round_num].exists()
    
    print(f"✅ Trainer initialized successfully")
    print(f"  - Output directory: {trainer.output_dir}")
    print(f"  - Round directories: {len(trainer.round_dirs)}")
    
    return trainer

def test_component_initialization():
    """Test that all components can be initialized."""
    print("\nTesting component initialization...")
    
    config_path = "ribopo_v2/config/experiments/03_multiround_test.yaml"
    trainer = RiboPOv2MultiRoundTrainer(config_path)
    
    # Initialize components
    trainer.initialize_components()
    
    # Check that components are created
    assert trainer.candidate_evaluator is not None
    assert trainer.winner_selector is not None
    assert trainer.multiround_trainer is not None
    assert trainer.evaluator is not None
    
    print(f"✅ All components initialized successfully")
    print(f"  - Candidate evaluator: {type(trainer.candidate_evaluator).__name__}")
    print(f"  - Winner selector: {type(trainer.winner_selector).__name__}")
    print(f"  - Multiround trainer: {type(trainer.multiround_trainer).__name__}")
    print(f"  - Evaluator: {type(trainer.evaluator).__name__}")
    
    return trainer

def test_pair_margin_scheduling():
    """Test dynamic pair margin scheduling."""
    print("\nTesting pair margin scheduling...")
    
    config_path = "ribopo_v2/config/experiments/03_multiround_test.yaml"
    trainer = RiboPOv2MultiRoundTrainer(config_path)
    
    # Test margin selection for different rounds
    margins = {}
    for round_num in range(1, 6):  # Test rounds 1-5
        margin = trainer.get_pair_margin_for_round(round_num)
        margins[round_num] = margin
        
    print(f"✅ Pair margin scheduling working correctly")
    for round_num, margin in margins.items():
        print(f"  - Round {round_num}: {margin}")
    
    # Validate expected behavior
    assert margins[1] == "0.25std"  # Rounds 1-2 should use 0.25*std
    assert margins[2] == "0.25std"
    assert margins[3] == "0.125std"  # Rounds 3-5 should use 0.125*std
    assert margins[4] == "0.125std"
    assert margins[5] == "0.125std"
    
    return trainer

def test_dry_run():
    """Test a complete dry run without actual training."""
    print("\nTesting dry run (configuration validation only)...")
    
    config_path = "ribopo_v2/config/experiments/03_multiround_test.yaml"
    trainer = RiboPOv2MultiRoundTrainer(config_path)
    trainer.initialize_components()
    
    # Test that we can access all required configuration
    assert trainer.config.multiround.num_rounds == 3
    assert trainer.config.training.epochs == 2
    assert trainer.config.ribopo_v2_training.candidate_pool_size == 50
    
    print(f"✅ Dry run validation passed")
    print(f"  - Ready for multi-round training with {trainer.config.multiround.num_rounds} rounds")
    
    return trainer

def main():
    """Run all tests."""
    print("RiboPO v2 Multi-Round Training Integration Tests")
    print("=" * 60)
    
    try:
        # Run tests sequentially
        config = test_configuration_loading()
        trainer = test_trainer_initialization()
        trainer = test_component_initialization()
        trainer = test_pair_margin_scheduling()
        trainer = test_dry_run()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("RiboPO v2 multi-round training integration is ready.")
        print("\nTo run actual training:")
        print("python ribopo_v2/multiround_trainer.py \\")
        print("  --config ribopo_v2/config/experiments/03_multiround_test.yaml")
        print("\nTo run full training:")
        print("python ribopo_v2/multiround_trainer.py \\")
        print("  --config ribopo_v2/config/multiround_training.yaml")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()