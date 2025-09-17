#!/usr/bin/env python3
"""
Test script to verify that MultiRoundDPOTrainer can be imported and initialized
without the SimpleNamespace errors or deepcopy issues.
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

def test_multiround_init():
    """Test that MultiRoundDPOTrainer can be initialized."""
    print("🧪 Testing MultiRoundDPOTrainer initialization...")
    
    try:
        # Load config
        config_path = "multiround/config/experiments/plan_a_margin125.yaml"
        print(f"Loading config: {config_path}")
        cfg = load_cfg(config_path)
        
        # Test that we can import the trainer
        from multiround.trainer import MultiRoundDPOTrainer
        print("✅ MultiRoundDPOTrainer imported successfully")
        
        # Test basic initialization (without actually creating models)
        print("🔧 Testing trainer initialization...")
        
        # Override wandb to disabled mode for testing
        cfg.wandb.enable = False
        
        trainer = MultiRoundDPOTrainer(cfg)
        print("✅ MultiRoundDPOTrainer initialized successfully")
        
        # Test some basic attributes
        print(f"   Rounds: {trainer.num_rounds}")
        print(f"   Epochs per round: {trainer.epochs_per_round}")
        print(f"   Current round: {trainer.current_round}")
        print(f"   Output root: {trainer.output_root}")
        
        print("\n🎉 All tests passed! MultiRoundDPOTrainer is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multiround_init()
    sys.exit(0 if success else 1)