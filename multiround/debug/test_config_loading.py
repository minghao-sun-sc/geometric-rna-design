# multiround/debug/test_config_loading.py
"""
Test script for multiround configuration loading and validation.
"""
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from multiround.train import load_multiround_config, validate_config, print_config_summary


def test_config_inheritance():
    """Test configuration inheritance functionality."""
    print("🧪 Testing configuration inheritance...")
    
    # Test loading margin125 config
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if os.path.exists(config_path):
        try:
            cfg = load_multiround_config(config_path)
            print("✅ Successfully loaded plan_a_margin125.yaml")
            print(f"   Experiment name: {getattr(cfg.experiment, 'name', 'N/A')}")
            print(f"   Pair margin: {getattr(cfg.paths, 'pair_margin', 'N/A')}")
            print(f"   DPO beta: {getattr(cfg.dpo, 'beta', 'N/A')}")
            print(f"   Num rounds: {getattr(cfg.multiround, 'num_rounds', 'N/A')}")
        except Exception as e:
            print(f"❌ Failed to load plan_a_margin125.yaml: {e}")
    else:
        print(f"❌ Config file not found: {config_path}")
    
    # Test loading margin25 config
    config_path = "multiround/config/experiments/plan_a_margin25.yaml"
    if os.path.exists(config_path):
        try:
            cfg = load_multiround_config(config_path)
            print("✅ Successfully loaded plan_a_margin25.yaml")
            print(f"   Experiment name: {getattr(cfg.experiment, 'name', 'N/A')}")
            print(f"   Pair margin: {getattr(cfg.paths, 'pair_margin', 'N/A')}")
            print(f"   DPO beta: {getattr(cfg.dpo, 'beta', 'N/A')}")
        except Exception as e:
            print(f"❌ Failed to load plan_a_margin25.yaml: {e}")
    else:
        print(f"❌ Config file not found: {config_path}")


def test_config_validation():
    """Test configuration validation."""
    print("\n🧪 Testing configuration validation...")
    
    config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if os.path.exists(config_path):
        try:
            cfg = load_multiround_config(config_path)
            is_valid = validate_config(cfg)
            
            if is_valid:
                print("✅ Configuration validation passed")
                print_config_summary(cfg)
            else:
                print("❌ Configuration validation failed")
                
        except Exception as e:
            print(f"❌ Validation test failed: {e}")
    else:
        print(f"❌ Config file not found for validation: {config_path}")


def test_pair_file_existence():
    """Test that preference pair files exist and are accessible."""
    print("\n🧪 Testing preference pair file accessibility...")
    
    # Test both margin datasets
    margins = ["125", "25"]
    splits = ["train", "val", "test"]
    
    for margin in margins:
        print(f"\n📁 Testing margin{margin} pairs:")
        for split in splits:
            pair_path = f"data/pairs_margin{margin}/by_das/clean/{split}.clean.jsonl"
            if os.path.exists(pair_path):
                try:
                    # Try to read first few lines
                    with open(pair_path, 'r') as f:
                        lines = [f.readline().strip() for _ in range(3)]
                        valid_lines = [l for l in lines if l]
                        
                    # Try to parse as JSON
                    import json
                    for line in valid_lines:
                        json.loads(line)  # Should not raise exception
                    
                    print(f"   ✅ {split}.clean.jsonl - accessible and valid JSON")
                    
                except Exception as e:
                    print(f"   ❌ {split}.clean.jsonl - error reading: {e}")
            else:
                print(f"   ❌ {split}.clean.jsonl - file not found")


def test_checkpoint_and_data_files():
    """Test that required data files exist."""
    print("\n🧪 Testing required data files...")
    
    required_files = {
        "processed.pt": "data/processed.pt",
        "das_split.pt": "data/das_split.pt", 
        "base_checkpoint": "checkpoints/gRNAde_ARv1_1state_das.h5"
    }
    
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"   ✅ {name} - found at {path}")
            
            # Check file size for basic sanity
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"      Size: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {name} - not found at {path}")


def main():
    """Run all configuration tests."""
    print("🚀 Starting multiround configuration tests...\n")
    
    test_config_inheritance()
    test_config_validation()
    test_pair_file_existence()
    test_checkpoint_and_data_files()
    
    print("\n✅ Configuration testing completed!")


if __name__ == "__main__":
    main()