#!/usr/bin/env python3
"""
Debug script to test multiround configuration loading and validation.
"""

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import os
import sys
sys.path.append('/mnt/rna01/smh/projects/ribopo')

from multiround.train import load_multiround_config, validate_config
from multiround.pair_provider import MultiRoundPairProvider

def test_config_loading():
    """Test configuration loading and validation."""
    config_path = "multiround/config/experiments/multiround_dynamic_margins_v1.yaml"
    
    print("🔧 Testing multiround configuration...")
    print(f"Config path: {config_path}")
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        cfg = load_multiround_config(config_path)
        print("✅ Configuration loaded successfully")
        
        # Validate configuration
        print("\n2. Validating configuration...")
        is_valid = validate_config(cfg)
        if is_valid:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        # Test pair provider initialization
        print("\n3. Testing pair provider...")
        pair_provider = MultiRoundPairProvider(cfg)
        print("✅ Pair provider initialized successfully")
        
        # Test dynamic pair switching for round 1 and 3
        print("\n4. Testing dynamic pair switching...")
        for round_num in [1, 3]:
            print(f"\n   Round {round_num}:")
            paths = pair_provider.update_config_for_round(round_num)
            margin_type = pair_provider.get_current_margin_type()
            print(f"   Margin type: {margin_type}")
            print(f"   Train path: {paths['train']}")
            
            # Verify files exist
            if os.path.exists(paths['train']):
                print(f"   ✅ Training file exists")
            else:
                print(f"   ❌ Training file missing: {paths['train']}")
                
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)