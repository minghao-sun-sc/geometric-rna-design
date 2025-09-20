#!/usr/bin/env python3
"""
Test configuration loading after fixes
"""

import sys
import os
sys.path.append('/mnt/rna01/smh/projects/ribopo')

from multiround.train import load_multiround_config

def test_config_loading():
    """Test that checkpoints section is now available."""
    config_path = "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/sft_ablation_v1.yaml"
    
    print("🔧 Testing configuration loading after fixes...")
    
    try:
        cfg = load_multiround_config(config_path)
        
        # Test checkpoints access
        print(f"📋 Testing checkpoints access...")
        checkpoints_cfg = getattr(cfg, 'checkpoints', None)
        if checkpoints_cfg is not None:
            metric_name = getattr(checkpoints_cfg, 'metric_for_best', 'tm_mean')
            print(f"✅ Successfully accessed checkpoints.metric_for_best: {metric_name}")
            
            # Test other checkpoint settings
            save_freq = getattr(checkpoints_cfg, 'save_frequency', 'per_round')
            keep_best = getattr(checkpoints_cfg, 'keep_best_n', 3)
            print(f"✅ save_frequency: {save_freq}")
            print(f"✅ keep_best_n: {keep_best}")
        else:
            print("❌ checkpoints section not found")
            return False
            
        # Test other config sections
        print(f"✅ wandb.enable: {cfg.wandb.enable}")
        print(f"✅ multiround.num_rounds: {cfg.multiround.num_rounds}")
        print(f"✅ dpo.beta: {cfg.dpo.beta}")
        
        print("\n🎉 Configuration loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_loading()