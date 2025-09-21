#!/usr/bin/env python3
"""
Test debug configuration files to ensure they're correct and ready for multiround training.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def test_config(config_path):
    """Test a single configuration file."""
    print(f"\n🧪 Testing {os.path.basename(config_path)}")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print(f"✅ Config loaded successfully")
        
        # Check critical multiround settings
        print(f"📊 Multiround Settings:")
        print(f"   Rounds: {cfg.multiround.num_rounds}")
        print(f"   Epochs per round: {cfg.multiround.epochs_per_round}")
        print(f"   Update reference: {cfg.multiround.update_reference}")
        print(f"   Dynamic pairs: {cfg.multiround.dynamic_pairs}")
        
        # Check GPU optimization settings
        print(f"⚡ GPU Optimization:")
        print(f"   Training batch size: {cfg.training.batch_size}")
        print(f"   Eval batch size: {cfg.eval.batch_size}")
        print(f"   Precision: {cfg.training.precision}")
        print(f"   Grad accum steps: {cfg.training.grad_accum_steps}")
        effective_batch = cfg.training.batch_size * cfg.training.grad_accum_steps
        print(f"   Effective batch size: {effective_batch}")
        
        # Check paths exist
        print(f"📁 Path Validation:")
        required_paths = [
            ('Base checkpoint', cfg.paths.base_checkpoint),
            ('Processed data', cfg.paths.processed_pt),
            ('Split data', cfg.paths.split_pt),
            ('Train pairs', cfg.paths.pairs.train),
            ('Val pairs', cfg.paths.pairs.val),
            ('Test pairs', cfg.paths.pairs.test)
        ]
        
        all_paths_exist = True
        for name, path in required_paths:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"   {status} {name}: {path}")
            if not exists:
                all_paths_exist = False
        
        # Check pair provider setup
        print(f"🔄 Pair Provider Test:")
        from multiround.pair_provider import MultiRoundPairProvider
        pair_provider = MultiRoundPairProvider(cfg)
        
        for round_num in [1, 2]:
            try:
                pair_config = pair_provider.get_config_for_round(round_num)
                print(f"   Round {round_num}: {pair_config.margin_type}*std pairs")
            except Exception as e:
                print(f"   ❌ Round {round_num}: {e}")
                all_paths_exist = False
        
        # Check evaluator setup
        print(f"📊 Evaluator Test:")
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        print(f"   Dataset: {len(evaluator.eval_dataset.data_list)} structures")
        print(f"   Pass@k rounds: {evaluator.passk_rounds}")
        
        # Check WandB config
        print(f"📝 WandB Configuration:")
        print(f"   Project: {cfg.wandb.project}")
        print(f"   Run name: {cfg.wandb.run_name}")
        print(f"   Tags: {cfg.wandb.tags}")
        
        # Check DPO parameters
        print(f"🎯 DPO Parameters:")
        print(f"   Beta: {cfg.dpo.beta}")
        print(f"   SFT lambda: {cfg.dpo.sft_lambda}")
        print(f"   Learning rate: {cfg.optimizer.lr}")
        
        # Summary
        if all_paths_exist:
            print(f"\n🎉 {os.path.basename(config_path)} - ALL CHECKS PASSED!")
            return True
        else:
            print(f"\n⚠️ {os.path.basename(config_path)} - Some paths missing!")
            return False
            
    except Exception as e:
        print(f"❌ {os.path.basename(config_path)} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test both debug configurations."""
    print("🔧 Debug Configuration Validation")
    print("Testing both A100 and A40 debug configurations.")
    
    configs_to_test = [
        "multiround/config/experiments/00_debug_a100_80g_verified.yaml",
        "multiround/config/experiments/00_debug_a40_verified.yaml"
    ]
    
    results = {}
    for config_path in configs_to_test:
        if os.path.exists(config_path):
            results[config_path] = test_config(config_path)
        else:
            print(f"\n❌ {config_path} - FILE NOT FOUND")
            results[config_path] = False
    
    # Final summary
    print(f"\n🏁 FINAL SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for config_path, passed in results.items():
        config_name = os.path.basename(config_path)
        status = "✅ READY" if passed else "❌ NEEDS FIX"
        print(f"{status} {config_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🚀 ALL DEBUG CONFIGS READY FOR TRAINING!")
        print(f"   Both A100 and A40 configurations validated.")
        print(f"   Ready to run multiround training trials.")
        print(f"\n💡 Commands to run:")
        print(f"   # A100 80GB:")
        print(f"   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        print(f"   python -m multiround.train --config multiround/config/experiments/00_debug_a100_80g_verified.yaml")
        print(f"")
        print(f"   # A40 48GB:")
        print(f"   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
        print(f"   python -m multiround.train --config multiround/config/experiments/00_debug_a40_verified.yaml")
    else:
        print(f"\n⚠️ SOME DEBUG CONFIGS NEED FIXES")
        print(f"   Please address the issues above before training.")

if __name__ == "__main__":
    main()