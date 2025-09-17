#!/usr/bin/env python3
"""Test that diversity_3mer is properly enabled"""

import sys
import os
import yaml

def test_diversity_fix():
    print("=" * 60)
    print("DIVERSITY METRIC FIX VERIFICATION")
    print("=" * 60)
    
    # 1. Check bench_full.yaml has diversity_3mer
    print("\n1. CONFIG FILE CHECK (bench_full.yaml):")
    with open('/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        metrics = cfg.get('eval', {}).get('metrics', [])
        if 'diversity_3mer' in metrics:
            print(f"   ✅ diversity_3mer IS in config metrics: {metrics}")
        else:
            print(f"   ❌ diversity_3mer NOT in config metrics: {metrics}")
    
    # 2. Check eval_full.py doesn't override with defaults
    print("\n2. COMMAND LINE DEFAULT CHECK:")
    with open('/mnt/rna01/smh/projects/ribopo/dpo/bench/eval_full.py', 'r') as f:
        content = f.read()
        if 'default=None,  # Use config file metrics' in content:
            print("   ✅ Fixed: Command line won't override config metrics")
        else:
            print("   ❌ Issue: Command line may override config metrics")
            
        if 'if args.metrics is None:' in content and 'args.metrics = cfg.eval.metrics' in content:
            print("   ✅ Fixed: Uses config metrics when not specified on command line")
        else:
            print("   ❌ Issue: May not use config metrics properly")
    
    # 3. Check implementation exists
    print("\n3. IMPLEMENTATION CHECK:")
    checks = [
        ("diversity_3mer calculation", "if 'diversity_3mer' in metrics:"),
        ("get_three_mer_corr import", "from src.evaluator import get_three_mer_corr"),
        ("diversity storage", "diversity_3mer_list = []"),
        ("diversity aggregation", "results[\"diversity_3mer\"]"),
        ("diversity display", "3-mer diversity:")
    ]
    
    with open('/mnt/rna01/smh/projects/ribopo/dpo/bench/eval_full.py', 'r') as f:
        content = f.read()
        for check_name, search_str in checks:
            if search_str in content:
                print(f"   ✅ {check_name} found")
            else:
                print(f"   ❌ {check_name} NOT found")
    
    print("\n" + "=" * 60)
    print("ISSUE DIAGNOSIS:")
    print("=" * 60)
    print("""
    The problem was that --metrics command line argument had a DEFAULT value
    that overrode the config file. When you ran:
    
      python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml
    
    It used the hardcoded default ['recovery', 'perplexity', 'sc_eternafold', 'sc_rhofold']
    instead of reading metrics from the config file.
    
    NOW FIXED:
    - Command line --metrics defaults to None
    - If None, uses config file metrics
    - Config includes diversity_3mer
    
    To run with diversity metric:
      python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml
    
    To override (without diversity):
      python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml --metrics recovery perplexity sc_rhofold
    """)

if __name__ == "__main__":
    test_diversity_fix()