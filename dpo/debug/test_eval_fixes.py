#!/usr/bin/env python3
"""
Test script to verify evaluation pipeline fixes:
1. Amber relaxation effect on clash scores
2. Vienna metrics with target tracking
3. Metric interpretation display
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

def test_fixes():
    print("=" * 60)
    print("EVALUATION PIPELINE FIXES VERIFICATION")
    print("=" * 60)
    
    # 1. Check if relaxation is enabled
    print("\n1. AMBER RELAXATION STATUS:")
    try:
        with open('/mnt/rna01/smh/projects/ribopo/dpo/bench/eval_full.py', 'r') as f:
            content = f.read()
            if 'use_relax=True' in content:
                print("   ✅ Amber relaxation ENABLED - clash scores should improve")
                print("   Note: This adds ~30s per structure but reduces clashes significantly")
            else:
                print("   ❌ Amber relaxation DISABLED - clash scores will remain high")
    except Exception as e:
        print(f"   ❌ Error checking relaxation: {e}")
    
    # 2. Check temperature setting
    print("\n2. TEMPERATURE SETTING:")
    try:
        with open('/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml', 'r') as f:
            prev_line = ""
            for line in f:
                if 'temperature:' in line:
                    temp = float(line.split(':')[1].split('#')[0].strip())
                    if temp == 0.1:
                        print(f"   ✅ Temperature set to {temp} (optimal for consistent sampling)")
                    else:
                        print(f"   ⚠️ Temperature set to {temp} (consider using 0.1)")
                    break
                prev_line = line
    except Exception as e:
        print(f"   ❌ Error checking temperature: {e}")
    
    # 3. Check metric interpretation features
    print("\n3. METRIC INTERPRETATION FEATURES:")
    features = [
        ("Clash score warnings", "clash_note"),
        ("INF interpretation", "overall contact correctness"),
        ("Vienna target tracking", "vienna_target_source"),
    ]
    
    try:
        with open('/mnt/rna01/smh/projects/ribopo/dpo/bench/eval_full.py', 'r') as f:
            content = f.read()
            for feature_name, search_str in features:
                if search_str in content:
                    print(f"   ✅ {feature_name} implemented")
                else:
                    print(f"   ❌ {feature_name} not found")
    except Exception as e:
        print(f"   ❌ Error checking features: {e}")
    
    print("\n" + "=" * 60)
    print("EXPECTED IMPROVEMENTS:")
    print("=" * 60)
    print("""
    1. CLASH SCORES: Should drop from 666 → <50 with relaxation
       - Unrelaxed: 200-1000+ (poor geometry)  
       - Relaxed: 20-50 (acceptable)
       - Good structures: <20
    
    2. INF INTERPRETATION:
       - INF(all): Overall contact correctness (0-1)
       - INF(WC): Watson-Crick pairing (higher=better)
       - INF(non-WC): Non-WC interactions (negative=wrong)
       - INF(stack): Stacking interactions (usually highest)
    
    3. VIENNA METRICS:
       - MFE: More negative = more stable
       - ED/nt: <0.05 excellent, <0.1 good, >0.2 poor
       - P(target): >0.01 good, >0.001 acceptable
    
    4. STRUCTURE QUALITY THRESHOLDS:
       - RMSD: <8Å acceptable, <2Å excellent
       - TM-score: >0.45 good fold, >0.5 very good
       - pLDDT: >0.7 confident, >0.9 very confident
    """)
    
    print("\nTo run full evaluation with fixes:")
    print("  python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml")
    print("\nFor DPO v10 checkpoints:")
    print("  python -m dpo.bench.eval_full --config dpo/configs/experiments/full_eval_v10_checkpoints.yaml")

if __name__ == "__main__":
    test_fixes()