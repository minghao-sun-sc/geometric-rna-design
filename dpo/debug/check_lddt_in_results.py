#!/usr/bin/env python3
"""
Quick check to see if lDDT is in evaluation results
"""

import json
import os
import numpy as np

def check_eval_results_for_lddt():
    print("📊 Checking Evaluation Results for lDDT Scores")
    print("=" * 60)
    
    eval_dir = "/mnt/rna01/smh/projects/ribopo/dpo/eval_results"
    
    # Find the specific file mentioned
    target_file = "eval_test_eval_dpo_v5_best_temp0.1_20250918_192931.json"
    json_path = os.path.join(eval_dir, target_file)
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {target_file}")
        # Try to find any eval results
        if os.path.exists(eval_dir):
            files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
            if files:
                print(f"\n📁 Available files:")
                for f in sorted(files)[-5:]:  # Show last 5
                    print(f"   - {f}")
                json_path = os.path.join(eval_dir, files[-1])
                print(f"\n🔍 Checking most recent: {files[-1]}")
            else:
                print("❌ No JSON files found in eval_results/")
                return
        else:
            print(f"❌ Directory not found: {eval_dir}")
            return
    else:
        print(f"✅ Found target file: {target_file}")
    
    # Load and analyze
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n📋 File Contents:")
    print("-" * 40)
    
    # Check metadata
    if 'metadata' in data:
        meta = data['metadata']
        print(f"   Temperature: {meta.get('temperature', 'N/A')}")
        print(f"   N samples: {meta.get('n_samples_per_structure', 'N/A')}")
        print(f"   Metrics computed: {meta.get('metrics_computed', [])}")
    
    # Check summary for lDDT
    print("\n🔍 Checking for lDDT in summary:")
    if 'summary' in data:
        summary = data['summary']
        if 'lddt' in summary:
            lddt_stats = summary['lddt']
            print(f"   ✅ lDDT found in summary!")
            print(f"      Mean: {lddt_stats.get('mean', 'N/A')}")
            print(f"      Std:  {lddt_stats.get('std', 'N/A')}")
            print(f"      Min:  {lddt_stats.get('min', 'N/A')}")
            print(f"      Max:  {lddt_stats.get('max', 'N/A')}")
        else:
            print(f"   ❌ lDDT NOT found in summary")
            print(f"   Available summary keys: {list(summary.keys())[:10]}")
    
    # Check per-checkpoint results
    print("\n🔍 Checking per-checkpoint results:")
    if 'per_checkpoint_results' in data:
        results = data['per_checkpoint_results']
        if results and len(results) > 0:
            first_result = results[0]
            if 'lddt' in first_result:
                lddt_val = first_result['lddt']
                if isinstance(lddt_val, (int, float)):
                    if np.isnan(lddt_val):
                        print(f"   ⚠️ lDDT present but is NaN")
                    else:
                        print(f"   ✅ lDDT score: {lddt_val:.4f}")
                else:
                    print(f"   ⚠️ lDDT present but unusual type: {type(lddt_val)}")
            else:
                print(f"   ❌ lDDT NOT in checkpoint results")
                
            # Show all available metrics
            print(f"\n   All metrics in results:")
            metrics = [k for k in first_result.keys() if not k.startswith('ckpt')]
            for i in range(0, len(metrics), 5):
                print(f"      {', '.join(metrics[i:i+5])}")
    
    print("\n" + "=" * 60)
    print("💡 DIAGNOSIS:")
    
    # Check if the config might have had use_lddt disabled
    config_path = "/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        use_lddt = config.get('eval', {}).get('use_lddt', False)
        
        if use_lddt:
            print(f"   ✅ Config has use_lddt=True (lDDT should be calculated)")
            if 'lddt' not in (data.get('per_checkpoint_results', [{}])[0] if data.get('per_checkpoint_results') else {}):
                print(f"   ⚠️ But lDDT is missing from results!")
                print(f"   Possible reasons:")
                print(f"      1. Old evaluation before lDDT was fixed")
                print(f"      2. All lDDT calculations returned NaN")
                print(f"      3. RhoFold failures prevented structure generation")
        else:
            print(f"   ❌ Config has use_lddt=False (lDDT won't be calculated)")
            print(f"   Fix: Set 'use_lddt: true' in eval section of config")

if __name__ == "__main__":
    check_eval_results_for_lddt()