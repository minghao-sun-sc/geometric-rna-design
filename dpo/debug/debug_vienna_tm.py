#!/usr/bin/env python3
"""
Debug why Vienna melting temperature (Tm) is not appearing in eval results
"""

import sys
import os
import json

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def check_vienna_tm_in_config():
    """Check what Vienna metrics are enabled in the config"""
    print("🔬 Debugging Vienna Melting Temperature (Tm) Issue")
    print("=" * 70)
    
    config_path = "/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    metrics = config.get('eval', {}).get('metrics', [])
    print(f"📋 Metrics in config: {metrics}")
    
    vienna_metrics = [m for m in metrics if 'vienna' in m.lower()]
    print(f"🧬 Vienna metrics found: {vienna_metrics}")
    
    if 'sc_score_vienna' in metrics:
        print("✅ sc_score_vienna found - this would enable Tm calculation in src/evaluator.py")
    else:
        print("❌ sc_score_vienna NOT found - Vienna Tm will not be calculated")
        print("   Individual Vienna metrics found:", vienna_metrics)
        print("   These are handled separately in eval_full.py")

def check_vienna_tm_in_evaluator():
    """Test Vienna Tm calculation from src/evaluator.py"""
    print("\n🧪 Testing Vienna Tm calculation...")
    
    try:
        from src.evaluator import vienna_Tm_by_pS0, vienna_mfe
        
        # Test sequence and structure
        test_seq = "GGCAAGCCUGCGAUGGCCAAGCCUGCG"
        
        # Get MFE structure as target
        mfe, target_db = vienna_mfe(test_seq, 37.0)
        print(f"   Test sequence: {test_seq}")
        print(f"   MFE structure: {target_db}")
        print(f"   MFE energy: {mfe:.2f} kcal/mol")
        
        # Calculate Tm
        tm = vienna_Tm_by_pS0(test_seq, target_db, Tmin=10, Tmax=95, step=2.0, threshold=0.5)
        print(f"   Melting Temperature: {tm:.1f} °C")
        
        if tm > 0:
            print("✅ Vienna Tm calculation works correctly!")
        else:
            print("⚠️ Vienna Tm calculation returned invalid result")
            
    except Exception as e:
        print(f"❌ Vienna Tm calculation failed: {e}")
        import traceback
        traceback.print_exc()

def check_recent_eval_results():
    """Check recent evaluation results for Vienna Tm"""
    print("\n📊 Checking Recent Evaluation Results...")
    
    eval_dir = "/mnt/rna01/smh/projects/ribopo/dpo/eval_results"
    
    if not os.path.exists(eval_dir):
        print(f"❌ Eval results directory not found: {eval_dir}")
        return
    
    json_files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
    if not json_files:
        print("❌ No JSON evaluation results found")
        return
    
    # Check the most recent file
    latest_file = sorted(json_files)[-1]
    json_path = os.path.join(eval_dir, latest_file)
    
    print(f"   Checking: {latest_file}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check summary for Vienna metrics
    summary = data.get('summary', {})
    vienna_keys = [k for k in summary.keys() if 'vienna' in k.lower()]
    
    print(f"   Vienna metrics in summary: {vienna_keys}")
    
    # Check per-checkpoint results
    per_ckpt = data.get('per_checkpoint_results', [])
    if per_ckpt:
        first_result = per_ckpt[0]
        vienna_metrics_in_results = [k for k in first_result.keys() if 'vienna' in k.lower()]
        print(f"   Vienna metrics in checkpoint results: {vienna_metrics_in_results}")
        
        if 'vienna_Tm' in first_result:
            tm_val = first_result['vienna_Tm']
            print(f"✅ Found vienna_Tm: {tm_val}")
        else:
            print("❌ vienna_Tm not found in results")
    
    print(f"\n💡 DIAGNOSIS:")
    if 'vienna_Tm' not in (per_ckpt[0] if per_ckpt else {}):
        print("   Problem: vienna_Tm is not being calculated in eval_full.py")
        print("   Solution: Need to add Vienna Tm calculation to eval_full.py")
        print("   The code in src/evaluator.py works but only runs with 'sc_score_vienna' metric")

def show_solution():
    """Show the solution to fix Vienna Tm"""
    print("\n" + "=" * 70)
    print("🛠️ SOLUTION:")
    print("   1. The config uses separate Vienna metrics: 'vienna_mfe', 'vienna_ED'")
    print("   2. eval_full.py handles these individually but doesn't calculate Tm")
    print("   3. Need to add Vienna Tm calculation to the Vienna metrics section in eval_full.py")
    print("   4. The calculation function vienna_Tm_by_pS0() already exists and works")
    print("\n   RECOMMENDED FIX:")
    print("   - Add vienna_Tm calculation alongside vienna_mfe and vienna_ED in eval_full.py")
    print("   - Store results in vienna_tm_list and include in final results dict")

if __name__ == "__main__":
    check_vienna_tm_in_config()
    check_vienna_tm_in_evaluator()
    check_recent_eval_results()
    show_solution()