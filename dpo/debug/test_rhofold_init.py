#!/usr/bin/env python3
"""
Test RhoFold initialization to see what's failing
"""
import sys
import os
import torch

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_rhofold_init():
    print("🔍 Testing RhoFold Initialization")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        print("Step 1: Importing RhoFold...")
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        print("✅ RhoFold imports successful")
        
        print("Step 2: Creating RhoFold instance...")
        rhofold = RhoFold(rhofold_config, device)
        print("✅ RhoFold instance created")
        
        print("Step 3: Loading model checkpoint...")
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        print(f"Loading: {rhofold_path}")
        
        if not os.path.exists(rhofold_path):
            print(f"❌ Checkpoint not found: {rhofold_path}")
            return False
        
        checkpoint = torch.load(rhofold_path, map_location=torch.device('cpu'))
        print(f"✅ Checkpoint loaded, keys: {list(checkpoint.keys())}")
        
        rhofold.load_state_dict(checkpoint['model'])
        print("✅ Model state dict loaded")
        
        print("Step 4: Moving to device...")
        rhofold = rhofold.to(device)
        rhofold.eval()
        print("✅ Model moved to device and set to eval mode")
        
        print("✅ RhoFold initialization SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ RhoFold initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_src_evaluator_rhofold_section():
    """Test the exact code path from src.evaluator.evaluate()"""
    print("\n🔍 Testing src.evaluator RhoFold Section")
    print("=" * 50)
    
    # Simulate the exact conditions in src.evaluator.evaluate()
    metrics = ['recovery', 'perplexity', 'sc_rhofold']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Metrics: {metrics}")
    print(f"'sc_score_rhofold' in metrics: {'sc_score_rhofold' in metrics}")
    print(f"'sc_rhofold' in metrics: {'sc_rhofold' in metrics}")
    
    # This is the issue! The config uses 'sc_rhofold' but src.evaluator checks for 'sc_score_rhofold'
    if 'sc_score_rhofold' in metrics:
        print("✅ Would initialize RhoFold")
        return test_rhofold_init()
    else:
        print("❌ RhoFold section SKIPPED - metric name mismatch!")
        print("Config has 'sc_rhofold', but evaluator checks for 'sc_score_rhofold'")
        return False

def main():
    success1 = test_rhofold_init()
    success2 = test_src_evaluator_rhofold_section()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS")
    print("=" * 50)
    
    if not success2:
        print("❌ METRIC NAME MISMATCH DETECTED!")
        print("Your config uses: 'sc_rhofold'")
        print("src.evaluator expects: 'sc_score_rhofold'")
        print("\n🔧 SOLUTION: Change config metrics from:")
        print("  - sc_rhofold")
        print("to:")
        print("  - sc_score_rhofold")
    elif success1:
        print("✅ RhoFold works correctly")
        print("❌ Issue is elsewhere in evaluation pipeline")
    else:
        print("❌ RhoFold initialization failed")

if __name__ == "__main__":
    main()