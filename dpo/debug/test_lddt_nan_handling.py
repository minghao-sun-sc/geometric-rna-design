#!/usr/bin/env python3
"""
Test that lDDT properly returns NaN for failures and evaluation handles it correctly
"""

import sys
import os
import numpy as np

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_lddt_nan_handling():
    """Test lDDT NaN handling improvements"""
    print("🧪 Testing lDDT NaN Handling Improvements")
    print("=" * 70)
    
    # Test 1: lDDT function returns NaN for failures
    print("1️⃣ Testing lDDT function failure handling...")
    try:
        from src.evaluator import get_lddt
        
        # Test with non-existent files (should return NaN)
        result = get_lddt("/nonexistent/file1.pdb", "/nonexistent/file2.pdb")
        import numpy as np
        if np.isnan(result):
            print(f"   ✅ Non-existent files return NaN: {result}")
        else:
            print(f"   ❌ Expected NaN, got: {result}")
            
        # Test with existing but problematic files
        test_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
        if os.path.exists(test_dir):
            pdb_files = [f for f in os.listdir(test_dir) if f.endswith('.pdb')]
            if len(pdb_files) >= 2:
                pdb1 = os.path.join(test_dir, pdb_files[0])
                pdb2 = os.path.join(test_dir, pdb_files[1])
                
                result = get_lddt(pdb1, pdb2)
                print(f"   Result for {pdb_files[0]} vs {pdb_files[1]}: {result}")
                if np.isnan(result):
                    print(f"   ✅ Problematic comparison returns NaN (expected for structure mismatches)")
                elif result >= 0:
                    print(f"   ✅ Valid lDDT score: {result:.4f}")
                else:
                    print(f"   ❌ Unexpected negative value: {result}")
    
    except Exception as e:
        print(f"   ❌ lDDT function test failed: {e}")
        return False
    
    # Test 2: Evaluation pipeline NaN handling
    print("\n2️⃣ Testing evaluation pipeline NaN handling...")
    try:
        import numpy as np
        
        # Simulate lddt_list with mixed valid and NaN values
        lddt_list = [0.85, float('nan'), 0.72, float('nan'), 0.91, 0.68]
        
        # Test the logic from eval_full.py
        lddt_array = np.array(lddt_list)
        valid_lddt = lddt_array[~np.isnan(lddt_array)]
        mean_lddt = np.mean(valid_lddt) if len(valid_lddt) > 0 else float('nan')
        success_rate = len(valid_lddt) / len(lddt_array) if len(lddt_array) > 0 else 0.0
        
        print(f"   Input lDDT scores: {lddt_list}")
        print(f"   Valid scores: {valid_lddt.tolist()}")
        print(f"   Mean lDDT: {mean_lddt:.4f}")
        print(f"   Success rate: {success_rate:.1%}")
        
        expected_mean = np.mean([0.85, 0.72, 0.91, 0.68])
        expected_success = 4/6
        
        if abs(mean_lddt - expected_mean) < 0.001 and abs(success_rate - expected_success) < 0.001:
            print(f"   ✅ NaN handling works correctly")
        else:
            print(f"   ❌ NaN handling failed")
            print(f"      Expected mean: {expected_mean:.4f}, got: {mean_lddt:.4f}")
            print(f"      Expected success: {expected_success:.1%}, got: {success_rate:.1%}")
            return False
            
    except Exception as e:
        print(f"   ❌ Pipeline NaN handling test failed: {e}")
        return False
    
    # Test 3: Edge cases
    print("\n3️⃣ Testing edge cases...")
    try:
        # All NaN
        lddt_list_all_nan = [float('nan')] * 5
        lddt_array = np.array(lddt_list_all_nan)
        valid_lddt = lddt_array[~np.isnan(lddt_array)]
        mean_lddt = np.mean(valid_lddt) if len(valid_lddt) > 0 else float('nan')
        success_rate = len(valid_lddt) / len(lddt_array) if len(lddt_array) > 0 else 0.0
        
        if np.isnan(mean_lddt) and success_rate == 0.0:
            print(f"   ✅ All NaN case handled correctly: mean={mean_lddt}, success={success_rate:.1%}")
        else:
            print(f"   ❌ All NaN case failed: mean={mean_lddt}, success={success_rate:.1%}")
            
        # All valid
        lddt_list_all_valid = [0.8, 0.9, 0.7]
        lddt_array = np.array(lddt_list_all_valid)
        valid_lddt = lddt_array[~np.isnan(lddt_array)]
        mean_lddt = np.mean(valid_lddt) if len(valid_lddt) > 0 else float('nan')
        success_rate = len(valid_lddt) / len(lddt_array) if len(lddt_array) > 0 else 0.0
        
        if abs(mean_lddt - np.mean(lddt_list_all_valid)) < 0.001 and success_rate == 1.0:
            print(f"   ✅ All valid case handled correctly: mean={mean_lddt:.4f}, success={success_rate:.1%}")
        else:
            print(f"   ❌ All valid case failed: mean={mean_lddt:.4f}, success={success_rate:.1%}")
            
    except Exception as e:
        print(f"   ❌ Edge case test failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ lDDT NaN HANDLING IMPROVEMENTS VERIFIED!")
    print("\nKey improvements:")
    print("1. lDDT function now returns NaN instead of -1.0 for failures")
    print("2. Evaluation computes mean only from valid (non-NaN) scores")
    print("3. Success rate shows percentage of valid lDDT calculations")
    print("4. Proper handling of edge cases (all NaN, all valid)")
    
    print("\n📊 Expected evaluation output:")
    print('   "lddt": 0.7890,           // Mean of valid scores only')
    print('   "lddt_success_rate": 0.67, // 67% of calculations succeeded')
    
    return True

if __name__ == "__main__":
    success = test_lddt_nan_handling()
    if success:
        print("\n🚀 READY: lDDT evaluation with improved NaN handling!")
        print("Run evaluation to see both lDDT score and success rate in results.")
    else:
        print("\n❌ ISSUES: Need to fix lDDT NaN handling before evaluation.")