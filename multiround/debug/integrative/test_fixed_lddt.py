#!/usr/bin/env python3
"""
Test the fixed lDDT function with actual evaluation files.
"""

import os
import sys
sys.path.insert(0, '/mnt/rna01/smh/projects/ribopo')
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from src.evaluator import get_lddt_openstructure_v2

def test_fixed_lddt():
    """Test the fixed lDDT function with actual evaluation files."""
    
    predicted_pdb = "/mnt/rna01/smh/projects/ribopo/multiround/eval_multiround/01_base_dpo_t05/designs_BASE/20250919_153427/sample0/design0_unrelaxed.pdb"
    native_pdb = "/mnt/rna01/smh/projects/ribopo/data/raw/3B58_1_B-C-A.pdb"
    
    print("=== Testing Fixed lDDT Function ===")
    print(f"Predicted: {predicted_pdb}")
    print(f"Native:    {native_pdb}")
    print(f"Files exist: {os.path.exists(predicted_pdb)} & {os.path.exists(native_pdb)}")
    
    if not os.path.exists(predicted_pdb) or not os.path.exists(native_pdb):
        print("❌ Required files missing!")
        return
    
    print("\n--- Running Fixed lDDT Calculation ---")
    try:
        lddt_score = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
        
        print(f"lDDT Result: {lddt_score}")
        print(f"Type: {type(lddt_score)}")
        
        if lddt_score != lddt_score:  # Check for NaN
            print("❌ Still getting NaN - fix didn't work")
            print("This indicates the chain mapping approach needs refinement")
        elif lddt_score == float('inf') or lddt_score == float('-inf'):
            print("❌ Got infinite value - unexpected result")
        elif 0.0 <= lddt_score <= 1.0:
            print(f"✅ SUCCESS! Got valid lDDT score: {lddt_score:.4f}")
            print("The chain mapping fix worked!")
        else:
            print(f"⚠️  Got unusual score: {lddt_score} (outside 0-1 range)")
            print("This might indicate a calculation issue")
            
    except Exception as e:
        print(f"❌ Exception in fixed lDDT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_lddt()