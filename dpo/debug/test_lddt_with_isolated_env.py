#!/usr/bin/env python3
"""
Test lDDT calculation using the updated evaluator functions with isolated env
"""

import sys
import os

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def test_lddt_isolated():
    print("🔬 Testing lDDT with Isolated Environment")
    print("=" * 50)
    
    # Import the updated evaluator functions
    from src.evaluator import get_lddt, get_lddt_inverse_folding
    
    # Test files from debug examples
    model_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/model.pdb')
    native_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/native.pdb')
    
    if not os.path.exists(model_pdb) or not os.path.exists(native_pdb):
        print(f"❌ Test PDB files not found")
        print(f"   Model: {model_pdb}")
        print(f"   Native: {native_pdb}")
        return False
    
    print(f"✅ Test files found")
    print(f"   Model: {model_pdb}")
    print(f"   Native: {native_pdb}")
    
    # Test the main get_lddt function
    print(f"\n🧪 Testing get_lddt function...")
    lddt_score = get_lddt(model_pdb, native_pdb)
    
    print(f"\n📊 Results:")
    if lddt_score > 0:
        print(f"✅ lDDT score: {lddt_score:.4f}")
        print(f"✅ lDDT calculation with isolated environment SUCCESSFUL!")
        return True
    else:
        print(f"❌ lDDT failed (score: {lddt_score})")
        return False

if __name__ == "__main__":
    success = test_lddt_isolated()
    sys.exit(0 if success else 1)
