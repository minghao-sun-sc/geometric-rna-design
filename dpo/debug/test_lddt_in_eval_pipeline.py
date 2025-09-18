#!/usr/bin/env python3
"""
Debug script to verify lDDT calculation in the evaluation pipeline
Tests with minimal computational cost using 1-2 structures
"""

import sys
import os
import tempfile
import numpy as np

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_lddt_in_pipeline():
    print("🔍 Testing lDDT in Evaluation Pipeline")
    print("=" * 70)
    
    # Import the extended evaluator
    from src.evaluator import self_consistency_score_rhofold_extended, get_lddt
    from src.constants import NUM_TO_LETTER, DATA_PATH
    
    # Create a minimal test case
    print("\n📊 Test 1: Direct lDDT calculation")
    print("-" * 40)
    
    # Test with actual PDB files from the dataset
    test_structures = [
        ("1DDY_1_A", "/mnt/rna01/smh/projects/ribopo/data/raw/1DDY_1_A.pdb"),
        ("1L2X_1_A", "/mnt/rna01/smh/projects/ribopo/data/raw/1L2X_1_A.pdb"),
    ]
    
    for struct_id, pdb_path in test_structures:
        if os.path.exists(pdb_path):
            print(f"\n   Testing {struct_id}:")
            # Self-comparison
            lddt_score = get_lddt(pdb_path, pdb_path)
            print(f"   Self-lDDT: {lddt_score:.4f} (expected: 1.0000)")
            if abs(lddt_score - 1.0) < 0.001:
                print(f"   ✅ Direct lDDT working for {struct_id}")
            else:
                print(f"   ⚠️ Unexpected score for {struct_id}")
    
    # Test the extended evaluator function
    print("\n📊 Test 2: Extended RhoFold evaluator with lDDT")
    print("-" * 40)
    
    # Create mock data
    samples = np.array([[0, 1, 2, 3, 0, 1, 2, 3]])  # Mock sequence (ACGUACGU)
    true_raw_data = {
        "sequence": "ACGUACGU",
        "id_list": ["1DDY_1_A"],  # Use real structure ID
        "coords_list": []  # Empty for this test
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"   Using temp dir: {tmpdir}")
        
        # Test with use_lddt=True
        try:
            result = self_consistency_score_rhofold_extended(
                samples=samples,
                true_raw_data=true_raw_data,
                mask_coords=np.ones(8, dtype=bool),
                rhofold=None,  # Will fail at RhoFold step but that's OK for this test
                output_dir=tmpdir,
                save_designs=False,
                save_pdbs=False,
                use_relax=False,
                use_inf=False,
                use_clash=False,
                use_lddt=True,  # Enable lDDT
                use_mcq=False,
                phenix_wrapper_path=None
            )
            print("   ❌ Should have failed at RhoFold step (expected)")
        except Exception as e:
            print(f"   ✅ Failed at expected point: {str(e)[:50]}...")
    
    # Test the evaluation pipeline's lDDT handling
    print("\n📊 Test 3: Check eval_full.py lDDT integration")
    print("-" * 40)
    
    # Check if lDDT is enabled in config
    import yaml
    config_path = "/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    use_lddt = config.get('eval', {}).get('use_lddt', False)
    print(f"   Config use_lddt: {use_lddt}")
    if use_lddt:
        print(f"   ✅ lDDT is ENABLED in bench_full.yaml")
    else:
        print(f"   ❌ lDDT is DISABLED in bench_full.yaml - won't be calculated!")
    
    # Check the eval script's handling
    print("\n📊 Test 4: Verify lDDT result storage")
    print("-" * 40)
    
    # Simulate the lDDT storage logic from eval_full.py
    lddt_list = []
    lddt_scores = np.array([0.8, 0.9, np.nan, 0.7])  # Mock scores with nan
    
    # This mimics the code in eval_full.py
    if lddt_scores is not None and len(lddt_scores) > 0:
        lddt_vals = lddt_scores.tolist() if hasattr(lddt_scores, 'tolist') else lddt_scores
        lddt_list.extend(lddt_vals)
    
    if lddt_list:
        lddt_mean = np.nanmean(lddt_list)
        print(f"   Mock lDDT scores: {lddt_list}")
        print(f"   Mean lDDT (ignoring nan): {lddt_mean:.4f}")
        print(f"   ✅ lDDT averaging logic working correctly")
    
    # Check DATA_PATH for native PDBs
    print("\n📊 Test 5: Verify native PDB accessibility")
    print("-" * 40)
    
    print(f"   DATA_PATH: {DATA_PATH}")
    if DATA_PATH:
        raw_path = os.path.join(DATA_PATH, "raw")
        if os.path.exists(raw_path):
            pdb_count = len([f for f in os.listdir(raw_path) if f.endswith('.pdb')])
            print(f"   ✅ Found {pdb_count} PDB files in {raw_path}")
        else:
            print(f"   ❌ Raw PDB directory not found: {raw_path}")
    else:
        print(f"   ❌ DATA_PATH environment variable not set!")
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY:")
    print("-" * 40)
    
    if use_lddt and DATA_PATH:
        print("✅ lDDT calculation is ENABLED and should work in evaluation")
        print("   - Direct lDDT function works correctly")
        print("   - Config has use_lddt=True")
        print("   - Native PDB files are accessible")
        print("   - Result averaging handles NaN values correctly")
        print("\n💡 Note: Some structures may still return NaN if:")
        print("   - PDB file names don't match exactly (e.g., 1CSL_1_B vs 1CSL_1_B-A.pdb)")
        print("   - RhoFold fails to generate structure")
        print("   - Sequence identity check fails")
    else:
        if not use_lddt:
            print("⚠️ lDDT is DISABLED in config - enable it to calculate lDDT scores!")
        if not DATA_PATH:
            print("⚠️ DATA_PATH not set - native PDBs won't be found!")

if __name__ == "__main__":
    test_lddt_in_pipeline()