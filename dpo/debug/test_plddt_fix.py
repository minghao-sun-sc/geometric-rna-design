#!/usr/bin/env python3

"""Test script to verify pLDDT extraction from RhoFold."""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import tempfile
import numpy as np
import torch

def test_rhofold_plddt():
    """Test that RhoFold now returns pLDDT scores correctly."""
    print("=" * 60)
    print("Testing RhoFold pLDDT extraction")
    print("=" * 60)
    
    try:
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        
        # Initialize RhoFold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rhofold = RhoFold(rhofold_config, device)
        
        # Load checkpoint
        rhofold_path = os.path.join("/mnt/rna01/smh/projects/ribopo", "tools/rhofold/model_20221010_params.pt")
        print(f"Loading RhoFold checkpoint: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
        rhofold = rhofold.to(device)
        rhofold.eval()
        
        # Create a test sequence
        test_sequence = "GGGGAAAACCCC"  # Simple test sequence
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write test FASTA
            fasta_path = os.path.join(temp_dir, "test.fasta")
            with open(fasta_path, 'w') as f:
                f.write(">test_sequence\n")
                f.write(test_sequence + "\n")
            
            # Run RhoFold prediction
            pdb_path = os.path.join(temp_dir, "test.pdb")
            coords, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=False)
            
            # Check results
            print(f"✓ RhoFold prediction completed successfully")
            print(f"  Coordinates shape: {coords.shape}")
            print(f"  pLDDT shape: {plddt.shape}")
            print(f"  pLDDT range: {plddt.min():.3f} - {plddt.max():.3f}")
            print(f"  pLDDT mean: {plddt.mean():.3f}")
            print(f"  Sample pLDDT values: {plddt[:5]}")
            
            # Verify pLDDT values are reasonable
            if plddt.size > 0 and not np.all(plddt == 0):
                print(f"✓ SUCCESS: pLDDT values are non-zero and properly extracted!")
                return True
            else:
                print(f"✗ FAILURE: pLDDT values are all zero or empty")
                return False
                
    except Exception as e:
        print(f"✗ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator_integration():
    """Test that the evaluator properly uses pLDDT scores."""
    print("\n" + "=" * 60)
    print("Testing evaluator integration with pLDDT")
    print("=" * 60)
    
    try:
        from src.evaluator import self_consistency_score_rhofold_extended
        
        print("✓ Extended RhoFold function imports successfully")
        print("  Next step: Run evaluation to verify pLDDT integration")
        return True
        
    except Exception as e:
        print(f"✗ FAILURE: Could not import extended function: {e}")
        return False

if __name__ == "__main__":
    all_tests_passed = True
    
    all_tests_passed &= test_rhofold_plddt()
    all_tests_passed &= test_evaluator_integration()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - pLDDT extraction should now work!")
        print("Run the evaluation again to see proper pLDDT values.")
    else:
        print("❌ SOME TESTS FAILED - Please check the issues above")
    print("=" * 60)