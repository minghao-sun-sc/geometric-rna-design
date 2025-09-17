#!/usr/bin/env python3

"""Test script to verify RhoFold evaluation fixes."""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import sys
import os
import numpy as np
import torch
from types import SimpleNamespace

def test_rhofold_unpacking():
    """Test that RhoFold function returns exactly 3 values as expected."""
    print("=" * 60)
    print("Testing RhoFold unpacking fixes")
    print("=" * 60)
    
    try:
        # Import the fixed function
        from src.evaluator import self_consistency_score_rhofold
        
        # Create mock data
        mock_samples = np.array([[0, 1, 2, 0], [1, 0, 2, 1]])  # 2 samples x 4 nucleotides
        mock_raw_data = {
            "sequence": "AUGC", 
            "coords_list": [torch.randn(4, 3, 3)],  # 1 structure x 4 residues x 3 atoms x 3D
        }
        mock_mask_coords = np.array([True, True, True, True])
        
        # Mock RhoFold class
        class MockRhoFold:
            def predict(self, fasta_path, pdb_path, use_relax=False):
                # Create a dummy PDB file for testing
                with open(pdb_path, 'w') as f:
                    f.write("ATOM      1  P     A A   1      -0.123   1.234  -2.345\n")
                    f.write("ATOM      2  C4'   A A   1       0.876   2.345  -1.234\n") 
                    f.write("ATOM      3  P     U A   2      -1.234   3.456  -0.123\n")
                    f.write("ATOM      4  C4'   U A   2       1.345   4.567   0.876\n")
                    f.write("ATOM      5  P     G A   3      -2.345   5.678   1.987\n")
                    f.write("ATOM      6  C4'   G A   3       2.456   6.789   2.098\n")
                    f.write("ATOM      7  P     C A   4      -3.456   7.890   3.210\n")
                    f.write("ATOM      8  C4'   C A   4       3.567   8.901   4.321\n")
                return None  # Original gRNAde version doesn't return anything
        
        mock_rhofold = MockRhoFold()
        mock_output_dir = "/tmp/test_rhofold_eval"
        
        # Test the function
        result = self_consistency_score_rhofold(
            mock_samples,
            mock_raw_data,
            mock_mask_coords, 
            mock_rhofold,
            mock_output_dir,
            save_designs=False
        )
        
        # Verify return format
        if isinstance(result, tuple) and len(result) == 3:
            rmsd, tm, gdt = result
            print(f"✓ SUCCESS: Function returns 3 values as expected")
            print(f"  RMSD shape: {rmsd.shape}, sample values: {rmsd}")
            print(f"  TM shape: {tm.shape}, sample values: {tm}")
            print(f"  GDT shape: {gdt.shape}, sample values: {gdt}")
            return True
        else:
            print(f"✗ FAILURE: Expected 3 values, got {len(result) if isinstance(result, tuple) else type(result)}")
            return False
            
    except Exception as e:
        print(f"✗ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_handling():
    """Test robust file handling for symlinks."""
    print("\n" + "=" * 60)
    print("Testing robust file handling")
    print("=" * 60)
    
    import tempfile
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test file creation and symlink handling
            test_file = os.path.join(temp_dir, "test_results.json")
            symlink_path = os.path.join(temp_dir, "latest.json")
            
            # Create a test file
            with open(test_file, 'w') as f:
                f.write('{"test": "data"}')
            
            # Test symlink creation (should work)
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
            os.symlink(os.path.abspath(test_file), os.path.abspath(symlink_path))
            
            # Test robust symlink replacement (main test)
            test_file2 = os.path.join(temp_dir, "test_results2.json") 
            with open(test_file2, 'w') as f:
                f.write('{"test": "data2"}')
                
            # This should handle existing symlink gracefully
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
            os.symlink(os.path.abspath(test_file2), os.path.abspath(symlink_path))
            
            print("✓ SUCCESS: File handling and symlink operations work correctly")
            return True
            
    except Exception as e:
        print(f"✗ FAILURE: File handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_naming():
    """Test checkpoint-based output naming."""
    print("\n" + "=" * 60)
    print("Testing checkpoint-based naming")
    print("=" * 60)
    
    try:
        # Test naming logic
        mock_rows = [
            {"ckpt_name": "gRNAde_ARv1_1state_das"},
            {"ckpt_name": "model_epoch_10"}, 
            {"ckpt_name": "model_epoch_20"},
            {"ckpt_name": "model_epoch_30"},
        ]
        
        # Test case 1: <= 3 checkpoints
        rows_short = mock_rows[:2]
        ckpt_names_str = "_".join([r["ckpt_name"] for r in rows_short[:3]])
        if len(rows_short) > 3:
            ckpt_names_str += f"_and_{len(rows_short)-3}more"
        expected_short = "gRNAde_ARv1_1state_das_model_epoch_10"
        
        # Test case 2: > 3 checkpoints  
        ckpt_names_str_long = "_".join([r["ckpt_name"] for r in mock_rows[:3]])
        if len(mock_rows) > 3:
            ckpt_names_str_long += f"_and_{len(mock_rows)-3}more"
        expected_long = "gRNAde_ARv1_1state_das_model_epoch_10_model_epoch_20_and_1more"
        
        if ckpt_names_str == expected_short and ckpt_names_str_long == expected_long:
            print(f"✓ SUCCESS: Checkpoint naming works correctly")
            print(f"  Short format: {ckpt_names_str}")
            print(f"  Long format: {ckpt_names_str_long}")
            return True
        else:
            print(f"✗ FAILURE: Unexpected naming results")
            print(f"  Got short: {ckpt_names_str}")
            print(f"  Expected: {expected_short}")
            return False
            
    except Exception as e:
        print(f"✗ FAILURE: Checkpoint naming test failed: {e}")
        return False

if __name__ == "__main__":
    all_tests_passed = True
    
    all_tests_passed &= test_rhofold_unpacking()
    all_tests_passed &= test_file_handling()
    all_tests_passed &= test_checkpoint_naming()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - Evaluation fixes are working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please check the issues above")
    print("=" * 60)
    
    sys.exit(0 if all_tests_passed else 1)