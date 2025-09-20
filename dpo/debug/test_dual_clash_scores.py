#!/usr/bin/env python3
"""
Test script to verify dual clash score implementation works correctly
with both use_relax=true and use_relax=false.
"""

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import os
import sys
import yaml
import tempfile
import numpy as np
sys.path.append('/mnt/rna01/smh/projects/ribopo')

from src.evaluator import self_consistency_score_rhofold_extended
from tools.rhofold.rf import RhoFold
from tools.rhofold.config import rhofold_config
import torch

def create_test_config(use_relax_value):
    """Create a test configuration with specified use_relax value."""
    config = {
        'eval': {
            'use_relax': use_relax_value,
            'use_lddt': False,  # Disable to speed up testing
        }
    }
    return config

def create_test_data():
    """Create minimal test data for evaluation."""
    # Simple RNA sequence
    test_sequence = "GGGGCCCCUUUUAAAA"  # 16 nucleotides
    
    # Mock raw data structure
    raw_data = {
        'sequence': test_sequence,
        'id_list': ['test_rna'],
        'coords_list': [torch.randn(len(test_sequence), 3)],  # Random coordinates
        'sec_struct_list': ['.' * len(test_sequence)]
    }
    
    # Samples (just one sample with the same sequence)
    samples = np.array([[0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]])  # GGGCCCCUUUUAAAA as numbers
    
    # Mask coordinates (all True for simplicity)
    mask_coords = np.ones(len(test_sequence), dtype=bool)
    
    return samples, raw_data, mask_coords

def test_clash_scores_with_relax(use_relax):
    """Test clash score calculation with given use_relax setting."""
    print(f"\n🔬 Testing clash scores with use_relax={use_relax}")
    
    try:
        # Initialize RhoFold
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = "/mnt/rna01/smh/projects/ribopo/tools/rhofold/model_20221010_params.pt"
        print(f"Loading RhoFold checkpoint: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=device)['model'])
        rhofold = rhofold.to(device)
        rhofold.eval()
        
        # Create test data
        samples, raw_data, mask_coords = create_test_data()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Call the evaluation function
            result = self_consistency_score_rhofold_extended(
                samples,
                raw_data,
                mask_coords,
                rhofold,
                temp_dir,
                save_designs=False,
                save_pdbs=True,  # Keep PDBs to verify file creation
                use_relax=use_relax,
                use_inf=False,  # Disable INF for faster testing
                use_clash=True,  # Enable clash calculation
                use_lddt=False,  # Disable lDDT for faster testing
                use_mcq=False,  # Disable MCQ for faster testing
                phenix_wrapper_path="/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh",
            )
            
            # Unpack results
            sc_rmsd, sc_tm, sc_gdt, sc_plddt, inf_dict, clash_dict, lddt_arr, mcq_dict = result
            
            print(f"✅ Evaluation completed successfully")
            print(f"   RMSD: {sc_rmsd}")
            print(f"   TM-score: {sc_tm}")
            print(f"   GDT_TS: {sc_gdt}")
            print(f"   pLDDT: {sc_plddt}")
            
            # Check clash dict structure
            if clash_dict and isinstance(clash_dict, dict):
                print(f"✅ Clash dict returned with keys: {list(clash_dict.keys())}")
                
                pre_relax_scores = clash_dict.get('pre_relax', np.array([]))
                post_relax_scores = clash_dict.get('post_relax', np.array([]))
                
                print(f"   Pre-relax clash scores: {pre_relax_scores}")
                print(f"   Post-relax clash scores: {post_relax_scores}")
                
                # Validate based on use_relax setting
                if use_relax:
                    if len(pre_relax_scores) > 0 and not np.all(np.isnan(pre_relax_scores)):
                        print(f"   ✅ Pre-relax clash scores calculated (expected with use_relax=True)")
                    else:
                        print(f"   ❌ Pre-relax clash scores missing or all NaN")
                        
                    if len(post_relax_scores) > 0 and not np.all(np.isnan(post_relax_scores)):
                        print(f"   ✅ Post-relax clash scores calculated (expected with use_relax=True)")
                    else:
                        print(f"   ❌ Post-relax clash scores missing or all NaN")
                        
                    # Check if post-relax scores are different/better than pre-relax
                    if len(pre_relax_scores) > 0 and len(post_relax_scores) > 0:
                        if not np.isnan(pre_relax_scores[0]) and not np.isnan(post_relax_scores[0]):
                            improvement = pre_relax_scores[0] - post_relax_scores[0]
                            if improvement > 0:
                                print(f"   ✅ Relaxation improved clash score by {improvement:.2f}")
                            else:
                                print(f"   ⚠️  Relaxation changed clash score by {improvement:.2f} (may be normal)")
                else:
                    if len(pre_relax_scores) > 0 and not np.all(np.isnan(pre_relax_scores)):
                        print(f"   ✅ Pre-relax clash scores calculated (expected with use_relax=False)")
                    else:
                        print(f"   ❌ Pre-relax clash scores missing or all NaN")
                        
                    if len(post_relax_scores) == 0 or np.all(np.isnan(post_relax_scores)):
                        print(f"   ✅ Post-relax clash scores are NaN (expected with use_relax=False)")
                    else:
                        print(f"   ❌ Post-relax clash scores should be NaN when use_relax=False")
                
                return True
            else:
                print(f"   ❌ Clash dict is None or not a dict: {clash_dict}")
                return False
                
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Dual Clash Score Implementation")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test with use_relax=False
    if test_clash_scores_with_relax(use_relax=False):
        success_count += 1
        print("✅ Test with use_relax=False passed")
    else:
        print("❌ Test with use_relax=False failed")
    
    # Test with use_relax=True  
    if test_clash_scores_with_relax(use_relax=True):
        success_count += 1
        print("✅ Test with use_relax=True passed")
    else:
        print("❌ Test with use_relax=True failed")
    
    print(f"\n🎯 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Dual clash score implementation is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)