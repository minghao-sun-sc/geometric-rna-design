#!/usr/bin/env python3
"""Test 3-mer diversity calculation"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

def test_3mer_diversity():
    from src.evaluator import get_three_mer_corr
    from src.constants import NUM_TO_LETTER
    
    # Test data
    native_seq = "GGAGAGCCGCCAGATCCCCACGAAGGGGCAGGGGCCACAGAAGAGACGCCACGGGCGCCGG"
    
    # Create some sample sequences (as strings)
    sample_seqs = [
        "GGAGAGCCGCCAGATCCCCACGAAGGGGCAGGGGCCACAGAAGAGACGCCACGGGCGCCGG",  # Same as native
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # All A's
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",  # Alternating GC
    ]
    
    # Create mask (all True for simplicity)
    mask_coords = np.ones(len(native_seq), dtype=bool)
    
    print("Testing 3-mer diversity calculation...")
    print(f"Native sequence length: {len(native_seq)}")
    print(f"Number of samples: {len(sample_seqs)}")
    
    try:
        # Call the function
        corr_scores = get_three_mer_corr(sample_seqs, native_seq, mask_coords)
        diversity_scores = 1.0 - corr_scores
        
        print("\n✅ Success! Results:")
        for i, (seq, corr, div) in enumerate(zip(sample_seqs[:10], corr_scores, diversity_scores)):
            if i == 0:
                print(f"  Sample {i} (identical): corr={corr:.3f}, diversity={div:.3f}")
            elif i == 1:
                print(f"  Sample {i} (all A's):   corr={corr:.3f}, diversity={div:.3f}")
            elif i == 2:
                print(f"  Sample {i} (GC repeat): corr={corr:.3f}, diversity={div:.3f}")
            else:
                print(f"  Sample {i}: corr={corr:.3f}, diversity={div:.3f}")
        
        print(f"\nMean diversity: {np.mean(diversity_scores):.3f}")
        
        # Expected results:
        print("\n📊 Expected behavior:")
        print("  - Identical sequence: corr ≈ 1.0, diversity ≈ 0.0")
        print("  - Very different: corr ≈ 0.0, diversity ≈ 1.0")
        print("  - Random sequences: diversity ≈ 0.3-0.7")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔍 Debugging the issue...")
        print("The function might expect different input format.")

if __name__ == "__main__":
    test_3mer_diversity()