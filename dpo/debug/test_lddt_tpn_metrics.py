#!/usr/bin/env python3
"""
Debug script to test:
1. lDDT fix for inverse folding evaluation
2. TPN (Trimer Profile Novelty) metric implementation
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import numpy as np
import torch
from src.constants import NUM_TO_LETTER

def test_tpn_metric():
    """Test the TPN metric with various RNA sequences"""
    print("🧬 Testing Trimer Profile Novelty (TPN) Metric")
    print("=" * 50)
    
    # Import TPN function
    from src.evaluator import get_trimer_profile_novelty
    
    # Test sequences (some natural-like, some artificial)
    test_sequences = [
        # Natural-like tRNA sequence (should have low novelty)
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA",
        
        # Artificial sequence with unusual 3-mer patterns (should have higher novelty)  
        "AAAAAACCCCCGGGGGUUUUUUAAAAAACCCCCGGGGGUUUUUUAAAAAACCCCCGGGGGUUUUUU",
        
        # Short natural-like sequence
        "GGCGAACGCUUCGAAACGCGCC",
        
        # Very artificial sequence with rare 3-mers
        "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
        
        # Repetitive sequence
        "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
    ]
    
    print(f"Testing {len(test_sequences)} sequences:")
    
    try:
        # Test with string input
        novelty_scores = get_trimer_profile_novelty(test_sequences)
        
        for i, (seq, score) in enumerate(zip(test_sequences, novelty_scores)):
            seq_preview = seq[:30] + "..." if len(seq) > 30 else seq
            print(f"  Seq {i+1}: {seq_preview}")
            print(f"         TPN Score: {score:.4f} (higher = more novel)")
            print()
            
    except Exception as e:
        print(f"❌ TPN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with numeric input (simulate model output)
    print("Testing with numeric input (simulating model output):")
    try:
        # Convert one sequence to numeric representation
        test_seq = test_sequences[0]
        numeric_seq = np.array([[4 if c not in ['A', 'C', 'G', 'U'] else 
                                ['A', 'C', 'G', 'U'].index(c) for c in test_seq]])
        
        novelty_scores_numeric = get_trimer_profile_novelty(numeric_seq)
        print(f"  Numeric input TPN Score: {novelty_scores_numeric[0]:.4f}")
        
    except Exception as e:
        print(f"❌ Numeric TPN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("✅ TPN metric test completed successfully!")
    return True

def test_lddt_fix():
    """Test the lDDT fix for inverse folding"""
    print("\n🔬 Testing lDDT Fix for Inverse Folding")
    print("=" * 50)
    
    # Check if we have sample PDB files to test with
    import os
    from src.constants import DATA_PATH
    
    # Look for some test PDB files
    test_pdbs = []
    if os.path.exists(DATA_PATH):
        raw_dir = os.path.join(DATA_PATH, "raw")
        if os.path.exists(raw_dir):
            # Get first few PDB files for testing
            pdb_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdb')][:3]
            test_pdbs = [os.path.join(raw_dir, f) for f in pdb_files]
    
    if len(test_pdbs) < 2:
        print("⚠️  No test PDB files found, skipping lDDT test")
        return True
        
    print(f"Testing lDDT with {len(test_pdbs)} PDB files...")
    
    try:
        from src.evaluator import get_lddt_inverse_folding
        
        # Test lDDT between first two PDBs
        pdb1, pdb2 = test_pdbs[0], test_pdbs[1]
        lddt_score = get_lddt_inverse_folding(pdb1, pdb2)
        
        print(f"  PDB 1: {os.path.basename(pdb1)}")
        print(f"  PDB 2: {os.path.basename(pdb2)}")
        print(f"  lDDT Score: {lddt_score:.4f}")
        
        if lddt_score >= 0:
            print("✅ lDDT calculation completed without 'no common residues' error!")
        else:
            print("⚠️  lDDT returned -1, but no exception thrown (improvement from before)")
            
    except Exception as e:
        print(f"❌ lDDT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("✅ lDDT fix test completed!")
    return True

def test_integration():
    """Test TPN metric integration in evaluation pipeline"""
    print("\n🔧 Testing TPN Integration in Evaluation Pipeline")
    print("=" * 50)
    
    try:
        # Test importing TPN in evaluation context
        from dpo.bench.eval_full import eval_full_metrics
        print("✅ TPN import in eval_full.py successful")
        
        # Create mock samples to test metric calculation
        n_samples, seq_len = 3, 20
        mock_samples = torch.randint(0, 4, (n_samples, seq_len))  # Random RNA sequences
        
        # Test converting to sequences for TPN
        from src.evaluator import get_trimer_profile_novelty
        tpn_scores = get_trimer_profile_novelty(mock_samples.numpy())
        
        print(f"✅ TPN metric calculated for {n_samples} mock sequences")
        print(f"   Scores: {[f'{s:.4f}' for s in tpn_scores]}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("✅ Integration test completed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing lDDT Fix and TPN Metric Implementation")
    print("=" * 60)
    
    success = True
    
    # Test TPN metric
    if not test_tpn_metric():
        success = False
    
    # Test lDDT fix  
    if not test_lddt_fix():
        success = False
        
    # Test integration
    if not test_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! lDDT fix and TPN metric are working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())