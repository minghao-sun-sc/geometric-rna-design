#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

# Test Vienna metrics integration in eval_full.py
def test_vienna_integration():
    try:
        from src.evaluator import vienna_ensemble_metrics, vienna_mfe, _sanitize_db_for_vienna
        
        # Test a simple sequence (matching lengths)
        test_seq = "GGAGAGCCGCCAGATCCCCACGAAGGGGCAGGGGCCACAGAAGAGACGCCACGGGCGCCGGGTGGGAGGGAAGGCCCGCCCCAA"
        target_db = "(((.((.((((((.......)))).))..))))))..(((((.(((((.......))))).)))))))................"
        
        print(f"Testing Vienna metrics with:")
        print(f"  Sequence: {test_seq}")
        print(f"  Target:   {target_db}")
        
        # Test Vienna functions
        mfe, mfe_db = vienna_mfe(test_seq, 37.0)
        print(f"\nMFE: {mfe:.2f} kcal/mol")
        print(f"MFE structure: {mfe_db}")
        
        # Test ensemble metrics
        v = vienna_ensemble_metrics(test_seq, target_db=target_db, T=37.0)
        print(f"\nEnsemble metrics:")
        print(f"  MFE: {v['mfe']:.2f} kcal/mol")
        print(f"  Ensemble Defect: {v['ED']:.2f} nt")
        print(f"  ED per nt: {v['ED_per_nt']:.4f}")
        print(f"  P(target): {v['pS0']:.4f}")
        print(f"  Shannon Entropy: {v['entropy_mean']:.3f}")
        print(f"  Diversity: {v['diversity']:.3f}")
        
        print("\n✅ Vienna metrics integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Vienna integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_vienna_integration()