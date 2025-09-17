#!/usr/bin/env python3

# Test the threshold fixes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

def test_threshold_constants():
    try:
        from src.constants import RMSD_THRESHOLD, RMSD_THRESHOLD_2, PLDDT_THRESHOLD
        
        print("✅ Constants imported successfully:")
        print(f"  RMSD_THRESHOLD (8Å): {RMSD_THRESHOLD}")
        print(f"  RMSD_THRESHOLD_2 (2Å): {RMSD_THRESHOLD_2}")
        print(f"  PLDDT_THRESHOLD (0.70): {PLDDT_THRESHOLD}")
        
        # Test threshold calculations
        import numpy as np
        
        # Mock data
        rmsd_values = np.array([1.5, 5.0, 10.0, 15.0])  # Mix of good/bad RMSDs
        plddt_values = np.array([0.85, 0.75, 0.65, 0.45])  # Mix of good/bad pLDDTs
        
        rmsd_2A_pct = (rmsd_values <= RMSD_THRESHOLD_2).sum() / len(rmsd_values)
        rmsd_8A_pct = (rmsd_values <= RMSD_THRESHOLD).sum() / len(rmsd_values)
        plddt_070_pct = (plddt_values >= PLDDT_THRESHOLD).sum() / len(plddt_values)
        
        print(f"\n✅ Mock threshold calculations:")
        print(f"  % RMSD ≤ 2Å: {rmsd_2A_pct:.2%}")
        print(f"  % RMSD ≤ 8Å: {rmsd_8A_pct:.2%}")
        print(f"  % pLDDT ≥ 0.70: {plddt_070_pct:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Threshold test failed: {e}")
        return False

if __name__ == "__main__":
    test_threshold_constants()