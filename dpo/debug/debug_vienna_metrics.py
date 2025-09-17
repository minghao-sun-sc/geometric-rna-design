#!/usr/bin/env python3
"""
Debug script to test Vienna RNA metrics functionality.
Tests MFE, ensemble defect, Shannon entropy, and melting temperature calculations.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import traceback
from typing import List, Dict, Any

# Import Vienna RNA functions
from src.evaluator import (
    vienna_mfe,
    vienna_ensemble_metrics, 
    vienna_Tm_by_pS0
)

def test_vienna_tools():
    """Test if Vienna RNA tools are accessible."""
    print("Testing Vienna RNA tool accessibility...")
    
    try:
        import subprocess
        
        # Test RNAfold
        result = subprocess.run(['RNAfold', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✓ RNAfold available: {result.stdout.strip()}")
        
        # Test RNAeval  
        result = subprocess.run(['RNAeval', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✓ RNAeval available: {result.stdout.strip()}")
        
        # Test RNAheat
        result = subprocess.run(['RNAheat', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✓ RNAheat available: {result.stdout.strip()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking Vienna tools: {e}")
        return False


def test_vienna_mfe():
    """Test Vienna MFE calculation."""
    print("\n" + "="*50)
    print("TESTING VIENNA MFE")
    print("="*50)
    
    test_sequences = [
        "AUGCGCUAGCUA",
        "GGGAAAUUUCCC", 
        "AUCGAUCGAUCGAUCG",
        "AUGCGCUAGCUAGCUGCGCAU"
    ]
    
    for i, seq in enumerate(test_sequences, 1):
        print(f"\nTest {i}: {seq}")
        try:
            mfe_val, ss = vienna_mfe(seq, T=37.0)
            print(f"  MFE: {mfe_val:.2f} kcal/mol")
            print(f"  Secondary structure: {ss}")
            
            # Test different temperatures
            mfe_30, _ = vienna_mfe(seq, T=30.0)
            mfe_50, _ = vienna_mfe(seq, T=50.0)
            print(f"  MFE @ 30°C: {mfe_30:.2f} kcal/mol")
            print(f"  MFE @ 50°C: {mfe_50:.2f} kcal/mol")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()


def test_vienna_ensemble():
    """Test Vienna ensemble metrics."""
    print("\n" + "="*50) 
    print("TESTING VIENNA ENSEMBLE METRICS")
    print("="*50)
    
    test_cases = [
        {
            'seq': "GGGAAAUUUCCC",
            'target_db': "(((.....)))"  # Target secondary structure (same length: 12)
        },
        {
            'seq': "AUCGAUCGAUCGAUCG", 
            'target_db': "................"  # No structure (same length: 16)
        },
        {
            'seq': "AUGCGCUAGCUAGCUGCGCAU",
            'target_db': "(((((...)))))........" # Complex structure (same length: 21)
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        seq = case['seq']
        target_db = case['target_db']
        print(f"\nTest {i}: {seq}")
        print(f"  Target: {target_db}")
        
        try:
            # Test with target structure
            metrics = vienna_ensemble_metrics(
                seq, target_db=target_db, T=37.0, return_positional_entropy=False
            )
            
            print(f"  Ensemble Defect: {metrics.get('ED', 'N/A')}")
            print(f"  Shannon Entropy: {metrics.get('shannon_entropy', 'N/A')}")
            print(f"  Frequency: {metrics.get('frequency', 'N/A')}")
            
            # Test without target (should use MFE structure)
            metrics_mfe = vienna_ensemble_metrics(
                seq, target_db=None, T=37.0, return_positional_entropy=False
            )
            print(f"  ED (vs MFE): {metrics_mfe.get('ED', 'N/A')}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()


def test_vienna_tm():
    """Test melting temperature calculation."""
    print("\n" + "="*50)
    print("TESTING VIENNA MELTING TEMPERATURE")
    print("="*50)
    
    test_cases = [
        {
            'seq': "GGGAAAUUUCCC",
            'target_db': "(((.....)))"  # Should have clear melting (same length: 12)
        },
        {
            'seq': "GCGCGCGCGCGC", 
            'target_db': "(((((((((((." # High GC content (same length: 12)
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        seq = case['seq']
        target_db = case['target_db']
        print(f"\nTest {i}: {seq}")
        print(f"  Target: {target_db}")
        
        try:
            tm = vienna_Tm_by_pS0(seq, target_db, Tmin=10, Tmax=95, step=5.0, threshold=0.5)
            print(f"  Melting Temperature: {tm:.1f}°C")
            
            # Test different thresholds
            tm_30 = vienna_Tm_by_pS0(seq, target_db, Tmin=10, Tmax=95, step=5.0, threshold=0.3)
            tm_70 = vienna_Tm_by_pS0(seq, target_db, Tmin=10, Tmax=95, step=5.0, threshold=0.7)
            print(f"  Tm @ 30% threshold: {tm_30:.1f}°C")
            print(f"  Tm @ 70% threshold: {tm_70:.1f}°C")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "="*50)
    print("TESTING ERROR HANDLING")
    print("="*50)
    
    # Invalid sequences
    invalid_seqs = [
        "",           # Empty
        "ATCGXYZ",    # Invalid characters  
        "A",          # Too short
        "ATCG" * 500  # Very long
    ]
    
    for seq in invalid_seqs:
        print(f"\nTesting invalid sequence: {seq[:20]}{'...' if len(seq) > 20 else ''}")
        
        try:
            mfe_val, _ = vienna_mfe(seq, T=37.0)
            print(f"  Unexpected success: MFE = {mfe_val}")
        except Exception as e:
            print(f"  ✓ Correctly handled error: {type(e).__name__}")


def main():
    print("🧪 VIENNA RNA METRICS DEBUG SUITE")
    print("="*60)
    
    # Check tool availability
    if not test_vienna_tools():
        print("❌ Vienna RNA tools not properly installed. Aborting tests.")
        sys.exit(1)
    
    # Run tests
    test_vienna_mfe()
    test_vienna_ensemble()
    test_vienna_tm()
    test_error_handling()
    
    print("\n" + "="*60)
    print("🎉 Vienna RNA metrics testing complete!")
    print("Check output above for any errors or unexpected behavior.")
    print("="*60)


if __name__ == "__main__":
    main()