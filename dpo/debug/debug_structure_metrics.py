#!/usr/bin/env python3
"""
Debug script to test structure-based metrics functionality.
Tests EternaFold, RhoFold, USalign, and other structural evaluation tools.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import tempfile
import traceback
from typing import List, Dict, Any
import torch
import numpy as np

# Import structure evaluation functions
from src.evaluator import (
    self_consistency_score_eternafold,
    self_consistency_score_rhofold,
    get_three_mer_corr
)

def create_mock_raw_data(sequence: str) -> Dict[str, Any]:
    """Create mock raw data for testing purposes."""
    seq_len = len(sequence)
    
    # Create mock 3D coordinates (simple linear chain)
    coords = torch.randn(seq_len, 3) * 2.0  # Random coordinates
    coords_list = [coords]
    
    # Create simple secondary structure
    sec_struct = "." * seq_len  # No structure
    sec_struct_list = [sec_struct]
    
    raw_data = {
        'sequence': sequence,
        'coords_list': coords_list,
        'sec_struct_list': sec_struct_list,
        'id_list': ['test_structure'],
        'rfam_list': ['RF00000'],
        'eq_class_list': ['test'],
        'cluster_structsim0.45': 'test_cluster'
    }
    
    return raw_data


def test_eternafold_integration():
    """Test EternaFold 2D self-consistency."""
    print("\n" + "="*50)
    print("TESTING ETERNAFOLD INTEGRATION")
    print("="*50)
    
    # Test data
    test_sequences = [
        "GGGAAAUUUCCC",
        "AUCGAUCGAUCG", 
        "AUGCGCUAGCUAGCUGCGCAU"
    ]
    
    for i, seq in enumerate(test_sequences, 1):
        print(f"\nTest {i}: {seq}")
        
        # Create mock data
        raw_data = create_mock_raw_data(seq)
        
        # Create candidate sequences (including original + variations)
        candidates = [
            seq,  # Original
            seq.replace('A', 'G', 1) if 'A' in seq else seq + 'A',  # Small variation
            seq.replace('U', 'C', 1) if 'U' in seq else seq + 'C',  # Another variation
        ]
        
        print(f"  Candidates: {len(candidates)}")
        for j, cand in enumerate(candidates):
            print(f"    {j+1}: {cand}")
        
        try:
            # Test EternaFold self-consistency
            results = self_consistency_score_eternafold(candidates, raw_data)
            
            print(f"  Results:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Print more details for debugging
            print(f"  Error type: {type(e).__name__}")
            traceback.print_exc()


def test_rhofold_integration():
    """Test RhoFold 3D self-consistency.""" 
    print("\n" + "="*50)
    print("TESTING RHOFOLD INTEGRATION")
    print("="*50)
    
    # Use shorter sequences for RhoFold (it's computationally expensive)
    test_sequences = [
        "GGGAAAUUUCCC",
        "AUCGAUCGAUCG"
    ]
    
    for i, seq in enumerate(test_sequences, 1):
        print(f"\nTest {i}: {seq}")
        
        # Create mock data with better coordinates
        raw_data = create_mock_raw_data(seq)
        
        # Create fewer candidates for RhoFold (expensive)
        candidates = [
            seq,  # Original
            seq.replace('A', 'G', 1) if 'A' in seq else seq + 'A',  # One variation
        ]
        
        print(f"  Candidates: {len(candidates)}")
        for j, cand in enumerate(candidates):
            print(f"    {j+1}: {cand}")
        
        try:
            # Test RhoFold self-consistency
            print(f"  Running RhoFold... (this may take a while)")
            results = self_consistency_score_rhofold(candidates, raw_data)
            
            print(f"  Results:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Error type: {type(e).__name__}")
            # Don't print full traceback for RhoFold errors (often environmental)
            print("  (RhoFold errors are often due to missing dependencies)")


def test_diversity_metrics():
    """Test sequence diversity metrics."""
    print("\n" + "="*50)
    print("TESTING DIVERSITY METRICS") 
    print("="*50)
    
    test_cases = [
        {
            'native': "GGGAAAUUUCCC",
            'designed': [
                "GGGAAAUUUCCC",  # Identical
                "GGGAAAUUUCCA",  # 1 difference
                "GGGAAAUUUCCA",  # 2 differences
                "AAAGGGUUUCCC",  # More differences
            ]
        },
        {
            'native': "AUCGAUCGAUCG",
            'designed': [
                "AUCGAUCGAUCG",  # Identical
                "GUCGAUCGAUCG",  # Different start
                "AUCGAUCGAUCG",  # Identical (duplicate)
                "CCCCCCCCCCCC",  # Very different
            ]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        native = case['native']
        designed = case['designed']
        
        print(f"\nTest {i}: {native}")
        print(f"  Designed sequences: {len(designed)}")
        
        for j, seq in enumerate(designed):
            print(f"    {j+1}: {seq}")
        
        try:
            # Test 3-mer correlation
            mask_coords = None  # No masking for now
            diversity_score = get_three_mer_corr(designed, native, mask_coords)
            
            print(f"  3-mer correlation score: {diversity_score:.4f}")
            print(f"  (Lower = more diverse from native)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()


def test_tool_availability():
    """Test availability of external structure tools."""
    print("\n" + "="*50)
    print("TESTING EXTERNAL TOOL AVAILABILITY")
    print("="*50)
    
    tools_to_check = [
        ('EternaFold', '/mnt/rna01/smh/projects/ribopo/tools/EternaFold'),
        ('RhoFold', '/mnt/rna01/smh/projects/ribopo/tools/rhofold'),  
        ('USalign', '/mnt/rna01/smh/projects/ribopo/tools/USalign'),
        ('RNA_assessment', '/mnt/rna01/smh/projects/ribopo/tools/RNA_assessment'),
        ('Phenix', '/mnt/rna01/smh/projects/ribopo/tools/phenix-1.21.2-5419'),
        ('x3dna', '/mnt/rna01/smh/projects/ribopo/tools/x3dna-v2.4')
    ]
    
    for tool_name, tool_path in tools_to_check:
        print(f"\n{tool_name}:")
        if os.path.exists(tool_path):
            print(f"  ✓ Directory exists: {tool_path}")
            
            # Check for executables
            if tool_name == 'USalign':
                exe_path = os.path.join(tool_path, 'USalign')
                if os.path.isfile(exe_path):
                    print(f"  ✓ Executable found: {exe_path}")
                else:
                    print(f"  ✗ Executable not found: {exe_path}")
                    
            elif tool_name == 'EternaFold':
                # EternaFold might have different structure
                print(f"  ? Structure check needed for EternaFold")
                
            else:
                # List contents
                try:
                    contents = os.listdir(tool_path)
                    print(f"  Contents: {len(contents)} items")
                    # Show first few items
                    for item in contents[:3]:
                        print(f"    - {item}")
                    if len(contents) > 3:
                        print(f"    ... and {len(contents) - 3} more")
                except:
                    print(f"  ✗ Cannot list directory contents")
        else:
            print(f"  ✗ Directory not found: {tool_path}")


def main():
    print("🧬 STRUCTURE METRICS DEBUG SUITE")
    print("="*60)
    
    # Test external tool availability first
    test_tool_availability()
    
    # Test diversity metrics (fast)
    test_diversity_metrics()
    
    # Test EternaFold integration (moderate speed)
    test_eternafold_integration()
    
    # Test RhoFold integration (slow - comment out if needed)
    print(f"\nNote: RhoFold tests are slow and may fail due to environment setup.")
    print(f"Uncomment the line below to enable RhoFold testing:")
    print(f"# test_rhofold_integration()")
    
    # Uncomment to test RhoFold (warning: slow!)
    # test_rhofold_integration()
    
    print("\n" + "="*60)
    print("🎉 Structure metrics testing complete!")
    print("Check output above for any errors or missing tools.")
    print("="*60)


if __name__ == "__main__":
    main()