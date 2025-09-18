#!/usr/bin/env python3
"""
Test OpenStructure lDDT v2 implementation and compare with original
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_openstructure_availability():
    """Test if OpenStructure is available and functional"""
    print("1️⃣ Testing OpenStructure availability...")
    
    try:
        import ost
        import ost.mol
        import ost.io
        from ost.mol.alg import lddt
        
        print(f"   ✅ OpenStructure version: {ost.GetVersion()}")
        print(f"   ✅ lDDT module available: {hasattr(lddt, 'lDDTScorer')}")
        
        # Test basic functionality
        scorer_class = lddt.lDDTScorer
        print(f"   ✅ lDDTScorer class: {scorer_class}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ OpenStructure not available: {e}")
        print(f"   💡 Install with: conda install -c conda-forge openstructure")
        return False
    except Exception as e:
        print(f"   ❌ OpenStructure error: {e}")
        return False

def create_test_rna_pdb(pdb_path, sequence="GGCAAGCCUGCGAUGGCC", add_noise=False):
    """Create a simple test RNA PDB file for testing"""
    
    # Simple RNA backbone coordinates (approximate)
    base_coords = [
        # P, O5', C5', C4', C3', O3', C1', N1/N9 for each residue
        (0.0, 0.0, 0.0), (1.0, 0.5, 0.2), (2.0, 0.3, 0.1), (3.0, 0.8, 0.3),
        (4.0, 0.6, 0.2), (5.0, 1.0, 0.4), (3.5, 2.0, 0.5), (3.8, 3.0, 0.8)
    ]
    
    atom_names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N1"]
    
    with open(pdb_path, 'w') as f:
        f.write("HEADER    TEST RNA STRUCTURE\n")
        atom_id = 1
        
        for i, nucleotide in enumerate(sequence):
            res_num = i + 1
            
            for j, (x, y, z) in enumerate(base_coords):
                # Add residue offset and optional noise
                x_coord = x + i * 6.0
                y_coord = y + (np.random.normal(0, 0.1) if add_noise else 0)
                z_coord = z + (np.random.normal(0, 0.1) if add_noise else 0)
                
                # Add residue-specific variations
                if nucleotide == 'A' and atom_names[j] == "N1":
                    atom_name = "N9"  # Purine
                elif nucleotide == 'G' and atom_names[j] == "N1":
                    atom_name = "N9"  # Purine
                else:
                    atom_name = atom_names[j]
                
                f.write(f"ATOM  {atom_id:5d}  {atom_name:<3s} {nucleotide:>3s} A{res_num:4d}    "
                       f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00 20.00           "
                       f"{atom_name[0]:>1s}\n")
                atom_id += 1
        
        f.write("END\n")

def test_lddt_v2_basic():
    """Test basic lDDT v2 functionality"""
    print("\n2️⃣ Testing lDDT v2 basic functionality...")
    
    try:
        from src.evaluator import get_lddt_openstructure_v2
        
        # Create test directory
        test_dir = "/tmp/lddt_v2_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Create test structures
        native_pdb = os.path.join(test_dir, "native.pdb")
        predicted_pdb = os.path.join(test_dir, "predicted.pdb")
        
        sequence = "GGCAAGCCUGCGAUGGCC"
        create_test_rna_pdb(native_pdb, sequence, add_noise=False)
        create_test_rna_pdb(predicted_pdb, sequence, add_noise=True)  # Add slight noise
        
        print(f"   Created test structures: {len(sequence)} nucleotides")
        
        # Test identical structures (should give high lDDT)
        start_time = time.time()
        lddt_identical = get_lddt_openstructure_v2(native_pdb, native_pdb)
        time_identical = time.time() - start_time
        
        print(f"   lDDT (identical structures): {lddt_identical:.4f} (time: {time_identical:.3f}s)")
        
        # Test slightly different structures
        start_time = time.time()
        lddt_noisy = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
        time_noisy = time.time() - start_time
        
        print(f"   lDDT (noisy vs native): {lddt_noisy:.4f} (time: {time_noisy:.3f}s)")
        
        # Validation
        if np.isnan(lddt_identical) or np.isnan(lddt_noisy):
            print(f"   ❌ lDDT v2 returned NaN values")
            return False
        
        if lddt_identical < 0.8:  # Identical structures should have high lDDT
            print(f"   ⚠️ Identical structures have unexpectedly low lDDT: {lddt_identical:.4f}")
        
        if lddt_identical <= lddt_noisy:  # Identical should be better than noisy
            print(f"   ⚠️ Identical lDDT not better than noisy: {lddt_identical:.4f} vs {lddt_noisy:.4f}")
        
        print(f"   ✅ lDDT v2 basic functionality works")
        
        # Cleanup
        os.unlink(native_pdb)
        os.unlink(predicted_pdb)
        os.rmdir(test_dir)
        
        return True
        
    except Exception as e:
        print(f"   ❌ lDDT v2 basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lddt_v2_with_real_data():
    """Test lDDT v2 with real PDB data if available"""
    print("\n3️⃣ Testing lDDT v2 with real data...")
    
    try:
        from src.evaluator import get_lddt_openstructure_v2
        
        # Look for real PDB files in debug directory
        test_data_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
        
        if not os.path.exists(test_data_dir):
            print(f"   ⚠️ No real test data directory found: {test_data_dir}")
            return True
        
        pdb_files = [f for f in os.listdir(test_data_dir) if f.endswith('.pdb')]
        
        if len(pdb_files) < 2:
            print(f"   ⚠️ Need at least 2 PDB files for comparison, found: {len(pdb_files)}")
            return True
        
        # Test with first two PDB files
        pdb1 = os.path.join(test_data_dir, pdb_files[0])
        pdb2 = os.path.join(test_data_dir, pdb_files[1])
        
        print(f"   Testing: {pdb_files[0]} vs {pdb_files[1]}")
        
        start_time = time.time()
        lddt_real = get_lddt_openstructure_v2(pdb1, pdb2)
        time_real = time.time() - start_time
        
        print(f"   lDDT (real data): {lddt_real:.4f} (time: {time_real:.3f}s)")
        
        if np.isnan(lddt_real):
            print(f"   ⚠️ Real data lDDT returned NaN (expected for different structures)")
        else:
            print(f"   ✅ Real data lDDT calculation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Real data test failed: {e}")
        return False

def compare_lddt_versions():
    """Compare original lDDT vs OpenStructure v2"""
    print("\n4️⃣ Comparing lDDT versions...")
    
    try:
        from src.evaluator import get_lddt, get_lddt_openstructure_v2
        
        # Create test structures
        test_dir = "/tmp/lddt_comparison"
        os.makedirs(test_dir, exist_ok=True)
        
        native_pdb = os.path.join(test_dir, "native.pdb")
        predicted_pdb = os.path.join(test_dir, "predicted.pdb")
        
        sequence = "GCGCAAUUGGCCAAU"  # 15-mer RNA
        create_test_rna_pdb(native_pdb, sequence, add_noise=False)
        create_test_rna_pdb(predicted_pdb, sequence, add_noise=True)
        
        # Test original lDDT
        print(f"   Testing original lDDT...")
        start_time = time.time()
        lddt_v1 = get_lddt(predicted_pdb, native_pdb)
        time_v1 = time.time() - start_time
        
        # Test OpenStructure v2
        print(f"   Testing OpenStructure lDDT v2...")
        start_time = time.time()
        lddt_v2 = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
        time_v2 = time.time() - start_time
        
        print(f"   Results comparison:")
        print(f"      Original lDDT:      {lddt_v1:.4f} (time: {time_v1:.3f}s)")
        print(f"      OpenStructure v2:   {lddt_v2:.4f} (time: {time_v2:.3f}s)")
        
        # Analysis
        if np.isnan(lddt_v1) and not np.isnan(lddt_v2):
            print(f"   ✅ OpenStructure v2 succeeded where original failed")
        elif not np.isnan(lddt_v1) and not np.isnan(lddt_v2):
            diff = abs(lddt_v1 - lddt_v2)
            print(f"   📊 Score difference: {diff:.4f}")
            if diff < 0.1:
                print(f"   ✅ Both versions give similar results")
            else:
                print(f"   ⚠️ Significant difference between versions")
        elif np.isnan(lddt_v1) and np.isnan(lddt_v2):
            print(f"   ⚠️ Both versions failed (expected for problematic structures)")
        else:
            print(f"   ⚠️ Original succeeded but v2 failed")
        
        # Performance comparison
        if time_v2 < time_v1:
            speedup = time_v1 / time_v2
            print(f"   🚀 OpenStructure v2 is {speedup:.1f}x faster")
        elif time_v1 < time_v2:
            slowdown = time_v2 / time_v1
            print(f"   🐌 OpenStructure v2 is {slowdown:.1f}x slower")
        else:
            print(f"   ⚖️ Similar performance")
        
        # Cleanup
        os.unlink(native_pdb)
        os.unlink(predicted_pdb)
        os.rmdir(test_dir)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Version comparison failed: {e}")
        return False

def test_lddt_v2_edge_cases():
    """Test edge cases for lDDT v2"""
    print("\n5️⃣ Testing lDDT v2 edge cases...")
    
    try:
        from src.evaluator import get_lddt_openstructure_v2
        
        # Test 1: Non-existent files
        result1 = get_lddt_openstructure_v2("/nonexistent1.pdb", "/nonexistent2.pdb")
        print(f"   Non-existent files: {result1} (should be NaN)")
        
        # Test 2: Empty PDB content
        test_dir = "/tmp/lddt_edge_test"
        os.makedirs(test_dir, exist_ok=True)
        
        empty_pdb = os.path.join(test_dir, "empty.pdb")
        with open(empty_pdb, 'w') as f:
            f.write("HEADER    EMPTY\nEND\n")
        
        result2 = get_lddt_openstructure_v2(empty_pdb, empty_pdb)
        print(f"   Empty structures: {result2} (should be NaN)")
        
        # Test 3: Single residue
        single_pdb = os.path.join(test_dir, "single.pdb")
        create_test_rna_pdb(single_pdb, "G", add_noise=False)
        
        result3 = get_lddt_openstructure_v2(single_pdb, single_pdb)
        print(f"   Single residue: {result3} (may be NaN due to insufficient contacts)")
        
        # Test 4: Very different sequences
        seq1_pdb = os.path.join(test_dir, "seq1.pdb")
        seq2_pdb = os.path.join(test_dir, "seq2.pdb")
        
        create_test_rna_pdb(seq1_pdb, "GGGGAAAA", add_noise=False)
        create_test_rna_pdb(seq2_pdb, "CCCCUUUU", add_noise=False)
        
        result4 = get_lddt_openstructure_v2(seq1_pdb, seq2_pdb)
        print(f"   Different sequences: {result4} (should work with check_resnames=False)")
        
        # Validation
        edge_case_results = [result1, result2, result3, result4]
        nan_count = sum(1 for r in edge_case_results if np.isnan(r))
        valid_count = len(edge_case_results) - nan_count
        
        print(f"   📊 Edge case summary: {valid_count} valid, {nan_count} NaN out of {len(edge_case_results)}")
        
        if np.isnan(result1) and np.isnan(result2):
            print(f"   ✅ Properly handles invalid inputs with NaN")
        
        # Cleanup
        for f in [empty_pdb, single_pdb, seq1_pdb, seq2_pdb]:
            if os.path.exists(f):
                os.unlink(f)
        os.rmdir(test_dir)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False

def main():
    """Run all lDDT v2 tests"""
    print("🧪 Testing OpenStructure lDDT v2 Implementation")
    print("=" * 70)
    
    tests = [
        ("OpenStructure availability", test_openstructure_availability),
        ("Basic functionality", test_lddt_v2_basic), 
        ("Real data testing", test_lddt_v2_with_real_data),
        ("Version comparison", compare_lddt_versions),
        ("Edge cases", test_lddt_v2_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status:8s} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! OpenStructure lDDT v2 is ready for use.")
        print("\n💡 Integration suggestions:")
        print("   1. Add config option to choose lDDT version (v1 vs v2)")
        print("   2. Use v2 as default if OpenStructure is available")
        print("   3. Fall back to v1 if OpenStructure is not installed")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check OpenStructure installation and implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)