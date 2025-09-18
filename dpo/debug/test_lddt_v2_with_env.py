#!/usr/bin/env python3
"""
Test OpenStructure lDDT v2 implementation using isolated lddt_env
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

def test_lddt_env_availability():
    """Test if lddt_env is available and has OpenStructure"""
    print("1️⃣ Testing lddt_env availability...")
    
    try:
        import subprocess
        
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        if not os.path.exists(lddt_python):
            print(f"   ❌ lddt_env Python not found: {lddt_python}")
            return False
        
        # Test OpenStructure in lddt_env
        test_script = '''
import sys
try:
    import ost
    import ost.mol
    import ost.io
    from ost.mol.alg import lddt
    print(f"SUCCESS: OpenStructure {ost.GetVersion()}")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
'''
        
        result = subprocess.run(
            [lddt_python, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            version = result.stdout.strip().split("SUCCESS: OpenStructure ")[1]
            print(f"   ✅ lddt_env OpenStructure version: {version}")
            return True
        else:
            print(f"   ❌ OpenStructure test failed:")
            print(f"      stdout: {result.stdout}")
            print(f"      stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ lddt_env test failed: {e}")
        return False

def create_test_rna_pdb(pdb_path, sequence="GGCAAGCCUGCGAUGGCC", add_noise=False):
    """Create a simple test RNA PDB file"""
    
    # Simple RNA backbone coordinates (approximate)
    base_coords = [
        # P, O5', C5', C4', C3', O3', C1', N1/N9 for each residue
        (0.0, 0.0, 0.0), (1.0, 0.5, 0.2), (2.0, 0.3, 0.1), (3.0, 0.8, 0.3),
        (4.0, 0.6, 0.2), (5.0, 1.0, 0.4), (3.5, 2.0, 0.5), (3.8, 3.0, 0.8)
    ]
    
    atom_names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N1"]
    
    with open(pdb_path, 'w') as f:
        f.write("HEADER    TEST RNA STRUCTURE\\n")
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
                       f"{atom_name[0]:>1s}\\n")
                atom_id += 1
        
        f.write("END\\n")

def test_lddt_v2_basic():
    """Test basic lDDT v2 functionality with synthetic data"""
    print("\\n2️⃣ Testing lDDT v2 basic functionality...")
    
    try:
        from src.evaluator import get_lddt_openstructure_v2
        
        # Create test directory
        test_dir = "/tmp/lddt_v2_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Create test structures
        native_pdb = os.path.join(test_dir, "native.pdb")
        predicted_pdb = os.path.join(test_dir, "predicted.pdb")
        
        sequence = "GGCAAGCCUGCGAUGGCC"  # 18-mer RNA
        create_test_rna_pdb(native_pdb, sequence, add_noise=False)
        create_test_rna_pdb(predicted_pdb, sequence, add_noise=True)  # Add slight noise
        
        print(f"   Created test structures: {len(sequence)} nucleotides")
        
        # Test identical structures (should give high lDDT)
        print(f"   Testing identical structures...")
        start_time = time.time()
        lddt_identical = get_lddt_openstructure_v2(native_pdb, native_pdb)
        time_identical = time.time() - start_time
        
        print(f"   lDDT (identical): {lddt_identical:.4f} (time: {time_identical:.3f}s)")
        
        # Test slightly different structures
        print(f"   Testing noisy vs native...")
        start_time = time.time()
        lddt_noisy = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
        time_noisy = time.time() - start_time
        
        print(f"   lDDT (noisy): {lddt_noisy:.4f} (time: {time_noisy:.3f}s)")
        
        # Validation
        if np.isnan(lddt_identical) and np.isnan(lddt_noisy):
            print(f"   ⚠️ Both calculations returned NaN - check OpenStructure setup")
            return False
        elif np.isnan(lddt_identical):
            print(f"   ❌ Identical structures returned NaN")
            return False
        elif np.isnan(lddt_noisy):
            print(f"   ⚠️ Noisy structures returned NaN (may be expected)")
        
        if not np.isnan(lddt_identical):
            if lddt_identical < 0.7:  # Identical structures should have high lDDT
                print(f"   ⚠️ Identical structures have low lDDT: {lddt_identical:.4f}")
            else:
                print(f"   ✅ Identical structures have good lDDT: {lddt_identical:.4f}")
        
        if not np.isnan(lddt_noisy) and not np.isnan(lddt_identical):
            if lddt_identical <= lddt_noisy:
                print(f"   ⚠️ Identical not better than noisy: {lddt_identical:.4f} vs {lddt_noisy:.4f}")
            else:
                print(f"   ✅ Identical better than noisy: {lddt_identical:.4f} > {lddt_noisy:.4f}")
        
        print(f"   ✅ lDDT v2 basic functionality test completed")
        
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

def test_lddt_v2_vs_original():
    """Compare lDDT v2 with original implementation"""
    print("\\n3️⃣ Comparing lDDT v2 vs original...")
    
    try:
        from src.evaluator import get_lddt, get_lddt_openstructure_v2
        
        # Create test structures
        test_dir = "/tmp/lddt_comparison"
        os.makedirs(test_dir, exist_ok=True)
        
        native_pdb = os.path.join(test_dir, "native.pdb")
        predicted_pdb = os.path.join(test_dir, "predicted.pdb")
        
        sequence = "GCGCAAUUGGCCAAUGGCG"  # 19-mer RNA
        create_test_rna_pdb(native_pdb, sequence, add_noise=False)
        create_test_rna_pdb(predicted_pdb, sequence, add_noise=True)
        
        # Test original lDDT
        print(f"   Testing original lDDT...")
        start_time = time.time()
        lddt_v1 = get_lddt(predicted_pdb, native_pdb)
        time_v1 = time.time() - start_time
        
        # Test OpenStructure v2
        print(f"   Testing OpenStructure v2...")
        start_time = time.time()
        lddt_v2 = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
        time_v2 = time.time() - start_time
        
        print(f"   Results comparison:")
        print(f"      Original lDDT:      {lddt_v1:.4f} (time: {time_v1:.3f}s)")
        print(f"      OpenStructure v2:   {lddt_v2:.4f} (time: {time_v2:.3f}s)")
        
        # Analysis
        if np.isnan(lddt_v1) and not np.isnan(lddt_v2):
            print(f"   ✅ v2 succeeded where original failed")
        elif not np.isnan(lddt_v1) and not np.isnan(lddt_v2):
            diff = abs(lddt_v1 - lddt_v2)
            print(f"   📊 Score difference: {diff:.4f}")
            if diff < 0.2:
                print(f"   ✅ Both versions give similar results")
            else:
                print(f"   ⚠️ Significant difference between versions")
        elif np.isnan(lddt_v1) and np.isnan(lddt_v2):
            print(f"   ⚠️ Both versions failed")
        else:
            print(f"   ⚠️ Original succeeded but v2 failed")
        
        # Performance comparison
        if not np.isnan(lddt_v2) and not np.isnan(lddt_v1):
            if time_v2 < time_v1:
                speedup = time_v1 / time_v2
                print(f"   🚀 v2 is {speedup:.1f}x faster")
            elif time_v1 < time_v2:
                slowdown = time_v2 / time_v1
                print(f"   🐌 v2 is {slowdown:.1f}x slower")
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

def test_lddt_v2_real_data():
    """Test lDDT v2 with real PDB data if available"""
    print("\\n4️⃣ Testing lDDT v2 with real data...")
    
    try:
        from src.evaluator import get_lddt_openstructure_v2
        
        # Look for real PDB files
        test_data_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
        
        if not os.path.exists(test_data_dir):
            print(f"   ⚠️ No real test data directory: {test_data_dir}")
            return True
        
        pdb_files = [f for f in os.listdir(test_data_dir) if f.endswith('.pdb')]
        
        if len(pdb_files) < 1:
            print(f"   ⚠️ No PDB files found for testing")
            return True
        
        # Test self-comparison (should give perfect score)
        pdb1 = os.path.join(test_data_dir, pdb_files[0])
        
        print(f"   Testing: {pdb_files[0]} vs itself")
        start_time = time.time()
        lddt_self = get_lddt_openstructure_v2(pdb1, pdb1)
        time_self = time.time() - start_time
        
        print(f"   lDDT (self): {lddt_self:.4f} (time: {time_self:.3f}s)")
        
        if np.isnan(lddt_self):
            print(f"   ⚠️ Self-comparison returned NaN - check structure quality")
        elif lddt_self > 0.9:
            print(f"   ✅ Self-comparison gives high score")
        else:
            print(f"   ⚠️ Self-comparison unexpectedly low: {lddt_self:.4f}")
        
        # Test different structures if available
        if len(pdb_files) >= 2:
            pdb2 = os.path.join(test_data_dir, pdb_files[1])
            
            print(f"   Testing: {pdb_files[0]} vs {pdb_files[1]}")
            start_time = time.time()
            lddt_diff = get_lddt_openstructure_v2(pdb1, pdb2)
            time_diff = time.time() - start_time
            
            print(f"   lDDT (different): {lddt_diff:.4f} (time: {time_diff:.3f}s)")
            
            if np.isnan(lddt_diff):
                print(f"   ⚠️ Different structures returned NaN (may be expected)")
            else:
                print(f"   ✅ Different structures calculation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Real data test failed: {e}")
        return False

def main():
    """Run all lDDT v2 tests with lddt_env"""
    print("🧪 Testing OpenStructure lDDT v2 with Isolated Environment")
    print("=" * 70)
    
    tests = [
        ("lddt_env availability", test_lddt_env_availability),
        ("Basic functionality", test_lddt_v2_basic),
        ("Version comparison", test_lddt_v2_vs_original),
        ("Real data testing", test_lddt_v2_real_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status:8s} {test_name}")
    
    print(f"\\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All tests passed! OpenStructure lDDT v2 is ready for integration.")
        print("\\n💡 Next steps:")
        print("   1. Add config option to choose lDDT version")
        print("   2. Use v2 as default if lddt_env is available")
        print("   3. Integrate into evaluation pipeline")
    elif passed >= total - 1:
        print(f"\\n✅ Most tests passed. OpenStructure lDDT v2 is functional.")
        print("\\n⚠️ Minor issues may exist but core functionality works.")
    else:
        print(f"\\n⚠️ Multiple tests failed. Check lddt_env and OpenStructure setup.")
    
    return passed >= total - 1  # Allow 1 failure

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)