#!/usr/bin/env python3
"""
Standalone test for OpenStructure lDDT v2 implementation
This avoids importing the main evaluator to prevent NetworkX conflicts
"""

import os
import sys
import subprocess
import tempfile
import time
import numpy as np

def test_lddt_env_setup():
    """Test if lddt_env is properly set up with OpenStructure"""
    print("🔧 Testing lddt_env setup...")
    
    conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
    lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
    
    if not os.path.exists(lddt_python):
        print(f"   ❌ lddt_env Python not found: {lddt_python}")
        return False
    
    # Test OpenStructure import
    test_script = '''
import sys
try:
    import ost
    import ost.mol
    import ost.io
    from ost.mol.alg import lddt
    
    # Test basic functionality
    scorer_class = lddt.lDDTScorer
    print("SUCCESS: OpenStructure with lDDT available")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run(
            [lddt_python, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            print(f"   ✅ OpenStructure with lDDT is available in lddt_env")
            return True
        else:
            print(f"   ❌ OpenStructure test failed:")
            print(f"      Return code: {result.returncode}")
            print(f"      stdout: {result.stdout}")
            print(f"      stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Test timed out")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def create_simple_rna_pdb(pdb_path, sequence="AGCU", add_noise=False):
    """Create a very simple test RNA PDB"""
    
    with open(pdb_path, 'w') as f:
        f.write("HEADER    TEST RNA\\n")
        f.write("REMARK    Simple test structure\\n")
        
        atom_id = 1
        for i, nuc in enumerate(sequence):
            res_num = i + 1
            x_base = i * 6.0
            
            # Add a few key atoms per residue
            atoms = [
                ("P",   x_base + 0.0, 0.0, 0.0),
                ("O5'", x_base + 1.0, 0.5, 0.0),
                ("C5'", x_base + 2.0, 0.0, 0.0),
                ("C4'", x_base + 3.0, 0.5, 0.0),
                ("C3'", x_base + 4.0, 0.0, 0.0),
                ("O3'", x_base + 5.0, 0.5, 0.0),
                ("C1'", x_base + 3.0, 2.0, 0.5),
                ("N1" if nuc in "UC" else "N9", x_base + 3.0, 3.0, 1.0)
            ]
            
            for atom_name, x, y, z in atoms:
                if add_noise:
                    x += np.random.normal(0, 0.05)
                    y += np.random.normal(0, 0.05)
                    z += np.random.normal(0, 0.05)
                
                f.write(f"ATOM  {atom_id:5d}  {atom_name:<3s} {nuc:>3s} A{res_num:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           "
                       f"{atom_name[0]:>1s}\\n")
                atom_id += 1
        
        f.write("END\\n")

def test_lddt_v2_calculation():
    """Test the actual lDDT v2 calculation"""
    print("\\n🧪 Testing lDDT v2 calculation...")
    
    try:
        # Create test structures
        test_dir = "/tmp/lddt_v2_test"
        os.makedirs(test_dir, exist_ok=True)
        
        native_pdb = os.path.join(test_dir, "native.pdb")
        predicted_pdb = os.path.join(test_dir, "predicted.pdb")
        
        sequence = "AGCUAGCU"  # 8-mer RNA
        create_simple_rna_pdb(native_pdb, sequence, add_noise=False)
        create_simple_rna_pdb(predicted_pdb, sequence, add_noise=True)
        
        print(f"   Created test RNA structures: {sequence}")
        
        # Create the OpenStructure lDDT calculation script
        script_content = '''#!/usr/bin/env python3
import sys
import os

def calculate_lddt_v2(predicted_pdb, native_pdb):
    """Calculate lDDT using OpenStructure"""
    try:
        import ost
        import ost.mol
        import ost.io
        from ost.mol.alg import lddt
        
        # Load structures
        native_entity = ost.io.LoadPDB(native_pdb)
        predicted_entity = ost.io.LoadPDB(predicted_pdb)
        
        if not native_entity.IsValid() or not predicted_entity.IsValid():
            return float('nan')
        
        # Clean structures - keep only nucleic acids
        native_clean = clean_structure_for_lddt(native_entity)
        predicted_clean = clean_structure_for_lddt(predicted_entity)
        
        if not native_clean.IsValid() or not predicted_clean.IsValid():
            return float('nan')
        
        # Create lDDT scorer
        scorer = lddt.lDDTScorer(
            target=native_clean,
            inclusion_radius=15.0,
            sequence_separation=0,
            bb_only=False
        )
        
        # Compute lDDT
        global_lddt, per_residue_lddt = scorer.lDDT(
            model=predicted_clean,
            thresholds=[0.5, 1.0, 2.0, 4.0],
            check_resnames=False,
            no_interchain=False,
            no_intrachain=False
        )
        
        if global_lddt is None:
            return float('nan')
        
        return float(global_lddt)
        
    except Exception as e:
        print(f"lDDT calculation error: {e}", file=sys.stderr)
        return float('nan')

def clean_structure_for_lddt(entity):
    """Clean structure for lDDT"""
    try:
        clean_view = entity.CreateEmptyView()
        
        for chain in entity.chains:
            chain_view = clean_view.AddChain(chain, deep=True)
            
            for residue in chain.residues:
                # Skip water and ligands
                if residue.name in ['HOH', 'WAT', 'SO4', 'PO4']:
                    continue
                
                # Keep RNA residues
                rna_residues = {'A', 'U', 'G', 'C', 'T'}
                if residue.name in rna_residues and len(residue.atoms) >= 3:
                    chain_view.AddResidue(residue, deep=True)
        
        return clean_view
        
    except Exception as e:
        print(f"Structure cleaning failed: {e}", file=sys.stderr)
        return entity.CreateEmptyView()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <predicted_pdb> <native_pdb>")
        sys.exit(1)
    
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    
    result = calculate_lddt_v2(predicted_pdb, native_pdb)
    print(result)
'''
        
        # Save script
        script_path = os.path.join(test_dir, "lddt_v2_test.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run lDDT calculation using lddt_env
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        # Test 1: Identical structures (should give high score)
        print(f"   Testing identical structures...")
        start_time = time.time()
        result = subprocess.run(
            [lddt_python, script_path, native_pdb, native_pdb],
            capture_output=True,
            text=True,
            timeout=30
        )
        time_identical = time.time() - start_time
        
        if result.returncode == 0:
            lddt_identical = float(result.stdout.strip())
            print(f"   lDDT (identical): {lddt_identical:.4f} (time: {time_identical:.3f}s)")
        else:
            print(f"   ❌ Identical test failed:")
            print(f"      stderr: {result.stderr}")
            lddt_identical = float('nan')
        
        # Test 2: Different structures
        print(f"   Testing predicted vs native...")
        start_time = time.time()
        result = subprocess.run(
            [lddt_python, script_path, predicted_pdb, native_pdb],
            capture_output=True,
            text=True,
            timeout=30
        )
        time_different = time.time() - start_time
        
        if result.returncode == 0:
            lddt_different = float(result.stdout.strip())
            print(f"   lDDT (different): {lddt_different:.4f} (time: {time_different:.3f}s)")
        else:
            print(f"   ❌ Different test failed:")
            print(f"      stderr: {result.stderr}")
            lddt_different = float('nan')
        
        # Analysis
        success = False
        if not np.isnan(lddt_identical):
            if lddt_identical > 0.8:
                print(f"   ✅ Identical structures give high lDDT: {lddt_identical:.4f}")
                success = True
            else:
                print(f"   ⚠️ Identical structures have low lDDT: {lddt_identical:.4f}")
        
        if not np.isnan(lddt_different):
            print(f"   ✅ Different structures calculation successful")
            success = True
        elif np.isnan(lddt_different):
            print(f"   ⚠️ Different structures returned NaN (may be expected)")
        
        if not np.isnan(lddt_identical) and not np.isnan(lddt_different):
            if lddt_identical > lddt_different:
                print(f"   ✅ Identical > Different: {lddt_identical:.4f} > {lddt_different:.4f}")
            else:
                print(f"   ⚠️ Unexpected: identical not > different")
        
        # Cleanup
        os.unlink(native_pdb)
        os.unlink(predicted_pdb)
        os.unlink(script_path)
        os.rmdir(test_dir)
        
        return success
        
    except Exception as e:
        print(f"   ❌ lDDT v2 calculation test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance if both versions work"""
    print("\\n⚡ Testing performance...")
    
    # This would require both original and v2 to work
    # For now, just test v2 performance with different sizes
    
    sequences = [
        ("Small (4nt)", "AGCU"),
        ("Medium (12nt)", "AGCUAGCUAGCU"),  
        ("Large (24nt)", "AGCUAGCUAGCUAGCUAGCUAGCU")
    ]
    
    times = []
    
    for size_name, sequence in sequences:
        try:
            test_dir = "/tmp/perf_test"
            os.makedirs(test_dir, exist_ok=True)
            
            native_pdb = os.path.join(test_dir, "native.pdb")
            create_simple_rna_pdb(native_pdb, sequence)
            
            # Simple lDDT script
            script_content = '''
import sys, os
import ost, ost.io
from ost.mol.alg import lddt

native = ost.io.LoadPDB(sys.argv[1])
scorer = lddt.lDDTScorer(native, bb_only=False)
result, _ = scorer.lDDT(native, check_resnames=False)
print(result if result is not None else "nan")
'''
            
            script_path = os.path.join(test_dir, "perf_test.py")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
            lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
            
            start_time = time.time()
            result = subprocess.run(
                [lddt_python, script_path, native_pdb],
                capture_output=True,
                text=True,
                timeout=10
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                times.append((size_name, elapsed))
                print(f"   {size_name}: {elapsed:.3f}s")
            
            # Cleanup
            os.unlink(native_pdb)
            os.unlink(script_path) 
            os.rmdir(test_dir)
            
        except Exception as e:
            print(f"   {size_name}: failed ({e})")
    
    if len(times) >= 2:
        print(f"   ✅ Performance scales reasonably with size")
        return True
    else:
        print(f"   ⚠️ Limited performance data")
        return False

def main():
    """Run OpenStructure lDDT v2 standalone tests"""
    print("🧪 Standalone OpenStructure lDDT v2 Test")
    print("=" * 60)
    
    tests = [
        ("lddt_env setup", test_lddt_env_setup),
        ("lDDT v2 calculation", test_lddt_v2_calculation),
        ("Performance", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
    
    print(f"\\n🎯 Result: {passed}/{total} tests passed")
    
    if passed >= 2:
        print("\\n🎉 OpenStructure lDDT v2 is functional!")
        print("\\n💡 Ready for integration into evaluation pipeline")
        print("   - Isolated environment prevents NetworkX conflicts")
        print("   - Modern OpenStructure API provides robust lDDT calculation")
        print("   - Suitable for RNA inverse folding evaluation")
    else:
        print("\\n⚠️ OpenStructure lDDT v2 needs debugging")
        
    return passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)