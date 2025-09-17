#!/usr/bin/env python3
"""
Direct test of lDDT tool to verify OST installation works
"""
import os
import sys
import subprocess
import tempfile

# Test if we can import OST directly
try:
    import ost
    print("✅ OST (OpenStructure) successfully imported!")
    print(f"   OST version: {ost.__version__ if hasattr(ost, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ Failed to import OST: {e}")
    sys.exit(1)

# Test the lDDT script directly
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')

print(f"\n🔍 Testing lDDT script at: {lddt_script_path}")

if not os.path.exists(lddt_script_path):
    print(f"❌ lDDT script not found!")
    sys.exit(1)

# Create minimal test PDB files
test_pdb_content = """ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      3  P     C A   2       2.000   2.000   2.000  1.00  0.00           P
ATOM      4  C4'   C A   2       3.000   3.000   3.000  1.00  0.00           C
END
"""

with tempfile.TemporaryDirectory() as tmpdir:
    model_pdb = os.path.join(tmpdir, "model.pdb")
    native_pdb = os.path.join(tmpdir, "native.pdb")
    
    with open(model_pdb, 'w') as f:
        f.write(test_pdb_content)
    with open(native_pdb, 'w') as f:
        f.write(test_pdb_content)
    
    print(f"   Created test PDB files")
    
    # Test the lDDT script
    lddt_script_dir = os.path.dirname(lddt_script_path)
    chain_mapping = '{"A":"A"}'
    
    command = [
        sys.executable,
        lddt_script_path,
        model_pdb,
        native_pdb,
        chain_mapping
    ]
    
    print(f"   Running command: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
    
    print(f"\n📊 Results:")
    print(f"   Return code: {result.returncode}")
    print(f"   STDOUT: '{result.stdout.strip()}'")
    if result.stderr.strip():
        print(f"   STDERR: '{result.stderr.strip()}'")
    
    if result.returncode == 0 and result.stdout.strip():
        try:
            lddt_score = float(result.stdout.strip())
            print(f"\n✅ lDDT calculation successful! Score: {lddt_score:.4f}")
        except ValueError:
            print(f"\n⚠️ lDDT ran but output couldn't be parsed as float")
    else:
        print(f"\n❌ lDDT calculation failed")

print("\n🎯 Test complete!")