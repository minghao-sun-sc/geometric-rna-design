#!/usr/bin/env python3
"""
Test lDDT with identical structures to verify the isolated environment fix
"""

import sys
import os
import tempfile

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def create_test_pdbs():
    """Create identical test PDB structures"""
    test_pdb_content = """ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      3  P     C A   2       2.000   2.000   2.000  1.00  0.00           P
ATOM      4  C4'   C A   2       3.000   3.000   3.000  1.00  0.00           C
ATOM      5  P     A A   3       4.000   4.000   4.000  1.00  0.00           P
ATOM      6  C4'   A A   3       5.000   5.000   5.000  1.00  0.00           C
END
"""
    
    tmpdir = tempfile.mkdtemp()
    model_pdb = os.path.join(tmpdir, "model.pdb")
    native_pdb = os.path.join(tmpdir, "native.pdb")
    
    with open(model_pdb, 'w') as f:
        f.write(test_pdb_content)
    with open(native_pdb, 'w') as f:
        f.write(test_pdb_content)
    
    return model_pdb, native_pdb, tmpdir

def test_lddt_simple():
    print("🔬 Testing lDDT with Identical Simple Structures")
    print("=" * 60)
    
    # Create identical test structures
    model_pdb, native_pdb, tmpdir = create_test_pdbs()
    
    print(f"✅ Created test structures:")
    print(f"   Model: {model_pdb}")
    print(f"   Native: {native_pdb}")
    
    # Import the updated evaluator functions
    from src.evaluator import get_lddt
    
    # Test the lDDT function
    print(f"\n🧪 Testing get_lddt function...")
    lddt_score = get_lddt(model_pdb, native_pdb)
    
    print(f"\n📊 Results:")
    if lddt_score > 0:
        print(f"✅ lDDT score: {lddt_score:.4f}")
        print(f"✅ lDDT calculation with isolated environment SUCCESSFUL!")
        
        # Clean up
        import shutil
        shutil.rmtree(tmpdir)
        return True
    else:
        print(f"❌ lDDT failed (score: {lddt_score})")
        
        # Clean up
        import shutil
        shutil.rmtree(tmpdir)
        return False

if __name__ == "__main__":
    success = test_lddt_simple()
    sys.exit(0 if success else 1)
