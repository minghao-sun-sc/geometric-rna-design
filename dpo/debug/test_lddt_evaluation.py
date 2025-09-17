#!/usr/bin/env python3
"""
Test lDDT evaluation with the fixed OST installation.
This script tests lDDT without importing the problematic biotite/networkx modules.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import tempfile
import subprocess
import torch
import numpy as np
from pathlib import Path

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
DATA_PATH = os.path.join(PROJECT_PATH, "data")

def test_lddt_with_rhofold():
    """Test lDDT calculation on a RhoFold-predicted structure."""
    print("🔬 Testing lDDT with RhoFold prediction")
    print("=" * 50)
    
    # Test sequence and structure
    test_id = "3SLQ_1_A"
    test_sequence = "GGGCUCAGUACGGUGGUAUACAGCGCCUCUGAGUCAGCCCUUCAGGCAACUGGGGGAACUGAGGCC"
    native_pdb = os.path.join(DATA_PATH, "raw", f"{test_id}.pdb")
    
    if not os.path.exists(native_pdb):
        print(f"❌ Native PDB not found: {native_pdb}")
        return False
    
    print(f"✅ Native PDB found: {native_pdb}")
    
    # Create output directory
    output_dir = "dpo/debug/lddt_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import RhoFold
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        
        print("\n🧬 Running RhoFold prediction...")
        
        # Initialize RhoFold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        
        print(f"   Loading model from: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=device)['model'])
        rhofold.to(device)
        rhofold.eval()
        
        # Create FASTA file
        fasta_path = os.path.join(output_dir, "test_sequence.fasta")
        pdb_path = os.path.join(output_dir, "predicted_structure.pdb")
        
        seq_record = SeqRecord(Seq(test_sequence), id="test_seq", description="Test sequence")
        SeqIO.write(seq_record, fasta_path, "fasta")
        
        # Predict structure
        coords, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=False)
        
        print(f"✅ RhoFold prediction complete")
        print(f"   Mean pLDDT: {plddt.mean():.3f}")
        print(f"   Output PDB: {pdb_path}")
        
        # Now test lDDT calculation
        print("\n🔬 Testing lDDT calculation...")
        
        # Import the lDDT function
        from src.evaluator import get_lddt
        
        # Calculate lDDT
        lddt_score = get_lddt(pdb_path, native_pdb)
        
        print(f"\n📊 Results:")
        if lddt_score > 0:
            print(f"✅ lDDT score: {lddt_score:.4f}")
            return True
        else:
            print(f"❌ lDDT calculation failed (returned {lddt_score})")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lddt_direct():
    """Direct test of lDDT tool."""
    print("\n🔍 Direct lDDT tool test")
    print("=" * 50)
    
    lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
    
    # Create minimal test PDBs
    test_pdb = """ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      3  P     C A   2       2.000   2.000   2.000  1.00  0.00           P
ATOM      4  C4'   C A   2       3.000   3.000   3.000  1.00  0.00           C
END
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_pdb = os.path.join(tmpdir, "model.pdb")
        native_pdb = os.path.join(tmpdir, "native.pdb")
        
        with open(model_pdb, 'w') as f:
            f.write(test_pdb)
        with open(native_pdb, 'w') as f:
            f.write(test_pdb)
        
        lddt_script_dir = os.path.dirname(lddt_script_path)
        chain_mapping = '{"A":"A"}'
        
        command = [
            sys.executable,
            lddt_script_path,
            model_pdb,
            native_pdb,
            chain_mapping
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
        
        if result.returncode == 0 and result.stdout.strip():
            lddt_score = float(result.stdout.strip())
            print(f"✅ Direct lDDT test successful: {lddt_score:.4f}")
            return True
        else:
            print(f"❌ Direct lDDT test failed")
            print(f"   Return code: {result.returncode}")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False

if __name__ == "__main__":
    print("🚀 lDDT Evaluation Test Suite")
    print("=" * 50)
    
    # First test direct lDDT tool
    direct_success = test_lddt_direct()
    
    # Then test with RhoFold
    rhofold_success = test_lddt_with_rhofold()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print(f"   Direct lDDT tool: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"   RhoFold + lDDT: {'✅ PASSED' if rhofold_success else '❌ FAILED'}")
    
    if direct_success and rhofold_success:
        print("\n✅ All tests passed! lDDT evaluation is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)