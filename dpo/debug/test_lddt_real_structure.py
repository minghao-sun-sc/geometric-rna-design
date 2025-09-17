#!/usr/bin/env python3
"""
Test lDDT with real RNA structure using RhoFold prediction
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import sys
import os
import tempfile
import torch

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def test_lddt_real_structure():
    print("🔬 Testing lDDT with Real RNA Structure")
    print("=" * 50)
    
    # Test sequence and structure
    test_id = "3SLQ_1_A"
    test_sequence = "GGGCUCAGUACGGUGGUAUACAGCGCCUCUGAGUCAGCCCUUCAGGCAACUGGGGGAACUGAGGCC"
    native_pdb = os.path.join(PROJECT_PATH, "data/raw", f"{test_id}.pdb")
    
    if not os.path.exists(native_pdb):
        print(f"❌ Native PDB not found: {native_pdb}")
        return False
    
    print(f"✅ Native PDB found: {native_pdb}")
    
    # Create output directory
    output_dir = tempfile.mkdtemp()
    print(f"📁 Working directory: {output_dir}")
    
    try:
        # Import RhoFold
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        
        print(f"\n🧬 Running RhoFold prediction...")
        
        # Initialize RhoFold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        
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
        
        # Test lDDT calculation
        print(f"\n🔬 Testing lDDT calculation...")
        
        from src.evaluator import get_lddt
        
        lddt_score = get_lddt(pdb_path, native_pdb)
        
        print(f"\n📊 Results:")
        if lddt_score > 0:
            print(f"✅ lDDT score: {lddt_score:.4f}")
            print(f"✅ Real structure lDDT calculation SUCCESSFUL!")
            
            # Clean up
            import shutil
            shutil.rmtree(output_dir)
            return True
        else:
            print(f"❌ lDDT failed (score: {lddt_score})")
            print(f"   This may be due to sequence differences between prediction and native")
            
            # Clean up
            import shutil
            shutil.rmtree(output_dir)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        import shutil
        shutil.rmtree(output_dir)
        return False

if __name__ == "__main__":
    success = test_lddt_real_structure()
    sys.exit(0 if success else 1)
