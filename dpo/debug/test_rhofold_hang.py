#!/usr/bin/env python3
"""
Debug script to isolate RhoFold hanging issue.
Tests RhoFold with a simple sequence to see if it's hanging in general or specific to our data.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import tempfile
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def test_rhofold_basic():
    """Test RhoFold with a simple RNA sequence"""
    print("Testing RhoFold with simple sequence...")
    
    # Import RhoFold
    from tools.rhofold.rf import RhoFold
    from tools.rhofold.config import rhofold_config
    import torch
    
    # Initialize RhoFold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rhofold = RhoFold(rhofold_config, device)
    rhofold_path = os.path.join("/mnt/rna01/smh/projects/ribopo", "tools/rhofold/model_20221010_params.pt")
    print(f"Loading RhoFold checkpoint: {rhofold_path}")
    rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
    rhofold = rhofold.to(device)
    rhofold.eval()
    
    # Simple test sequence (16-mer)
    test_seq = "GCGCAAAAAGCGCUGU"
    print(f"Test sequence: {test_seq} (length: {len(test_seq)})")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        fasta_path = os.path.join(temp_dir, "test.fasta")
        pdb_path = os.path.join(temp_dir, "test.pdb")
        
        # Write FASTA
        seq_record = SeqRecord(Seq(test_seq), id="test", description="test sequence")
        SeqIO.write(seq_record, fasta_path, "fasta")
        
        print(f"Running RhoFold prediction...")
        print(f"  Input: {fasta_path}")
        print(f"  Output: {pdb_path}")
        
        try:
            # This is where it might hang
            coords, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=False)
            print(f"✅ RhoFold prediction completed successfully!")
            print(f"  Coords shape: {coords.shape if coords is not None else 'None'}")
            print(f"  pLDDT shape: {plddt.shape if plddt is not None else 'None'}")
            print(f"  Output PDB size: {os.path.getsize(pdb_path) if os.path.exists(pdb_path) else 'N/A'} bytes")
            
        except Exception as e:
            print(f"❌ RhoFold prediction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rhofold_basic()