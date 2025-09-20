#!/usr/bin/env python3
"""
Test the specific sequence that gave 0.0 clash score.
"""

import os
import sys
import numpy as np
sys.path.append('/mnt/rna01/smh/projects/ribopo')
sys.path.append('/mnt/rna01/smh/projects/ribopo/dpo')

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from src.evaluator import get_clash_score_phenix
from tools.rhofold.rf import RhoFold
from tools.rhofold.config import rhofold_config
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile

def debug_exact_sequence():
    """Test the exact sequence that gave suspicious 0.0 clash score."""
    # This was the sequence in our original test
    original_sequence = "GGGGCCCCUUUUAAAA"
    
    print(f"🔬 Testing original problematic sequence: {original_sequence}")
    
    try:
        # Initialize RhoFold
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = "/mnt/rna01/smh/projects/ribopo/tools/rhofold/model_20221010_params.pt"
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=device)['model'])
        rhofold = rhofold.to(device)
        rhofold.eval()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temp dir: {temp_dir}")
            
            # Create FASTA file
            fasta_path = os.path.join(temp_dir, "test.fasta")
            pdb_path = os.path.join(temp_dir, "test.pdb")
            
            record = SeqRecord(Seq(original_sequence), id="test", description="Original test")
            SeqIO.write(record, fasta_path, "fasta")
            
            # Run RhoFold with relaxation
            print("Running RhoFold with use_relax=True...")
            _, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=True, relax_steps=100)
            
            # Check files
            files = os.listdir(temp_dir)
            print(f"Files created: {files}")
            
            unrelaxed_path = f"{pdb_path[:-4]}_unrelaxed.pdb"
            relaxed_path = pdb_path
            
            # Calculate clash scores with detailed debugging
            phenix_wrapper = "/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh"
            
            print("\n🔍 Debugging clash score calculation...")
            
            # Test pre-relax
            if os.path.exists(unrelaxed_path):
                print(f"Pre-relax file exists: {unrelaxed_path}")
                try:
                    import subprocess
                    # Run Phenix directly and capture full output
                    result = subprocess.run(
                        [phenix_wrapper, "phenix.molprobity", unrelaxed_path],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    print("Pre-relax Phenix output (relevant lines):")
                    for line in result.stdout.split('\n'):
                        if 'clashscore' in line.lower() or 'clash' in line.lower():
                            print(f"  {line}")
                    
                    clash_pre = get_clash_score_phenix(unrelaxed_path, phenix_wrapper)
                    print(f"Pre-relax clash score: {clash_pre}")
                except Exception as e:
                    print(f"Pre-relax clash failed: {e}")
            
            # Test post-relax  
            if os.path.exists(relaxed_path):
                print(f"\nPost-relax file exists: {relaxed_path}")
                try:
                    import subprocess
                    # Run Phenix directly and capture full output
                    result = subprocess.run(
                        [phenix_wrapper, "phenix.molprobity", relaxed_path],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    print("Post-relax Phenix output (relevant lines):")
                    for line in result.stdout.split('\n'):
                        if 'clashscore' in line.lower() or 'clash' in line.lower():
                            print(f"  {line}")
                    
                    clash_post = get_clash_score_phenix(relaxed_path, phenix_wrapper)
                    print(f"Post-relax clash score: {clash_post}")
                    
                    if clash_post == 0.0:
                        print("\n⚠️ FOUND THE 0.0 CLASH SCORE!")
                        print("Investigating potential causes...")
                        
                        # Check if file is valid PDB
                        with open(relaxed_path, 'r') as f:
                            content = f.read()
                            if 'ATOM' not in content:
                                print("❌ No ATOM records in relaxed PDB!")
                            elif len([l for l in content.split('\n') if l.startswith('ATOM')]) < 10:
                                print("❌ Too few ATOM records in relaxed PDB!")
                            else:
                                print("✅ PDB appears to have valid ATOM records")
                                
                        # Check if Phenix is actually finding 0 clashes vs returning error
                        if 'clashscore = 0' in result.stdout.lower() or 'no clashes' in result.stdout.lower():
                            print("📋 Phenix explicitly reports 0 clashes - structure is clash-free")
                        else:
                            print("🔍 Need to check Phenix parsing logic")
                            
                except Exception as e:
                    print(f"Post-relax clash failed: {e}")
                    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_exact_sequence()