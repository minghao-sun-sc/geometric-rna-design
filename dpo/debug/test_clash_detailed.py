#!/usr/bin/env python3
"""
Detailed test to investigate clash score calculation issues.
Tests with different sequences and checks file existence.
"""

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import os
import sys
import numpy as np
sys.path.append('/mnt/rna01/smh/projects/ribopo')

from src.evaluator import get_clash_score_phenix
from tools.rhofold.rf import RhoFold
from tools.rhofold.config import rhofold_config
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile

def test_clash_on_existing_structure():
    """Test clash score calculation on a known structure."""
    print("\n🔬 Test 1: Clash score on existing structure")
    
    # Use an example structure we know exists
    test_pdb = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data/usalign_rna_example_1.pdb"
    phenix_wrapper = "/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh"
    
    if os.path.exists(test_pdb):
        print(f"Testing on: {test_pdb}")
        try:
            clash = get_clash_score_phenix(test_pdb, phenix_wrapper)
            print(f"✅ Clash score: {clash:.2f}")
            if clash == 0.0:
                print("⚠️ WARNING: Clash score is exactly 0.0, which is suspiciously perfect!")
        except Exception as e:
            print(f"❌ Error calculating clash score: {e}")
    else:
        print(f"❌ Test PDB not found: {test_pdb}")

def test_rhofold_with_relax(sequence, seq_name="test"):
    """Test RhoFold prediction with relaxation and check both PDB files."""
    print(f"\n🔬 Test: RhoFold with relaxation for {seq_name}")
    print(f"   Sequence: {sequence}")
    
    try:
        # Initialize RhoFold
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = "/mnt/rna01/smh/projects/ribopo/tools/rhofold/model_20221010_params.pt"
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=device)['model'])
        rhofold = rhofold.to(device)
        rhofold.eval()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   Temp dir: {temp_dir}")
            
            # Create FASTA file
            fasta_path = os.path.join(temp_dir, f"{seq_name}.fasta")
            pdb_path = os.path.join(temp_dir, f"{seq_name}.pdb")
            
            record = SeqRecord(Seq(sequence), id=seq_name, description="Test RNA")
            SeqIO.write(record, fasta_path, "fasta")
            
            # Run RhoFold with relaxation
            print("   Running RhoFold with use_relax=True...")
            _, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=True, relax_steps=100)
            print(f"   pLDDT: {np.mean(plddt):.2f}")
            
            # Check what files were created
            files_created = os.listdir(temp_dir)
            print(f"   Files created: {files_created}")
            
            # Expected files when use_relax=True
            unrelaxed_path = f"{pdb_path[:-4]}_unrelaxed.pdb"
            relaxed_path = pdb_path
            
            # Check file existence and sizes
            if os.path.exists(unrelaxed_path):
                size_unrelaxed = os.path.getsize(unrelaxed_path)
                print(f"   ✅ Unrelaxed PDB exists: {os.path.basename(unrelaxed_path)} ({size_unrelaxed} bytes)")
                
                # Check if file has content
                with open(unrelaxed_path, 'r') as f:
                    lines = f.readlines()
                    atom_lines = [l for l in lines if l.startswith('ATOM')]
                    print(f"      - Contains {len(atom_lines)} ATOM lines")
            else:
                print(f"   ❌ Unrelaxed PDB NOT found: {unrelaxed_path}")
                
            if os.path.exists(relaxed_path):
                size_relaxed = os.path.getsize(relaxed_path)
                print(f"   ✅ Relaxed PDB exists: {os.path.basename(relaxed_path)} ({size_relaxed} bytes)")
                
                # Check if file has content
                with open(relaxed_path, 'r') as f:
                    lines = f.readlines()
                    atom_lines = [l for l in lines if l.startswith('ATOM')]
                    print(f"      - Contains {len(atom_lines)} ATOM lines")
            else:
                print(f"   ❌ Relaxed PDB NOT found: {relaxed_path}")
            
            # Calculate clash scores
            phenix_wrapper = "/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh"
            
            if os.path.exists(unrelaxed_path):
                try:
                    clash_pre = get_clash_score_phenix(unrelaxed_path, phenix_wrapper)
                    print(f"   Pre-relax clash score: {clash_pre:.2f}")
                    if clash_pre == 0.0:
                        print("   ⚠️ WARNING: Pre-relax clash is 0.0!")
                except Exception as e:
                    print(f"   ❌ Pre-relax clash calculation failed: {e}")
                    
            if os.path.exists(relaxed_path):
                try:
                    clash_post = get_clash_score_phenix(relaxed_path, phenix_wrapper)
                    print(f"   Post-relax clash score: {clash_post:.2f}")
                    if clash_post == 0.0:
                        print("   ⚠️ WARNING: Post-relax clash is exactly 0.0 - this is suspiciously perfect!")
                        print("   🔍 Checking PDB content for issues...")
                        
                        # Additional check: print first few ATOM lines
                        with open(relaxed_path, 'r') as f:
                            lines = f.readlines()
                            atom_lines = [l for l in lines if l.startswith('ATOM')][:5]
                            print("   First few ATOM lines:")
                            for line in atom_lines:
                                print(f"      {line.rstrip()}")
                                
                except Exception as e:
                    print(f"   ❌ Post-relax clash calculation failed: {e}")
                    
    except Exception as e:
        print(f"❌ Error during RhoFold test: {e}")
        import traceback
        traceback.print_exc()

def test_direct_phenix_output():
    """Test Phenix directly to see raw output."""
    print("\n🔬 Test: Direct Phenix output examination")
    
    test_pdb = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data/usalign_rna_example_1.pdb"
    
    if os.path.exists(test_pdb):
        import subprocess
        phenix_wrapper = "/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh"
        
        print(f"Running Phenix directly on: {test_pdb}")
        try:
            result = subprocess.run(
                [phenix_wrapper, "phenix.molprobity", test_pdb],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Look for clash score in output
            for line in result.stdout.split('\n'):
                if 'Clashscore' in line or 'clashscore' in line or 'clash' in line.lower():
                    print(f"   Found: {line}")
                    
            # Check if there's specific text indicating no clashes
            if 'no clashes' in result.stdout.lower() or 'clashscore = 0' in result.stdout.lower():
                print("   ⚠️ Phenix reports no clashes (score = 0)")
                
        except Exception as e:
            print(f"❌ Error running Phenix: {e}")

def main():
    """Run comprehensive clash score tests."""
    print("🧪 Comprehensive Clash Score Investigation")
    print("=" * 60)
    
    # Test 1: Known structure
    test_clash_on_existing_structure()
    
    # Test 2: Direct Phenix examination
    test_direct_phenix_output()
    
    # Test 3: Different RNA sequences with RhoFold
    test_sequences = [
        ("GGGGAAAACCCCUUUU", "simple_16nt"),
        ("GCGCGCGCAUAUAUAU", "alternating_16nt"),
        ("GGCUAAGCCUAGCUAGC", "mixed_17nt"),
        ("GGGAAACCC", "short_9nt"),
    ]
    
    for seq, name in test_sequences:
        test_rhofold_with_relax(seq, name)
    
    print("\n" + "=" * 60)
    print("🔍 Analysis Complete")
    print("\nKey things to check:")
    print("1. Are clash scores exactly 0.0? This is suspicious.")
    print("2. Are PDB files being created correctly?")
    print("3. Is Phenix actually calculating clashes or returning a default?")
    print("4. Do different sequences give different clash scores?")

if __name__ == "__main__":
    main()