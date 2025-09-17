#!/usr/bin/env python3
"""
Detailed debug of lDDT calculation with isolated environment
"""

import sys
import os
import tempfile
import subprocess
from Bio.PDB import PDBParser, Select, PDBIO
from Bio.SeqUtils import seq1

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def _get_residue_map(pdb_file):
    """Get residue map from PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_file)
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]
    residue_map = {
        res.get_id()[1]: seq1(res.get_resname().strip())
        for res in chain if res.get_id()[0] == ' '
    }
    return chain.id, residue_map

def _extract_selected_residues(input_pdb, output_pdb, chain_id, res_nums_to_keep):
    """Extract selected residues"""
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            return residue.get_parent().id == chain_id and residue.get_id()[1] in res_nums_to_keep

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect())

def test_lddt_detailed():
    print("🔬 Detailed lDDT Debug with Isolated Environment")
    print("=" * 60)
    
    # Test files
    model_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/model.pdb')
    native_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/native.pdb')
    
    print(f"📁 Test files:")
    print(f"   Model: {model_pdb}")
    print(f"   Native: {native_pdb}")
    
    # Step 1: Get residue maps
    print(f"\n🔍 Step 1: Getting residue maps...")
    model_chain, model_res_map = _get_residue_map(model_pdb)
    native_chain, native_res_map = _get_residue_map(native_pdb)
    
    print(f"   Model chain: {model_chain}, residues: {len(model_res_map)}")
    print(f"   Native chain: {native_chain}, residues: {len(native_res_map)}")
    
    # Step 2: Find common residues
    print(f"\n🔍 Step 2: Finding common residues...")
    common_res_nums = set(model_res_map.keys()).intersection(set(native_res_map.keys()))
    print(f"   Common residues: {len(common_res_nums)}")
    
    if len(common_res_nums) < 3:
        print(f"❌ Too few common residues")
        return False
    
    # Step 3: Extract structures
    print(f"\n🔍 Step 3: Extracting structures...")
    with tempfile.TemporaryDirectory() as temp_dir:
        extracted_model_path = os.path.join(temp_dir, 'model_common.pdb')
        extracted_native_path = os.path.join(temp_dir, 'native_common.pdb')
        
        _extract_selected_residues(model_pdb, extracted_model_path, model_chain, common_res_nums)
        _extract_selected_residues(native_pdb, extracted_native_path, native_chain, common_res_nums)
        
        print(f"   Extracted model: {os.path.getsize(extracted_model_path)} bytes")
        print(f"   Extracted native: {os.path.getsize(extracted_native_path)} bytes")
        
        # Step 4: Test lDDT tool with isolated environment
        print(f"\n🔍 Step 4: Testing lDDT tool with isolated environment...")
        
        lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
        lddt_script_dir = os.path.dirname(lddt_script_path)
        chain_mapping = f'{{"{model_chain}":"{native_chain}"}}'
        
        # Use isolated environment
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        command = [
            lddt_python,
            lddt_script_path,
            extracted_model_path,
            extracted_native_path,
            chain_mapping
        ]
        
        print(f"   Command: {' '.join(command)}")
        print(f"   Working dir: {lddt_script_dir}")
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
        
        print(f"\n📊 Results:")
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT: '{result.stdout.strip()}'")
        print(f"   STDERR: '{result.stderr.strip()}'")
        
        if result.returncode == 0 and result.stdout.strip():
            lddt_score = float(result.stdout.strip())
            print(f"\n✅ lDDT calculation SUCCESSFUL: {lddt_score:.4f}")
            return True
        else:
            print(f"\n❌ lDDT calculation FAILED")
            return False

if __name__ == "__main__":
    success = test_lddt_detailed()
    sys.exit(0 if success else 1)
