#!/usr/bin/env python3
"""
Minimal debug script for lDDT calculation issues.
Tests only the base model with minimal parameters to isolate the lDDT problem.

Usage: python -m dpo.debug.debug_lddt_only
"""

# IMPORTANT: Must be before any imports that use NetworkX
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import tempfile
import subprocess
from pathlib import Path

import torch
import yaml
from types import SimpleNamespace as SN

from src.constants import PROJECT_PATH, DATA_PATH
from src.evaluator import get_lddt, get_lddt_inverse_folding, _get_residue_map, _extract_selected_residues

def _to_sn(o):
    """Convert dict to SimpleNamespace recursively."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o

def load_config():
    """Load a minimal config for testing."""
    config_path = "dpo/configs/bench_full.yaml"
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return _to_sn(raw_config)

def get_simple_test_case():
    """Get a simple test case with known structure."""
    # Use a simple RNA structure for testing
    sequence = "GGGCUCAGUACGGUGGUAUACAGCGCCUCUGAGUCAGCCCUUCAGGCAACUGGGGGAACUGAGGCC"
    structure_id = "3SLQ_1_A"  # Known structure from the evaluation
    
    return {
        'sequence': sequence,
        'id': structure_id,
        'native_pdb': os.path.join(DATA_PATH, "raw", f"{structure_id}.pdb")
    }

def get_test_structure():
    """Get a simple test structure for debugging."""
    # Use the first structure from the evaluation dataset
    processed_path = "data/processed.pt"
    if not os.path.exists(processed_path):
        print(f"❌ Processed data not found: {processed_path}")
        return None
    
    print("🔍 Loading test structure...")
    data = torch.load(processed_path, map_location='cpu')
    
    # Get first test structure
    test_data = data['test'][0]  # First test structure
    print(f"   Test structure ID: {test_data['id_list'][0]}")
    print(f"   Sequence length: {len(test_data['sequence'])}")
    
    return test_data

def test_rhofold_prediction(sequence, output_dir):
    """Test RhoFold structure prediction."""
    print("🧬 Testing RhoFold prediction...")
    
    try:
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        
        # Initialize RhoFold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        
        if not os.path.exists(rhofold_path):
            print(f"❌ RhoFold model not found: {rhofold_path}")
            return None
        
        print(f"   Loading RhoFold model: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=device)['model'])
        rhofold.to(device)
        rhofold.eval()
        
        # Create FASTA file
        fasta_path = os.path.join(output_dir, "test_sequence.fasta")
        pdb_path = os.path.join(output_dir, "predicted_structure.pdb")
        
        seq_record = SeqRecord(Seq(sequence), id="test_seq", description="Test sequence for lDDT debug")
        SeqIO.write(seq_record, fasta_path, "fasta")
        
        print(f"   Predicting structure for sequence: {sequence[:50]}...")
        
        # Predict structure
        coords, plddt = rhofold.predict(fasta_path, pdb_path, use_relax=False)
        
        if os.path.exists(pdb_path):
            print(f"✅ RhoFold prediction successful")
            print(f"   Output PDB: {pdb_path}")
            print(f"   Mean pLDDT: {plddt.mean():.3f}")
            return pdb_path
        else:
            print("❌ RhoFold prediction failed - no PDB output")
            return None
            
    except Exception as e:
        print(f"❌ RhoFold prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_lddt_calculation_detailed(predicted_pdb, native_pdb):
    """Test lDDT calculation with detailed debugging."""
    print("\n🔬 Testing lDDT calculation (detailed debugging)...")
    
    # Step 1: Check file existence
    print(f"   Predicted PDB: {predicted_pdb}")
    print(f"   Native PDB: {native_pdb}")
    
    if not os.path.exists(predicted_pdb):
        print(f"❌ Predicted PDB not found: {predicted_pdb}")
        return None
    
    if not os.path.exists(native_pdb):
        print(f"❌ Native PDB not found: {native_pdb}")
        return None
    
    print("✅ Both PDB files exist")
    
    # Step 2: Test residue mapping
    print("\n🔍 Step 2: Testing residue mapping...")
    try:
        model_chain, model_res_map = _get_residue_map(predicted_pdb)
        print(f"   Predicted structure: Chain {model_chain}, {len(model_res_map)} residues")
        print(f"   Predicted residues: {sorted(list(model_res_map.keys())[:10])}..." if len(model_res_map) > 10 else f"   Predicted residues: {sorted(list(model_res_map.keys()))}")
        
        native_chain, native_res_map = _get_residue_map(native_pdb)
        print(f"   Native structure: Chain {native_chain}, {len(native_res_map)} residues")
        print(f"   Native residues: {sorted(list(native_res_map.keys())[:10])}..." if len(native_res_map) > 10 else f"   Native residues: {sorted(list(native_res_map.keys()))}")
        
    except Exception as e:
        print(f"❌ Residue mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 3: Find common residues
    print("\n🔍 Step 3: Finding common residues...")
    common_res_nums = set(model_res_map.keys()).intersection(set(native_res_map.keys()))
    print(f"   Common residues: {len(common_res_nums)}")
    
    if not common_res_nums:
        print("❌ No common residues found!")
        return None
    
    if len(common_res_nums) < 3:
        print(f"❌ Too few common residues ({len(common_res_nums)} < 3)")
        return None
    
    print(f"✅ Found {len(common_res_nums)} common residues")
    
    # Step 4: Test PDB extraction
    print("\n🔍 Step 4: Testing PDB extraction...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_model_path = os.path.join(temp_dir, 'model_common.pdb')
            extracted_native_path = os.path.join(temp_dir, 'native_common.pdb')
            
            _extract_selected_residues(predicted_pdb, extracted_model_path, model_chain, common_res_nums)
            _extract_selected_residues(native_pdb, extracted_native_path, native_chain, common_res_nums)
            
            if os.path.exists(extracted_model_path) and os.path.exists(extracted_native_path):
                print("✅ PDB extraction successful")
                
                # Check extracted file sizes
                model_size = os.path.getsize(extracted_model_path)
                native_size = os.path.getsize(extracted_native_path)
                print(f"   Extracted model PDB size: {model_size} bytes")
                print(f"   Extracted native PDB size: {native_size} bytes")
                
                # Step 5: Test lDDT tool execution
                print("\n🔍 Step 5: Testing lDDT tool execution...")
                
                lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
                print(f"   lDDT script path: {lddt_script_path}")
                
                if not os.path.exists(lddt_script_path):
                    print(f"❌ lDDT script not found: {lddt_script_path}")
                    return None
                
                lddt_script_dir = os.path.dirname(lddt_script_path)
                chain_mapping = f'{{"{model_chain}":"{native_chain}"}}'
                
                command = [
                    sys.executable,
                    lddt_script_path,
                    extracted_model_path,
                    extracted_native_path,
                    chain_mapping
                ]
                
                print(f"   Command: {' '.join(command)}")
                print(f"   Working directory: {lddt_script_dir}")
                
                result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
                
                print(f"   Return code: {result.returncode}")
                print(f"   STDOUT: '{result.stdout.strip()}'")
                print(f"   STDERR: '{result.stderr.strip()}'")
                
                if result.returncode == 0 and result.stdout.strip():
                    lddt_score = float(result.stdout.strip())
                    print(f"✅ lDDT calculation successful: {lddt_score:.4f}")
                    return lddt_score
                else:
                    print("❌ lDDT calculation failed")
                    if result.stderr:
                        print(f"   Error details: {result.stderr}")
                    return None
            else:
                print("❌ PDB extraction failed")
                return None
                
    except Exception as e:
        print(f"❌ Detailed lDDT test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_lddt_call(predicted_pdb, native_pdb):
    """Test the simple lDDT function call."""
    print("\n🔍 Testing simple lDDT function call...")
    
    try:
        lddt_score = get_lddt(predicted_pdb, native_pdb)
        print(f"   lDDT score: {lddt_score}")
        
        if lddt_score > 0:
            print("✅ lDDT calculation successful")
            return lddt_score
        else:
            print("❌ lDDT calculation returned failure value")
            return None
            
    except Exception as e:
        print(f"❌ Simple lDDT call failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debugging function."""
    print("🔬 lDDT Calculation Debug Script")
    print("="*50)
    
    # Create output directory
    output_dir = "dpo/debug/lddt_debug_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Debug output directory: {output_dir}")
    
    # Get simple test case
    test_case = get_simple_test_case()
    print(f"🧬 Test structure ID: {test_case['id']}")
    print(f"🧬 Test sequence: {test_case['sequence'][:50]}...")
    
    # Check native PDB
    native_pdb = test_case['native_pdb']
    if not os.path.exists(native_pdb):
        print(f"❌ Native PDB not found: {native_pdb}")
        return False
    
    print(f"✅ Native PDB found: {native_pdb}")
    
    # Test RhoFold prediction
    predicted_pdb = test_rhofold_prediction(test_case['sequence'], output_dir)
    if predicted_pdb is None:
        return False
    
    # Test lDDT calculation with detailed debugging
    lddt_detailed = test_lddt_calculation_detailed(predicted_pdb, native_pdb)
    
    # Test simple lDDT call
    lddt_simple = test_simple_lddt_call(predicted_pdb, native_pdb)
    
    # Summary
    print("\n" + "="*50)
    print("🎯 Debug Summary:")
    print(f"   Detailed lDDT: {lddt_detailed}")
    print(f"   Simple lDDT: {lddt_simple}")
    
    if lddt_detailed is not None and lddt_simple is not None:
        print("✅ lDDT calculation is working!")
        print(f"   Debug files saved in: {output_dir}")
        return True
    else:
        print("❌ lDDT calculation has issues")
        print("   Check the detailed error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)