# /mnt/rna01/smh/projects/ribopo/dpo/debug/eval_test_lddt.py
import os
import sys
import subprocess
import re
import tempfile
from Bio.PDB import PDBParser, Select, PDBIO
from Bio.SeqUtils import seq1

# --- Setup Project Paths ---
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)
base_dir = os.path.join(PROJECT_PATH, 'tools', 'RNA_assessment')
sys.path.insert(0, os.path.join(base_dir, 'lddt', 'bin'))

# --- Production-Ready lDDT Wrapper and Helpers ---

def _get_residue_map(pdb_file):
    """Parses a PDB and returns the first chain ID and a map of {res_num: res_name}."""
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
    """Extracts a specific set of residues from a specific chain."""
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            return residue.get_parent().id == chain_id and residue.get_id()[1] in res_nums_to_keep

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect())

def get_lddt_robust(predicted_pdb_path, native_pdb_path):
    """
    Robustly calculates lDDT by comparing only residues that have both
    matching numbers AND matching sequence identity.
    """
    try:
        model_chain, model_res_map = _get_residue_map(predicted_pdb_path)
        native_chain, native_res_map = _get_residue_map(native_pdb_path)

        # --- FIX: Find common residues by checking for matching numbers AND names ---
        common_res_nums = set()
        # Iterate through residue numbers present in the model
        for res_num, model_res_name in model_res_map.items():
            # Check if the same residue number exists in the native structure
            if res_num in native_res_map:
                # Check if the residue names (the bases) are identical
                if model_res_name == native_res_map[res_num]:
                    common_res_nums.add(res_num)

        if not common_res_nums:
            print("❌ ERROR: No common residues with matching sequence found.")
            return -1.0

        print(f"  -> Found {len(common_res_nums)} common residues with matching sequence to compare.")

        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_model_path = os.path.join(temp_dir, 'model_common.pdb')
            extracted_native_path = os.path.join(temp_dir, 'native_common.pdb')

            _extract_selected_residues(predicted_pdb_path, extracted_model_path, model_chain, common_res_nums)
            _extract_selected_residues(native_pdb_path, extracted_native_path, native_chain, common_res_nums)

            lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
            lddt_script_dir = os.path.dirname(lddt_script_path)
            chain_mapping = f'{{"{model_chain}":"{native_chain}"}}'

            command = [
                sys.executable,
                lddt_script_path,
                extracted_model_path,
                extracted_native_path,
                chain_mapping
            ]

            result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                print(f"--- lDDT Subprocess Failed ---")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return -1.0

    except Exception as e:
        print(f"An error occurred during robust lDDT calculation: {e}")
        return -1.0

# --- Main Test Execution ---
if __name__ == '__main__':
    print("--- 🧪 Testing the robust get_lddt function ---\n")

    model_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/model.pdb')
    native_pdb = os.path.join(PROJECT_PATH, 'dpo/debug/example_data/native.pdb')

    score = get_lddt_robust(model_pdb, native_pdb)
    expected_score = 0.6209

    if score != -1.0:
        print(f"\nCalculated lDDT Score: {score:.4f}")
        if abs(score - expected_score) < 0.01:
            print("✅ Test Passed! The robust function works correctly.")
        else:
            print(f"❌ Test Failed! Score ({score:.4f}) did not match expected value ({expected_score:.4f}).")
    else:
        print("❌ Test Failed! Function returned an error.")