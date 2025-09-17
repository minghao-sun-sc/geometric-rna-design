# dpo/debug/calculate_scores_rna_assess.py
import os
import sys
import subprocess
import re
from Bio.PDB import PDBParser, PDBIO, Select

# Add necessary local source directories to Python's path
base_dir = os.path.join('tools', 'RNA_assessment')
sys.path.insert(0, os.path.join(base_dir, 'lddt', 'bin'))

# Robust extraction function using Biopython
def extract_pdb_with_biopython(input_pdb, output_pdb, chain_id, start_res, end_res):
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            het, res_id, ins_code = residue.get_id()
            in_chain = residue.get_parent().id == chain_id
            in_range = start_res <= res_id <= end_res
            return in_chain and in_range

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect())
    print(f"  -> Saved extracted region to {output_pdb}")

print("--- 🧬 RNA Assessment Script (Definitive Version) ---")

# --- SETUP ---
data_dir = os.path.join('dpo', 'debug', 'example_data')
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)

predicted_pdb_rel = os.path.join(data_dir, 'model.pdb')
native_pdb_rel = os.path.join(data_dir, 'native.pdb')

lddt_script_path = os.path.abspath(os.path.join(base_dir, 'lddt', 'bin', 'complex_lddt_no_stereocheck.py'))
# --- FIX: Use the phenix.clashscore wrapper ---
clashscore_executable = os.path.abspath('tools/molprobity/build/bin/phenix.clashscore')
    
print(f"\n[INFO] Using Predicted Model: {predicted_pdb_rel}")
print(f"[INFO] Using Native Reference: {native_pdb_rel}\n")

# --- 1. Extract PDB regions using Biopython ---
print("[RUNNING] Extracting PDB regions with Biopython...")
try:
    extracted_model_path = os.path.join(processed_dir, 'model_extract.pdb')
    extracted_native_path = os.path.join(processed_dir, 'native_extract.pdb')
    extract_pdb_with_biopython(predicted_pdb_rel, extracted_model_path, 'U', 1, 31)
    extract_pdb_with_biopython(native_pdb_rel, extracted_native_path, 'A', 1, 31)
    print("✅ PDB extraction successful.\n")
except Exception as e:
    print(f"❌ PDB extraction failed.\n   Error: {e}\n")

# --- 2. Calculate lDDT on EXTRACTED files ---
try:
    print("[RUNNING] Calculating lDDT...")
    command = [
        sys.executable,
        lddt_script_path,
        os.path.abspath(extracted_model_path),
        os.path.abspath(extracted_native_path),
        '{"U":"A"}'
    ]
    lddt_script_dir = os.path.dirname(lddt_script_path)
    result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)

    if result.returncode == 0 and result.stdout.strip():
        # --- FIX: The script outputs a raw number, so we convert it directly ---
        lddt_score = float(result.stdout.strip())
        print(f"✅ lDDT Score: {lddt_score:.4f}\n")
    else:
        print(f"❌ lDDT script failed or produced no output.\n   Error: {result.stderr}\n")
except Exception as e:
    print(f"❌ Could not calculate lDDT.\n   Error: {e}\n")

# --- 3. Calculate Clash Score ---
try:
    print("[RUNNING] Calculating Clash Score...")
    pdb_to_check = os.path.abspath(predicted_pdb_rel)
    
    # --- FIX: Call the phenix.clashscore wrapper directly ---
    command = [clashscore_executable, pdb_to_check]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        match = re.search(r"clashscore = (\d+\.\d+)", result.stdout)
        if match:
            print(f"✅ Clash Score: {float(match.group(1))}\n")
        else:
            print(f"❌ Could not parse Clash Score from tool output.\n")
            print(f"   Tool Output:\n{result.stdout}")
    else:
        print(f"❌ MolProbity/Phenix failed.\n   Error:\n{result.stderr}")
except Exception as e:
    print(f"❌ Could not calculate Clash Score.\n   Error: {e}\n")

print("--- Calculations Complete ---")