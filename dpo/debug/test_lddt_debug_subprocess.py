#!/usr/bin/env python3
"""
Debug the subprocess call for OpenStructure lDDT v2
"""

import sys
import os
import subprocess
import tempfile

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def debug_lddt_subprocess():
    """Debug what's happening in the subprocess call"""
    
    # Create a simple RNA PDB
    rna_pdb = """HEADER    TEST RNA STRUCTURE
ATOM      1  P     A A   1      -0.314   2.117   1.395  1.00 20.00           P
ATOM      2  O5'   A A   1      -0.344   0.623   1.895  1.00 20.00           O  
ATOM      3  C5'   A A   1      -1.556   0.217   2.541  1.00 20.00           C
ATOM      4  C4'   A A   1      -1.208  -0.742   3.648  1.00 20.00           C
ATOM      5  O4'   A A   1      -0.571  -1.876   3.056  1.00 20.00           O
ATOM      6  C3'   A A   1      -0.182   0.019   4.475  1.00 20.00           C
ATOM      7  O3'   A A   1      -0.623   0.425   5.767  1.00 20.00           O
ATOM      8  C2'   A A   1       0.828  -1.006   4.930  1.00 20.00           C
ATOM      9  O2'   A A   1       1.238  -1.850   3.863  1.00 20.00           O
ATOM     10  C1'   A A   1       0.018  -1.835   3.930  1.00 20.00           C
ATOM     11  N9    A A   1       0.928  -2.511   3.002  1.00 20.00           N
ATOM     12  C8    A A   1       1.156  -2.363   1.661  1.00 20.00           C
ATOM     13  N7    A A   1       2.049  -3.142   1.113  1.00 20.00           N
ATOM     14  C5    A A   1       2.499  -3.952   2.123  1.00 20.00           C
ATOM     15  C6    A A   1       3.451  -4.913   2.326  1.00 20.00           C
ATOM     16  N6    A A   1       4.075  -5.128   1.311  1.00 20.00           N
ATOM     17  N1    A A   1       3.620  -5.589   3.490  1.00 20.00           N
ATOM     18  C2    A A   1       2.912  -5.373   4.507  1.00 20.00           C
ATOM     19  N3    A A   1       1.989  -4.476   4.463  1.00 20.00           N
ATOM     20  C4    A A   1       1.838  -3.815   3.298  1.00 20.00           C
ATOM     21  P     U A   2       0.214   1.293   6.789  1.00 20.00           P
ATOM     22  O5'   U A   2       0.694   0.384   7.895  1.00 20.00           O
ATOM     23  C5'   U A   2       1.744  -0.565   7.695  1.00 20.00           C
ATOM     24  C4'   U A   2       1.908  -1.408   8.938  1.00 20.00           C
ATOM     25  O4'   U A   2       1.156  -2.625   8.802  1.00 20.00           O
ATOM     26  C3'   U A   2       1.331  -0.742  10.183  1.00 20.00           C
ATOM     27  O3'   U A   2       2.130  -0.864  11.351  1.00 20.00           O
ATOM     28  C2'   U A   2       1.231  -1.897  11.179  1.00 20.00           C
ATOM     29  O2'   U A   2       2.540  -2.408  11.329  1.00 20.00           O
ATOM     30  C1'   U A   2       0.881  -3.032   9.878  1.00 20.00           C
ATOM     31  N1    U A   2      -0.552  -3.385   9.773  1.00 20.00           N
ATOM     32  C2    U A   2      -0.848  -4.625  10.278  1.00 20.00           C
ATOM     33  O2    U A   2      -0.009  -5.323  10.824  1.00 20.00           O
ATOM     34  N3    U A   2      -2.150  -4.948  10.190  1.00 20.00           N
ATOM     35  C4    U A   2      -3.148  -4.189   9.598  1.00 20.00           C
ATOM     36  O4    U A   2      -4.278  -4.540   9.566  1.00 20.00           O
ATOM     37  C5    U A   2      -2.776  -2.945   9.093  1.00 20.00           C
ATOM     38  C6    U A   2      -1.522  -2.621   9.174  1.00 20.00           C
END
"""
    
    # Create the lDDT calculation script
    lddt_script = '''#!/usr/bin/env python3
import sys
import os

def calculate_lddt_v2(predicted_pdb, native_pdb):
    """Calculate lDDT using OpenStructure v2"""
    try:
        import ost
        import ost.mol
        import ost.io
        from ost.mol.alg import lddt
        
        print(f"Loading: {predicted_pdb} and {native_pdb}", file=sys.stderr)
        
        # Load structures
        native_entity = ost.io.LoadPDB(native_pdb)
        predicted_entity = ost.io.LoadPDB(predicted_pdb)
        
        print(f"Native valid: {native_entity.IsValid()}, Predicted valid: {predicted_entity.IsValid()}", file=sys.stderr)
        
        if not native_entity.IsValid() or not predicted_entity.IsValid():
            print("Invalid entities", file=sys.stderr)
            return float('nan')
        
        # Clean structures - keep only nucleic acids
        native_clean = clean_structure_for_lddt(native_entity)
        predicted_clean = clean_structure_for_lddt(predicted_entity)
        
        print(f"Cleaned - Native valid: {native_clean.IsValid()}, Predicted valid: {predicted_clean.IsValid()}", file=sys.stderr)
        
        if not native_clean.IsValid() or not predicted_clean.IsValid():
            print("Invalid cleaned entities", file=sys.stderr)
            return float('nan')
        
        # Create lDDT scorer
        print("Creating lDDT scorer...", file=sys.stderr)
        scorer = lddt.lDDTScorer(
            target=native_clean,
            inclusion_radius=15.0,
            sequence_separation=0,
            bb_only=False
        )
        
        # Compute lDDT
        print("Computing lDDT...", file=sys.stderr)
        global_lddt, per_residue_lddt = scorer.lDDT(
            model=predicted_clean,
            thresholds=[0.5, 1.0, 2.0, 4.0],
            check_resnames=False,
            no_interchain=False,
            no_intrachain=False
        )
        
        print(f"lDDT result: {global_lddt}", file=sys.stderr)
        
        if global_lddt is None:
            print("Global lDDT is None", file=sys.stderr)
            return float('nan')
        
        return float(global_lddt)
        
    except Exception as e:
        print(f"lDDT calculation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return float('nan')

def clean_structure_for_lddt(entity):
    """Clean structure for lDDT calculation"""
    try:
        clean_view = entity.CreateEmptyView()
        
        for chain in entity.chains:
            chain_view = clean_view.AddChain(chain, deep=True)
            
            for residue in chain.residues:
                # Skip water and ligands
                if residue.name in ['HOH', 'WAT', 'SO4', 'PO4']:
                    continue
                
                # Keep RNA residues
                rna_residues = {'A', 'U', 'G', 'C', 'T'}
                if residue.name in rna_residues and len(residue.atoms) >= 3:
                    chain_view.AddResidue(residue, deep=True)
        
        print(f"Cleaned structure has {len(clean_view.residues)} residues", file=sys.stderr)
        return clean_view
        
    except Exception as e:
        print(f"Structure cleaning failed: {e}", file=sys.stderr)
        return entity.CreateEmptyView()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <predicted_pdb> <native_pdb>")
        sys.exit(1)
    
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    
    result = calculate_lddt_v2(predicted_pdb, native_pdb)
    print(result)  # This should be the only stdout output
'''
    
    test_dir = "/tmp/lddt_debug_test"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create test PDB
        native_path = os.path.join(test_dir, "native.pdb")
        with open(native_path, 'w') as f:
            f.write(rna_pdb)
        
        # Save the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(lddt_script)
            script_path = f.name
        
        try:
            # Use lddt_env to run the calculation
            conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
            lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
            
            print("Running OpenStructure lDDT subprocess...")
            result = subprocess.run(
                [lddt_python, script_path, native_path, native_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: '{result.stdout}'")
            print(f"STDERR: '{result.stderr}'")
            
            if result.returncode == 0:
                try:
                    output = result.stdout.strip()
                    print(f"Parsed output: '{output}'")
                    lddt_value = float(output)
                    print(f"lDDT value: {lddt_value}")
                except ValueError as e:
                    print(f"Could not parse as float: {e}")
            
        finally:
            # Clean up
            os.unlink(script_path)
        
    finally:
        # Clean up
        os.unlink(native_path)
        os.rmdir(test_dir)

if __name__ == "__main__":
    debug_lddt_subprocess()