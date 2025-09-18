#!/usr/bin/env python3
"""
Debug OpenStructure lDDT calculation issues
"""

import os
import subprocess
import tempfile

def debug_openstructure_lddt():
    """Debug why OpenStructure lDDT returns NaN"""
    
    # Create a simple, valid RNA PDB
    pdb_content = """HEADER    TEST RNA STRUCTURE
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
    
    debug_script = '''#!/usr/bin/env python3
import sys
import ost
import ost.io
from ost.mol.alg import lddt

def debug_lddt(pdb_path):
    """Debug lDDT calculation step by step"""
    print(f"=== Debug lDDT calculation for {pdb_path} ===")
    
    try:
        # Step 1: Load structure
        print("Step 1: Loading structure...")
        entity = ost.io.LoadPDB(pdb_path)
        print(f"  Entity valid: {entity.IsValid()}")
        print(f"  Chain count: {len(entity.chains)}")
        print(f"  Residue count: {len(entity.residues)}")
        print(f"  Atom count: {len(entity.atoms)}")
        
        if not entity.IsValid():
            print("  ERROR: Invalid entity")
            return
        
        # Step 2: Examine structure contents
        print("\\nStep 2: Structure analysis...")
        for i, chain in enumerate(entity.chains):
            print(f"  Chain {i}: {chain.name}, residues: {len(chain.residues)}")
            for j, res in enumerate(chain.residues):
                if j < 5:  # Show first 5 residues
                    print(f"    Residue {j}: {res.name} {res.number} atoms: {len(res.atoms)}")
                    chem_class = "unknown"
                    try:
                        if res.chem_class.IsNucleotideLinking():
                            chem_class = "nucleotide"
                        elif res.chem_class.IsPeptideLinking():
                            chem_class = "peptide"
                    except:
                        pass
                    print(f"      Chem class: {chem_class}")
        
        # Step 3: Try to create lDDT scorer
        print("\\nStep 3: Creating lDDT scorer...")
        try:
            scorer = lddt.lDDTScorer(
                target=entity,
                inclusion_radius=15.0,
                sequence_separation=0,
                bb_only=False
            )
            print("  ✅ lDDT scorer created successfully")
        except Exception as e:
            print(f"  ❌ Failed to create lDDT scorer: {e}")
            return
        
        # Step 4: Try lDDT calculation
        print("\\nStep 4: Computing lDDT (self vs self)...")
        try:
            global_lddt, per_residue_lddt = scorer.lDDT(
                model=entity,
                thresholds=[0.5, 1.0, 2.0, 4.0],
                check_resnames=False,
                no_interchain=False,
                no_intrachain=False
            )
            
            print(f"  Global lDDT: {global_lddt}")
            print(f"  Per-residue lDDT length: {len(per_residue_lddt) if per_residue_lddt else 'None'}")
            
            if per_residue_lddt:
                print(f"  First few per-residue lDDT: {per_residue_lddt[:5]}")
            
        except Exception as e:
            print(f"  ❌ lDDT calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 5: Try with backbone only
        print("\\nStep 5: Trying backbone-only mode...")
        try:
            scorer_bb = lddt.lDDTScorer(
                target=entity,
                inclusion_radius=15.0,
                sequence_separation=0,
                bb_only=True
            )
            
            global_lddt_bb, _ = scorer_bb.lDDT(
                model=entity,
                thresholds=[0.5, 1.0, 2.0, 4.0],
                check_resnames=False
            )
            
            print(f"  Backbone-only lDDT: {global_lddt_bb}")
            
        except Exception as e:
            print(f"  ❌ Backbone-only failed: {e}")
        
        # Step 6: Check contacts
        print("\\nStep 6: Checking contacts...")
        try:
            # Try to get number of contacts for first chain
            if len(entity.chains) > 0:
                chain_name = entity.chains[0].name
                n_contacts = scorer.GetNChainContacts(chain_name, no_interchain=False)
                print(f"  Contacts for chain {chain_name}: {n_contacts}")
            
        except Exception as e:
            print(f"  ❌ Contact check failed: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: debug_script.py <pdb_file>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    debug_lddt(pdb_file)
'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pdb_file:
        pdb_file.write(pdb_content)
        pdb_path = pdb_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_file.write(debug_script)
        script_path = script_file.name
    
    try:
        # Run debug script
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        print("🔍 Running OpenStructure lDDT debug...")
        result = subprocess.run(
            [lddt_python, script_path, pdb_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("=== DEBUG OUTPUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        
        print(f"=== Return code: {result.returncode} ===")
        
    finally:
        # Cleanup
        os.unlink(pdb_path)
        os.unlink(script_path)

if __name__ == "__main__":
    debug_openstructure_lddt()