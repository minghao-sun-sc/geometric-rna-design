#!/usr/bin/env python3
"""
Analyze chain structure to design proper chain mapping strategy.
"""

import os
import sys
import tempfile
import subprocess

def analyze_chain_structure():
    """Analyze the chain structure of both predicted and native structures."""
    
    predicted_pdb = "/mnt/rna01/smh/projects/ribopo/multiround/eval_multiround/01_base_dpo_t05/designs_BASE/20250919_153427/sample0/design0_unrelaxed.pdb"
    native_pdb = "/mnt/rna01/smh/projects/ribopo/data/raw/3B58_1_B-C-A.pdb"
    
    script_content = f'''#!/usr/bin/env python3
import sys

def analyze_chains(predicted_pdb, native_pdb):
    try:
        import ost
        import ost.mol
        import ost.io
        
        print("=== Chain Structure Analysis ===")
        
        # Load structures
        native_entity = ost.io.LoadPDB(native_pdb)
        predicted_entity = ost.io.LoadPDB(predicted_pdb)
        
        print(f"\\n--- Native Structure ({native_pdb.split('/')[-1]}) ---")
        print(f"Total residues: {{len(native_entity.residues)}}")
        print(f"Number of chains: {{len(native_entity.chains)}}")
        
        for chain in native_entity.chains:
            print(f"  Chain {{chain.name}}: {{len(chain.residues)}} residues")
            if len(chain.residues) > 0:
                first_res = chain.residues[0]
                last_res = chain.residues[-1]
                print(f"    Range: {{first_res.name}}{{first_res.number}} to {{last_res.name}}{{last_res.number}}")
        
        print(f"\\n--- Predicted Structure ({predicted_pdb.split('/')[-1]}) ---")
        print(f"Total residues: {{len(predicted_entity.residues)}}")
        print(f"Number of chains: {{len(predicted_entity.chains)}}")
        
        for chain in predicted_entity.chains:
            print(f"  Chain {{chain.name}}: {{len(chain.residues)}} residues")
            if len(chain.residues) > 0:
                first_res = chain.residues[0]
                last_res = chain.residues[-1]
                print(f"    Range: {{first_res.name}}{{first_res.number}} to {{last_res.name}}{{last_res.number}}")
        
        # Suggest mapping strategy
        print(f"\\n--- Mapping Strategy ---")
        native_chains = [(c.name, len(c.residues)) for c in native_entity.chains]
        predicted_chains = [(c.name, len(c.residues)) for c in predicted_entity.chains]
        
        print(f"Native chains: {{native_chains}}")
        print(f"Predicted chains: {{predicted_chains}}")
        
        # Simple strategy: map largest predicted chain to largest native chain
        if native_chains and predicted_chains:
            largest_native = max(native_chains, key=lambda x: x[1])
            largest_predicted = max(predicted_chains, key=lambda x: x[1])
            
            print(f"\\nSuggested mapping:")
            print(f"  Predicted chain '{{largest_predicted[0]}}' ({{largest_predicted[1]}} residues)")
            print(f"  -> Native chain '{{largest_native[0]}}' ({{largest_native[1]}} residues)")
            
            mapping = {{largest_predicted[0]: largest_native[0]}}
            print(f"\\nChain mapping dict: {{mapping}}")
            
            return mapping
        
        return None
        
    except Exception as e:
        print(f"Error analyzing chains: {{e}}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    mapping = analyze_chains(predicted_pdb, native_pdb)
    print(f"\\nFinal mapping: {{mapping}}")
'''
    
    # Write and run analysis script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        command = [lddt_python, script_path, predicted_pdb, native_pdb]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    finally:
        os.unlink(script_path)

if __name__ == "__main__":
    analyze_chain_structure()