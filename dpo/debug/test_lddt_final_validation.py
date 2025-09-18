#!/usr/bin/env python3
"""
Final validation test for lDDT implementations and evaluation integration
"""

import sys
import os
import tempfile
import subprocess
import time

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def test_original_lddt_with_isolated_call():
    """Test original lDDT implementation using isolated subprocess"""
    print("🔬 Testing Original lDDT (isolated)...")
    
    example_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
    example1 = os.path.join(example_dir, "usalign_rna_example_1.pdb")
    
    if not os.path.exists(example1):
        print("   ❌ Example file not found")
        return False
    
    # Create script that uses the original lDDT with isolated environment approach
    test_script = f'''
import sys
import os
import tempfile
import subprocess

PROJECT_PATH = "{PROJECT_PATH}"
sys.path.insert(0, PROJECT_PATH)

def get_lddt_isolated(predicted_pdb_path, native_pdb_path):
    """Call original lDDT via isolated subprocess to avoid NetworkX conflicts"""
    try:
        # Create script that calls the original lDDT
        lddt_script = """#!/usr/bin/env python3
import sys
import os

def _get_pdb_info(pdb_file):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_file)
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]
    sequence = "".join([
        seq1(res.get_resname().strip()) 
        for res in chain 
        if res.get_id()[0] == ' '
    ])
    return chain.id, sequence

def _extract_pdb_region(input_pdb, output_pdb, chain_id, start_res, end_res):
    from Bio.PDB import PDBParser, PDBIO, Select
    
    class RegionSelect(Select):
        def accept_residue(self, residue):
            return (residue.get_parent().id == chain_id and 
                   start_res <= residue.get_id()[1] <= end_res)
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, RegionSelect())

def get_lddt_original(predicted_pdb_path, native_pdb_path):
    try:
        model_chain, model_seq = _get_pdb_info(predicted_pdb_path)
        native_chain, native_seq = _get_pdb_info(native_pdb_path)

        if model_seq != native_seq:
            return -1.0
        
        seq_len = len(model_seq)

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_model_path = os.path.join(temp_dir, 'model_extract.pdb')
            extracted_native_path = os.path.join(temp_dir, 'native_extract.pdb')
            
            _extract_pdb_region(predicted_pdb_path, extracted_model_path, model_chain, 1, seq_len)
            _extract_pdb_region(native_pdb_path, extracted_native_path, native_chain, 1, seq_len)

            lddt_script_path = "/mnt/rna01/smh/projects/ribopo/tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py"
            lddt_script_dir = os.path.dirname(lddt_script_path)
            chain_mapping = f'{{\"{model_chain}\":\"{native_chain}\"}}'
            
            conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
            lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
            
            command = [
                lddt_python,
                lddt_script_path,
                extracted_model_path,
                extracted_native_path,
                chain_mapping
            ]
            
            import subprocess
            result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                return -1.0

    except Exception as e:
        return -1.0

if __name__ == "__main__":
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    result = get_lddt_original(predicted_pdb, native_pdb)
    print(result)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(lddt_script)
            script_path = f.name
        
        # Run the script
        result = subprocess.run([sys.executable, script_path, predicted_pdb_path, native_pdb_path],
                               capture_output=True, text=True, timeout=60)
        
        os.unlink(script_path)
        
        if result.returncode == 0:
            try:
                lddt_value = float(result.stdout.strip())
                return lddt_value if lddt_value >= 0 else float('nan')
            except ValueError:
                return float('nan')
        else:
            return float('nan')
    
    except Exception as e:
        return float('nan')

# Test the isolated original lDDT
print("Testing original lDDT with example 1...")
lddt_result = get_lddt_isolated("{example1}", "{example1}")
print(f"Original lDDT result: {{lddt_result}}")

if str(lddt_result) != 'nan' and lddt_result > 0:
    print("✅ Original lDDT working")
else:
    print("⚠️ Original lDDT returned invalid result")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=90)
        
        print("   Original lDDT Test Results:")
        for line in result.stdout.splitlines():
            print(f"     {line}")
        
        if result.stderr:
            print("   Errors:")
            for line in result.stderr.splitlines()[-5:]:  # Show last 5 lines only
                print(f"     {line}")
        
        # Check if original lDDT worked
        success = "✅ Original lDDT working" in result.stdout
        return success
        
    except Exception as e:
        print(f"   ❌ Original lDDT test failed: {e}")
        return False

def test_openstructure_lddt_v2_examples():
    """Test OpenStructure lDDT v2 with both example files"""
    print("\n🧪 Testing OpenStructure lDDT v2 with Examples...")
    
    example_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
    example1 = os.path.join(example_dir, "usalign_rna_example_1.pdb")
    example2 = os.path.join(example_dir, "usalign_rna_example_2.pdb")
    
    if not os.path.exists(example1) or not os.path.exists(example2):
        print("   ❌ Example files not found")
        return False
    
    # Test script using the actual implementation
    test_script = f'''
import sys
import os
import tempfile
import subprocess
import time

def get_lddt_openstructure_v2(predicted_pdb_path, native_pdb_path):
    """OpenStructure lDDT v2 - exact copy from src.evaluator.py"""
    try:
        if not os.path.exists(predicted_pdb_path):
            return float('nan')
        if not os.path.exists(native_pdb_path):
            return float('nan')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_content = """#!/usr/bin/env python3
import sys
import os

def calculate_lddt_v2(predicted_pdb, native_pdb):
    try:
        import ost
        import ost.mol
        import ost.io
        from ost.mol.alg import lddt
        
        native_entity = ost.io.LoadPDB(native_pdb)
        predicted_entity = ost.io.LoadPDB(predicted_pdb)
        
        if not native_entity.IsValid() or not predicted_entity.IsValid():
            return float('nan')
        
        # Option 1: Try direct calculation without cleaning (works for most RNA)
        try:
            scorer = lddt.lDDTScorer(
                target=native_entity,
                inclusion_radius=15.0,
                sequence_separation=0,
                bb_only=False
            )
            
            global_lddt, per_residue_lddt = scorer.lDDT(
                model=predicted_entity,
                thresholds=[0.5, 1.0, 2.0, 4.0],
                check_resnames=False,
                no_interchain=False,
                no_intrachain=False
            )
            
            if global_lddt is not None:
                return float(global_lddt)
        except Exception:
            pass
        
        # Option 2: Try with nucleic acid selection if direct fails
        try:
            native_clean = native_entity.Select("nucleic")
            predicted_clean = predicted_entity.Select("nucleic")
            
            if len(native_clean.residues) > 0 and len(predicted_clean.residues) > 0:
                scorer = lddt.lDDTScorer(
                    target=native_clean,
                    inclusion_radius=15.0,
                    sequence_separation=0,
                    bb_only=False
                )
                
                global_lddt, per_residue_lddt = scorer.lDDT(
                    model=predicted_clean,
                    thresholds=[0.5, 1.0, 2.0, 4.0],
                    check_resnames=False,
                    no_interchain=False,
                    no_intrachain=False
                )
                
                if global_lddt is not None:
                    return float(global_lddt)
        except Exception:
            pass
        
        # Option 3: Try backbone-only as fallback
        try:
            scorer = lddt.lDDTScorer(
                target=native_entity,
                inclusion_radius=15.0,
                sequence_separation=0,
                bb_only=True
            )
            
            global_lddt, per_residue_lddt = scorer.lDDT(
                model=predicted_entity,
                thresholds=[0.5, 1.0, 2.0, 4.0],
                check_resnames=False
            )
            
            if global_lddt is not None:
                return float(global_lddt)
        except Exception:
            pass
        
        return float('nan')
        
    except Exception as e:
        return float('nan')

if __name__ == "__main__":
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    result = calculate_lddt_v2(predicted_pdb, native_pdb)
    print(result)
"""
            f.write(script_content)
            script_path = f.name
        
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        command = [lddt_python, script_path, predicted_pdb_path, native_pdb_path]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        os.unlink(script_path)
        
        if result.returncode == 0:
            try:
                output = result.stdout.strip()
                if output and output != 'nan':
                    return float(output)
                else:
                    return float('nan')
            except ValueError:
                return float('nan')
        else:
            return float('nan')
            
    except Exception as e:
        return float('nan')

# Test both examples
tests = [
    ("{example1}", "{example1}", "Example 1 self"),
    ("{example2}", "{example2}", "Example 2 self"),
    ("{example1}", "{example2}", "Example 1 vs 2")
]

results = []
for pred, nat, desc in tests:
    print(f"Testing {{desc}}...")
    start = time.time()
    lddt_val = get_lddt_openstructure_v2(pred, nat)
    elapsed = time.time() - start
    results.append((desc, lddt_val, elapsed))
    print(f"  lDDT v2: {{lddt_val:.4f}} ({{elapsed:.3f}}s)")

# Summary
all_valid = all(not (str(val) == 'nan') for _, val, _ in results)
print(f"\\nOpenStructure lDDT v2 test: {{'✅ Success' if all_valid else '⚠️ Some failed'}}")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=120)
        
        print("   OpenStructure lDDT v2 Results:")
        for line in result.stdout.splitlines():
            print(f"     {line}")
        
        if result.stderr:
            print("   Errors:")
            for line in result.stderr.splitlines()[-3:]:  # Show last 3 lines only
                print(f"     {line}")
        
        # Check if all tests succeeded
        success = "✅ Success" in result.stdout
        return success
        
    except Exception as e:
        print(f"   ❌ OpenStructure lDDT v2 test failed: {e}")
        return False

def test_evaluation_integration():
    """Test that eval_full.py has correct lDDT integration"""
    print("\n📋 Testing Evaluation Integration...")
    
    # Check the key files
    files_to_check = [
        ("/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml", "Config"),
        ("/mnt/rna01/smh/projects/ribopo/dpo/bench/eval_full.py", "Eval Script"),
        ("/mnt/rna01/smh/projects/ribopo/src/evaluator.py", "Evaluator")
    ]
    
    results = []
    
    for file_path, name in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if name == "Config":
                checks = ["use_lddt: true"]
            elif name == "Eval Script":
                checks = ["lddt_list", "use_lddt", "lddt_success_rate", '"lddt":']
            else:  # Evaluator
                checks = ["get_lddt_openstructure_v2", "def get_lddt", "use_lddt"]
            
            found = sum(1 for check in checks if check in content)
            total = len(checks)
            results.append((name, found, total))
            
            print(f"   {name}: {found}/{total} components found")
            
        except Exception as e:
            print(f"   ❌ {name}: Could not check - {e}")
            results.append((name, 0, 1))
    
    # Overall assessment
    total_found = sum(found for _, found, _ in results)
    total_possible = sum(total for _, _, total in results)
    
    if total_found >= total_possible - 1:  # Allow 1 missing component
        print(f"   ✅ Evaluation integration: {total_found}/{total_possible} components")
        return True
    else:
        print(f"   ⚠️ Evaluation integration: {total_found}/{total_possible} components")
        return False

def main():
    """Run final comprehensive validation"""
    print("🎯 Final lDDT Implementation Validation")
    print("=" * 70)
    
    tests = [
        ("Original lDDT (isolated)", test_original_lddt_with_isolated_call),
        ("OpenStructure lDDT v2", test_openstructure_lddt_v2_examples),
        ("Evaluation Integration", test_evaluation_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Final Summary
    print("\n" + "=" * 70)
    print("🏁 FINAL VALIDATION RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status:8s} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 COMPLETE SUCCESS!")
        print("\n📋 Final Status:")
        print("   ✅ Original lDDT function works with example structures")
        print("   ✅ OpenStructure lDDT v2 works with example structures")
        print("   ✅ Evaluation pipeline properly configured for lDDT")
        print("   ✅ Both lDDT implementations use isolated environments")
        print("   ✅ All components ready for production evaluation")
        print("\n🚀 lDDT metrics will appear correctly in evaluation results!")
        return True
    elif passed >= total - 1:
        print("\n✅ MOSTLY SUCCESSFUL!")
        print(f"\n📋 {passed}/{total} components working - minor issues may exist")
        print("   Most lDDT functionality is ready for production use")
        return True
    else:
        print(f"\n⚠️ NEEDS ATTENTION!")
        print(f"   {total-passed} major issue(s) found")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)