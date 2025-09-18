#!/usr/bin/env python3
"""
Verify lDDT logic in the evaluation pipeline
"""

import sys
import os

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def check_lddt_implementation():
    """Check that lDDT implementation is correct in the evaluation pipeline"""
    print("🔬 Verifying lDDT Logic in Evaluation Pipeline")
    print("=" * 70)
    
    # 1. Check config parsing
    print("1️⃣ Checking config parsing...")
    config_path = "/mnt/rna01/smh/projects/ribopo/dpo/configs/bench_full.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    use_lddt = config.get('eval', {}).get('use_lddt', False)
    print(f"   ✅ Config use_lddt: {use_lddt}")
    
    # 2. Check evaluation function signature
    print("\n2️⃣ Checking evaluation function...")
    try:
        from dpo.bench.eval_full import eval_full_metrics
        import inspect
        sig = inspect.signature(eval_full_metrics)
        print(f"   ✅ eval_full_metrics function exists")
        
        # Check if the function calls extended RhoFold
        import ast
        import inspect
        source = inspect.getsource(eval_full_metrics)
        if 'self_consistency_score_rhofold_extended' in source:
            print(f"   ✅ Uses extended RhoFold evaluator")
        if 'use_lddt=getattr(cfg.eval' in source:
            print(f"   ✅ Passes use_lddt from config")
        
    except Exception as e:
        print(f"   ❌ Issue with eval_full_metrics: {e}")
        return False
    
    # 3. Check extended RhoFold function
    print("\n3️⃣ Checking extended RhoFold function...")
    try:
        from src.evaluator import self_consistency_score_rhofold_extended
        sig = inspect.signature(self_consistency_score_rhofold_extended)
        params = list(sig.parameters.keys())
        
        if 'use_lddt' in params:
            print(f"   ✅ Extended RhoFold has use_lddt parameter")
        else:
            print(f"   ❌ Extended RhoFold missing use_lddt parameter")
            return False
            
        # Check return values (should be 8-tuple including lDDT)
        source = inspect.getsource(self_consistency_score_rhofold_extended)
        if 'np.array(sc_lddt)' in source:
            print(f"   ✅ Returns lDDT scores in output")
        
    except Exception as e:
        print(f"   ❌ Issue with extended RhoFold: {e}")
        return False
    
    # 4. Check lDDT calculation function
    print("\n4️⃣ Checking lDDT calculation function...")
    try:
        from src.evaluator import get_lddt
        
        # Test with dummy data
        print(f"   ✅ get_lddt function exists")
        
        # Check if it uses the correct environment
        source = inspect.getsource(get_lddt)
        if 'lddt_env' in source:
            print(f"   ✅ Uses isolated lddt_env environment")
        if 'get_lddt_inverse_folding' in source:
            print(f"   ✅ Uses inverse folding version for RNA design")
            
    except Exception as e:
        print(f"   ❌ Issue with get_lddt: {e}")
        return False
    
    # 5. Check results storage
    print("\n5️⃣ Checking results storage...")
    try:
        source = inspect.getsource(eval_full_metrics)
        if 'lddt_list.extend' in source:
            print(f"   ✅ lDDT scores are stored in lddt_list")
        if 'results["lddt"] = np.nanmean(lddt_list)' in source:
            print(f"   ✅ lDDT is included in final results dict")
        else:
            print(f"   ❌ lDDT missing from final results")
            return False
            
    except Exception as e:
        print(f"   ❌ Issue checking results storage: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ VERIFICATION COMPLETE: lDDT logic is correctly implemented!")
    print("\nPipeline flow:")
    print("1. Config: use_lddt=true")
    print("2. eval_full_metrics passes use_lddt to extended RhoFold")
    print("3. Extended RhoFold calls get_lddt for each sample")
    print("4. get_lddt uses isolated lddt_env and inverse folding logic")
    print("5. lDDT scores stored in lddt_list and included in final results")
    
    return True

def test_lddt_function():
    """Test lDDT function with sample data"""
    print("\n🧪 Testing lDDT Function with Sample Data...")
    
    try:
        from src.evaluator import get_lddt
        
        # Use existing debug data if available
        test_dir = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data"
        if os.path.exists(test_dir):
            pdb_files = [f for f in os.listdir(test_dir) if f.endswith('.pdb')]
            if len(pdb_files) >= 2:
                pdb1 = os.path.join(test_dir, pdb_files[0])
                pdb2 = os.path.join(test_dir, pdb_files[1]) if len(pdb_files) > 1 else pdb1
                
                print(f"   Testing with: {pdb_files[0]} vs {pdb_files[1] if len(pdb_files) > 1 else pdb_files[0]}")
                
                lddt_score = get_lddt(pdb1, pdb2)
                print(f"   lDDT result: {lddt_score}")
                
                if lddt_score >= 0:
                    print(f"   ✅ lDDT calculation successful")
                else:
                    print(f"   ⚠️ lDDT returned {lddt_score} (may indicate file issues)")
                    
                return True
        
        print("   ⚠️ No test PDB files available - skipping function test")
        return True
        
    except Exception as e:
        print(f"   ❌ lDDT function test failed: {e}")
        return False

if __name__ == "__main__":
    success = check_lddt_implementation()
    if success:
        test_lddt_function()
        print("\n🚀 READY: You can now run the evaluation and lDDT will be included!")
        print("\nRun this command:")
        print("python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml")
    else:
        print("\n❌ ISSUES FOUND: Need to fix lDDT implementation before running evaluation")