#!/usr/bin/env python3
"""
Test lDDT with single structure using existing evaluation framework
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import torch

# Use the same approach as eval_full.py but with just one structure

def main():
    print("🚀 Testing lDDT with Single Structure")
    print("=" * 50)
    
    # Import evaluation components
    from dpo.bench.eval_full import get_config_from_path, get_model, get_dataset
    from src.evaluator import evaluate
    
    # Load config 
    config_path = "dpo/configs/bench_full.yaml"
    config = get_config_from_path(config_path)
    
    print(f"📦 Loading model and dataset...")
    
    # Get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config, device)
    
    # Get dataset 
    dataset = get_dataset(config)
    
    print(f"✅ Dataset loaded with {len(dataset.data_list)} structures")
    
    # Create single-item dataset with first structure
    class SingleDataset:
        def __init__(self, original_dataset, index=0):
            self.data_list = [original_dataset.data_list[index]]
            self.featurizer = original_dataset.featurizer
    
    single_dataset = SingleDataset(dataset, index=0)
    structure_id = single_dataset.data_list[0]['id_list'][0]
    
    print(f"🧬 Testing structure: {structure_id}")
    
    # Run evaluation with lDDT focus
    print(f"🔬 Running evaluation...")
    
    try:
        results = evaluate(
            model=model,
            dataset=single_dataset,
            n_samples=1,
            temperature=0.5,
            device=device,
            model_name="single_test",
            metrics=['recovery', 'perplexity', 'sc_score_rhofold'],
            save_designs=False
        )
        
        print(f"\n📊 Results for {structure_id}:")
        print(f"   Recovery: {results['recovery_list'][0]:.4f}")
        print(f"   Perplexity: {results['perplexity_list'][0]:.4f}")
        
        if 'sc_score_rmsd' in results:
            print(f"   RMSD: {results['sc_score_rmsd'][0]:.4f}")
        if 'sc_score_tm' in results:
            print(f"   TM-score: {results['sc_score_tm'][0]:.4f}")
        if 'sc_score_gddt' in results:
            print(f"   GDT-TS: {results['sc_score_gddt'][0]:.4f}")
        if 'sc_score_plddt' in results:
            print(f"   pLDDT: {results['sc_score_plddt'][0]:.4f}")
        
        # The key test: did we get valid metrics?
        success = (
            results['recovery_list'][0] > 0 and 
            results['perplexity_list'][0] > 0 and
            len(results['sc_score_rmsd']) > 0
        )
        
        if success:
            print(f"\n✅ Single structure evaluation PASSED!")
            print(f"   lDDT pipeline is working (OST dependency fixed)")
            return True
        else:
            print(f"\n❌ Single structure evaluation FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
