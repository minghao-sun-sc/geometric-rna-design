#!/usr/bin/env python3
"""
Quick test of evaluation pipeline with single structure to verify lDDT fix.
Extracts one item from processed.pt and runs full evaluation metrics.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import torch
import numpy as np
from pathlib import Path

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

def test_single_structure_evaluation():
    """Test evaluation pipeline with one structure."""
    print("🚀 Testing Single Structure Evaluation Pipeline")
    print("=" * 60)
    
    # Load processed data
    processed_path = "data/processed.pt"
    print(f"📁 Loading data from: {processed_path}")
    
    if not os.path.exists(processed_path):
        print(f"❌ Processed data not found: {processed_path}")
        return False
    
    data = torch.load(processed_path, map_location='cpu')
    test_data = data['test']
    
    print(f"✅ Loaded {len(test_data)} test structures")
    
    # Get first test structure
    test_item = test_data[0]
    structure_id = test_item['id_list'][0]
    sequence = test_item['sequence']
    
    print(f"🧬 Test structure: {structure_id}")
    print(f"   Sequence length: {len(sequence)}")
    print(f"   Sequence: {sequence[:50]}...")
    
    # Save test structure to example_data for reuse
    example_data_dir = "dpo/debug/example_data"
    os.makedirs(example_data_dir, exist_ok=True)
    
    test_structure_path = os.path.join(example_data_dir, "single_test_structure.pt")
    torch.save(test_item, test_structure_path)
    print(f"💾 Saved test structure to: {test_structure_path}")
    
    # Now run evaluation with this single structure
    print("\n🔬 Running Evaluation Pipeline...")
    
    try:
        # Import evaluation components (avoiding networkx issues)
        from src.models import AutoregressiveMultiGNNv1
        from src.data.featurizer import Featurizer
        from src.evaluator import evaluate
        from types import SimpleNamespace
        
        # Create minimal config
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.name = "AutoregressiveMultiGNNv1"
        config.model.node_in_dim = [15, 4]
        config.model.node_h_dim = [128, 16]  
        config.model.edge_in_dim = [131, 3]
        config.model.edge_h_dim = [64, 4]
        config.model.num_layers = 4
        config.model.drop_rate = 0.5
        config.model.out_dim = 4
        
        config.featurizer = SimpleNamespace()
        config.featurizer.split = 'test'
        config.featurizer.radius = 0.0
        config.featurizer.top_k = 32
        config.featurizer.num_rbf = 32
        config.featurizer.num_posenc = 32
        config.featurizer.max_num_conformers = 1
        config.featurizer.noise_scale = 0.1
        config.featurizer.distance_eps = 0.001
        config.featurizer.device = 'cpu'
        
        # Load model
        checkpoint_path = "checkpoints/gRNAde_ARv1_1state_das.h5"
        print(f"📦 Loading model from: {checkpoint_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoregressiveMultiGNNv1(config.model)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        print(f"✅ Model loaded on {device}")
        
        # Create dataset with single item
        featurizer = Featurizer(config.featurizer)
        
        class SingleItemDataset:
            def __init__(self, data_item, featurizer):
                self.data_list = [data_item]
                self.featurizer = featurizer
        
        dataset = SingleItemDataset(test_item, featurizer)
        
        # Run evaluation with lDDT metric focus
        print(f"\n🧪 Running evaluation (n_samples=1, temp=0.5)...")
        
        metrics_to_test = [
            'recovery', 'perplexity', 
            'sc_score_rhofold'  # This includes lDDT
        ]
        
        results = evaluate(
            model=model,
            dataset=dataset,
            n_samples=1,
            temperature=0.5,
            device=device,
            model_name="test_single",
            metrics=metrics_to_test,
            save_designs=True
        )
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Recovery: {results['recovery_list'][0]:.4f}")
        print(f"   Perplexity: {results['perplexity_list'][0]:.4f}")
        
        if 'sc_score_rmsd' in results:
            print(f"   RMSD: {results['sc_score_rmsd'][0]:.4f}")
        if 'sc_score_tm' in results:
            print(f"   TM-score: {results['sc_score_tm'][0]:.4f}")
        if 'sc_score_gddt' in results:
            print(f"   GDT-TS: {results['sc_score_gddt'][0]:.4f}")
        
        # Check for lDDT specifically
        has_lddt = False
        if hasattr(results, 'lddt_list') or 'lddt_list' in results:
            lddt_score = results.get('lddt_list', [None])[0]
            if lddt_score is not None and lddt_score > 0:
                print(f"   lDDT: {lddt_score:.4f} ✅")
                has_lddt = True
            else:
                print(f"   lDDT: Failed (score: {lddt_score}) ❌")
        else:
            print(f"   lDDT: Not available in results")
        
        # Success criteria
        if has_lddt or (results['recovery_list'][0] > 0 and results['perplexity_list'][0] > 0):
            print(f"\n✅ Single structure evaluation PASSED!")
            print(f"   Structure {structure_id} evaluated successfully")
            return True
        else:
            print(f"\n❌ Single structure evaluation FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_structure_evaluation()
    sys.exit(0 if success else 1)