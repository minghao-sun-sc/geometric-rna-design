#!/usr/bin/env python3
"""
Minimal evaluation test to verify lDDT appears in results
Uses just 1-2 test structures for speed
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

# Add project to path
PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def run_mini_lddt_eval():
    print("🚀 Running Minimal lDDT Evaluation Test")
    print("=" * 70)
    
    # Create a minimal test config
    import yaml
    import tempfile
    
    # Create temporary config with just 1 test structure
    test_config = {
        'seed': 42,
        'device': 'cuda',
        'paths': {
            'processed_pt': 'data/processed.pt',
            'split_pt': 'data/das_split.pt',
            'split_name': 'test',
            'checkpoints': [
                {
                    'name': 'test_lddt_eval',
                    'path': 'checkpoints/gRNAde_ARv1_1state_das.h5'
                }
            ]
        },
        'featurizer': {
            'split': 'test',
            'radius': 0.0,
            'top_k': 32,
            'num_rbf': 32,
            'num_posenc': 32,
            'max_num_conformers': 1,
            'noise_scale': 0.0,
            'distance_eps': 0.001,
            'device': 'cpu'
        },
        'model': {
            'name': 'AutoregressiveMultiGNNv1',
            'node_in_dim': [15, 4],
            'node_h_dim': [128, 16],
            'edge_in_dim': [131, 3],
            'edge_h_dim': [64, 4],
            'num_layers': 4,
            'drop_rate': 0.5,
            'out_dim': 4
        },
        'eval': {
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': False,
            'out_dir': 'dpo/debug/lddt_test_output',
            'n_samples': 1,  # Just 1 sample for speed
            'temperature': 0.5,
            'use_relax': False,
            'use_lddt': True,  # CRITICAL: Enable lDDT
            'metrics': ['recovery', 'sc_rhofold'],  # Minimal metrics
            'save_designs': False,
            'wandb': {'enable': False}
        }
    }
    
    # Save temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    print(f"📝 Created test config: {config_path}")
    print(f"   - use_lddt: {test_config['eval']['use_lddt']}")
    print(f"   - n_samples: {test_config['eval']['n_samples']}")
    print(f"   - metrics: {test_config['eval']['metrics']}")
    
    # Import evaluation components
    from dpo.bench.eval_full import load_cfg, FullEvalDataset, eval_full_metrics
    
    try:
        # Load config
        cfg = load_cfg(config_path)
        
        # Load dataset - but only use first item for speed
        print("\n📊 Loading test dataset...")
        from src.data.data_utils import load_processed_pt
        import torch
        
        all_items = load_processed_pt(cfg.paths.processed_pt)
        _, _, test_indices = torch.load(cfg.paths.split_pt, map_location='cpu')
        
        # Use only first test item
        test_indices = list(map(int, test_indices))[:1]
        print(f"   Using {len(test_indices)} test structure(s) for speed")
        
        # Create minimal dataset
        ds = FullEvalDataset(
            cfg.paths.processed_pt,
            cfg.paths.split_pt,
            cfg.paths.split_name,
            cfg.featurizer,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Manually limit to 1 item
        ds.data_list = ds.data_list[:1]
        
        print(f"   Dataset ready with {len(ds)} item(s)")
        
        # Run evaluation
        print("\n🔬 Running evaluation with lDDT...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_dir = "dpo/debug/lddt_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        stats = eval_full_metrics(
            cfg, ds, 
            name="test_lddt",
            ckpt_path=cfg.paths.checkpoints[0].path,
            device=device,
            n_samples=1,
            temperature=0.5,
            metrics=['recovery', 'perplexity', 'sc_rhofold'],
            save_designs=False,
            output_dir=output_dir
        )
        
        print("\n📈 Results:")
        print("-" * 40)
        
        # Check if lDDT is in results
        if 'lddt' in stats:
            lddt_value = stats['lddt']
            print(f"✅ lDDT score found: {lddt_value:.4f}" if not np.isnan(lddt_value) else "⚠️ lDDT: NaN (likely file naming issue)")
            
            if np.isnan(lddt_value):
                print("\n💡 Debugging NaN lDDT:")
                print("   Possible causes:")
                print("   1. PDB file naming mismatch (e.g., 1CSL_1_B vs 1CSL_1_B-A.pdb)")
                print("   2. RhoFold failed to generate structure")
                print("   3. No matching native PDB files found")
        else:
            print("❌ lDDT not found in results!")
            print("   Available metrics:", list(stats.keys()))
        
        # Save results for inspection
        results_file = os.path.join(output_dir, "lddt_test_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy values to float for JSON
            json_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Clean up temp config
        os.unlink(config_path)
        
        return stats
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)
        return None

def check_lddt_in_json_output():
    """Check if previous evaluations included lDDT"""
    print("\n" + "=" * 70)
    print("📋 Checking Previous Evaluation Results for lDDT")
    print("-" * 40)
    
    eval_results_dir = "/mnt/rna01/smh/projects/ribopo/dpo/eval_results"
    if os.path.exists(eval_results_dir):
        json_files = [f for f in os.listdir(eval_results_dir) if f.endswith('.json')]
        
        if json_files:
            # Check most recent file
            latest_file = sorted(json_files)[-1]
            json_path = os.path.join(eval_results_dir, latest_file)
            
            print(f"   Checking: {latest_file}")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if lDDT is in any checkpoint results
            if 'per_checkpoint_results' in data:
                for result in data['per_checkpoint_results']:
                    if 'lddt' in result:
                        lddt_val = result['lddt']
                        print(f"   ✅ Found lDDT: {lddt_val:.4f}" if lddt_val == lddt_val else "   ⚠️ Found lDDT: NaN")
                        break
                else:
                    print(f"   ❌ No lDDT found in checkpoint results")
                    print(f"   Available metrics: {list(data['per_checkpoint_results'][0].keys()) if data['per_checkpoint_results'] else 'None'}")

if __name__ == "__main__":
    # First check existing results
    check_lddt_in_json_output()
    
    # Then run mini evaluation
    print("\n" + "=" * 70)
    stats = run_mini_lddt_eval()