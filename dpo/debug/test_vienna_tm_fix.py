#!/usr/bin/env python3
"""
Test that Vienna melting temperature is now included in evaluation results
"""

import sys
import os
import json
import tempfile
import yaml
from datetime import datetime

PROJECT_PATH = "/mnt/rna01/smh/projects/ribopo"
sys.path.insert(0, PROJECT_PATH)

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

def test_vienna_tm_evaluation():
    """Run a minimal evaluation to test Vienna Tm calculation"""
    print("🧪 Testing Vienna Tm Fix in Evaluation Pipeline")
    print("=" * 70)
    
    # Create a minimal test config
    test_config = {
        'seed': 42,
        'device': 'cuda',
        'paths': {
            'processed_pt': 'data/processed.pt',
            'split_pt': 'data/das_split.pt',
            'split_name': 'test',
            'checkpoints': [{
                'name': 'test_vienna_tm',
                'path': 'checkpoints/gRNAde_ARv1_1state_das.h5'
            }]
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
            'out_dir': 'dpo/debug/vienna_tm_test_output',
            'n_samples': 2,  # Just 2 samples for speed
            'temperature': 0.5,
            'max_structures': 1,  # Limit to 1 structure for speed
            'metrics': ['recovery', 'vienna_mfe', 'vienna_ED'],  # Include Vienna metrics
            'save_designs': False,
            'wandb': {'enable': False}
        }
    }
    
    # Save temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        print(f"📝 Created test config: {config_path}")
        print(f"   Metrics: {test_config['eval']['metrics']}")
        print(f"   Max structures: {test_config['eval']['max_structures']}")
        print(f"   N samples: {test_config['eval']['n_samples']}")
        
        # Import and run evaluation
        from dpo.bench.eval_full import load_cfg, FullEvalDataset, eval_full_metrics
        import torch
        
        # Load config
        cfg = load_cfg(config_path)
        
        # Load minimal dataset
        print("\\n📊 Loading minimal test dataset...")
        ds = FullEvalDataset(
            cfg.paths.processed_pt,
            cfg.paths.split_pt,
            cfg.paths.split_name,
            cfg.featurizer,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Limit to 1 structure for speed
        ds.items = ds.items[:1]
        print(f"   Dataset ready with {len(ds)} structure(s)")
        
        # Run evaluation
        print("\\n🔬 Running evaluation with Vienna metrics...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_dir = "dpo/debug/vienna_tm_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        results = eval_full_metrics(
            cfg, ds,
            ckpt_name="test_vienna_tm",
            ckpt_path=cfg.paths.checkpoints[0].path,
            device=device,
            n_samples=2,
            temperature=0.5,
            metrics=['recovery', 'vienna_mfe', 'vienna_ED'],
            save_designs=False,
            output_dir=output_dir
        )
        
        print("\\n📈 Results:")
        print("-" * 40)
        
        # Check Vienna metrics
        vienna_metrics = {k: v for k, v in results.items() if 'vienna' in k.lower()}
        
        for metric, value in vienna_metrics.items():
            if metric == 'vienna_Tm':
                if value > 0:
                    print(f"✅ {metric}: {value:.1f} °C")
                else:
                    print(f"⚠️ {metric}: {value:.1f} (may be 0 if no valid calculations)")
            elif 'mfe' in metric.lower():
                print(f"✅ {metric}: {value:.2f} kcal/mol")
            else:
                print(f"✅ {metric}: {value:.4f}")
        
        # Check if vienna_Tm is present
        if 'vienna_Tm' in results:
            print(f"\\n🎉 SUCCESS: vienna_Tm is now included in evaluation results!")
            tm_value = results['vienna_Tm']
            if tm_value > 0:
                print(f"   Melting temperature: {tm_value:.1f} °C")
            else:
                print(f"   Note: Tm value is {tm_value:.1f} (may indicate calculation issues)")
        else:
            print(f"\\n❌ FAILURE: vienna_Tm still missing from results")
            print(f"   Available Vienna metrics: {list(vienna_metrics.keys())}")
        
        # Save results for inspection
        results_file = os.path.join(output_dir, "vienna_tm_test_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy values to float for JSON
            json_results = {k: float(v) if hasattr(v, 'item') else v for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        
        print(f"\\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"\\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Clean up temp config
        if os.path.exists(config_path):
            os.unlink(config_path)

if __name__ == "__main__":
    results = test_vienna_tm_evaluation()
    
    if results and 'vienna_Tm' in results:
        print("\\n" + "=" * 70)
        print("🎯 CONCLUSION: Vienna melting temperature fix is working!")
        print("   The evaluation pipeline now includes vienna_Tm in results.")
    else:
        print("\\n" + "=" * 70)
        print("⚠️ ISSUE: Need to investigate further why vienna_Tm is missing.")