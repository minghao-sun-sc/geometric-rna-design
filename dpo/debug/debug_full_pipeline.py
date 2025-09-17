#!/usr/bin/env python3
"""
Debug script for end-to-end comprehensive evaluation pipeline testing.
Tests the full evaluation workflow with a small subset of data.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import tempfile
import yaml
import json
import traceback
from pathlib import Path
from types import SimpleNamespace as SN

# Add project root to path
sys.path.insert(0, '/mnt/rna01/smh/projects/ribopo')

# Import the enhanced evaluation functions
from dpo.bench.eval_benchmark import (
    ComprehensiveDataset,
    eval_comprehensive_checkpoint,
    load_cfg,
    _to_sn
)
from dpo.utils import set_seed

def create_minimal_config() -> SN:
    """Create a minimal configuration for testing."""
    config_dict = {
        'seed': 42,
        'device': 'cuda',
        'paths': {
            'processed_pt': 'data/processed.pt',
            'split_pt': 'data/das_split.pt', 
            'split_name': 'test',
            'checkpoints': [
                {
                    'name': 'test_checkpoint',
                    'path': 'checkpoints/gRNAde_ARv1_1state_das.h5'  # Base model
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
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': True,
            'out_dir': 'dpo/debug/test_output',
            'n_samples': 2,  # Small for testing
            'temperature': 0.5,
            'metrics': [
                'recovery',
                'perplexity',
                'vienna_mfe',  # Start with basic Vienna metrics
                # 'sc_eternafold',  # Add these if tools work
                # 'sc_rhofold',
                # 'diversity_3mer'
            ],
            'save_designs': True,
            'wandb': {
                'enable': False  # Disable for testing
            }
        }
    }
    
    return _to_sn(config_dict)


def check_data_availability(cfg: SN) -> bool:
    """Check if required data files exist."""
    print("Checking data file availability...")
    
    required_files = [
        cfg.paths.processed_pt,
        cfg.paths.split_pt
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    # Check checkpoint
    for ckpt in cfg.paths.checkpoints:
        if os.path.exists(ckpt.path):
            print(f"✓ Checkpoint: {ckpt.path}")
        else:
            print(f"✗ Checkpoint: {ckpt.path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_dataset_loading(cfg: SN):
    """Test dataset loading functionality."""
    print("\n" + "="*50)
    print("TESTING DATASET LOADING")
    print("="*50)
    
    try:
        # Test dataset creation
        print("Creating ComprehensiveDataset...")
        dataset = ComprehensiveDataset(
            cfg.paths.processed_pt,
            cfg.paths.split_pt,
            cfg.paths.split_name,
            cfg.featurizer,
            device='cpu'  # Use CPU for testing
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  - Dataset size: {len(dataset)}")
        
        # Test loading first few items
        print(f"\nTesting data loading...")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"  Item {i}:")
            print(f"    - ID: {item.gid}")
            print(f"    - Sequence length: {len(item.native_seq_str)}")
            print(f"    - Graph nodes: {item.graph.seq.size(0)}")
            print(f"    - Coords available: {len(item.raw_data['coords_list'])}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        traceback.print_exc()
        return None


def test_basic_metrics_only(cfg: SN, dataset):
    """Test basic metrics (recovery, perplexity) without external tools."""
    print("\n" + "="*50)
    print("TESTING BASIC METRICS ONLY")
    print("="*50)
    
    # Modify config to only use basic metrics
    cfg.eval.metrics = ['recovery', 'perplexity']
    cfg.eval.n_samples = 1  # Minimal sampling
    
    try:
        # Create test output directory
        test_out_dir = 'dpo/debug/basic_metrics_test'
        os.makedirs(test_out_dir, exist_ok=True)
        
        # Get first checkpoint
        ckpt = cfg.paths.checkpoints[0]
        
        print(f"Testing with checkpoint: {ckpt.name}")
        print(f"Metrics: {cfg.eval.metrics}")
        
        # Run evaluation
        results = eval_comprehensive_checkpoint(
            cfg, dataset, ckpt.name, ckpt.path, 'cpu', test_out_dir
        )
        
        print(f"✓ Basic metrics evaluation completed")
        print(f"Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_vienna_metrics(cfg: SN, dataset):
    """Test Vienna RNA metrics integration."""
    print("\n" + "="*50)
    print("TESTING VIENNA METRICS INTEGRATION")
    print("="*50)
    
    # Modify config to include Vienna metrics
    cfg.eval.metrics = ['recovery', 'perplexity', 'vienna_mfe', 'vienna_ensemble']
    cfg.eval.n_samples = 2
    
    try:
        # Create test output directory
        test_out_dir = 'dpo/debug/vienna_metrics_test'
        os.makedirs(test_out_dir, exist_ok=True)
        
        # Get first checkpoint
        ckpt = cfg.paths.checkpoints[0]
        
        print(f"Testing Vienna metrics with checkpoint: {ckpt.name}")
        print(f"Metrics: {cfg.eval.metrics}")
        
        # Run evaluation
        results = eval_comprehensive_checkpoint(
            cfg, dataset, ckpt.name, ckpt.path, 'cpu', test_out_dir
        )
        
        print(f"✓ Vienna metrics evaluation completed")
        print(f"Vienna-specific results:")
        
        vienna_metrics = [k for k in results.keys() if 'vienna' in k]
        if vienna_metrics:
            for metric in vienna_metrics:
                value = results[metric]
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        else:
            print(f"  ⚠ No Vienna metrics found in results")
        
        return True
        
    except Exception as e:
        print(f"✗ Vienna metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_pipeline(cfg: SN, dataset):
    """Test the full comprehensive pipeline."""
    print("\n" + "="*50)
    print("TESTING COMPREHENSIVE PIPELINE")
    print("="*50)
    
    # Use comprehensive metrics (but with small samples)
    cfg.eval.metrics = [
        'recovery',
        'perplexity', 
        'vienna_mfe',
        'vienna_ensemble',
        'diversity_3mer',
        # 'sc_eternafold',  # Uncomment if EternaFold is working
        # 'sc_rhofold',     # Uncomment if RhoFold is working (slow!)
    ]
    cfg.eval.n_samples = 2
    cfg.eval.save_designs = True
    
    try:
        # Create test output directory
        test_out_dir = 'dpo/debug/comprehensive_test'
        os.makedirs(test_out_dir, exist_ok=True)
        
        # Get first checkpoint
        ckpt = cfg.paths.checkpoints[0]
        
        print(f"Testing comprehensive evaluation:")
        print(f"  Checkpoint: {ckpt.name}")
        print(f"  Metrics: {cfg.eval.metrics}")
        print(f"  Samples: {cfg.eval.n_samples}")
        print(f"  Output: {test_out_dir}")
        
        # Run evaluation
        results = eval_comprehensive_checkpoint(
            cfg, dataset, ckpt.name, ckpt.path, 'cpu', test_out_dir
        )
        
        print(f"✓ Comprehensive evaluation completed")
        
        # Display results by category
        categories = {
            'Basic': ['recovery', 'perplexity', 'n_graphs'],
            'Vienna': [k for k in results.keys() if 'vienna' in k],
            'Diversity': [k for k in results.keys() if 'diversity' in k],
            'Structure': [k for k in results.keys() if any(x in k for x in ['sc_', 'rmsd', 'tm'])],
            'Other': []
        }
        
        # Classify remaining metrics
        all_shown = set()
        for cat_metrics in categories.values():
            all_shown.update(cat_metrics)
        
        categories['Other'] = [k for k in results.keys() if k not in all_shown]
        
        for category, metrics in categories.items():
            if metrics:
                print(f"\n{category} Metrics:")
                for metric in metrics:
                    value = results[metric]
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        # Check output files
        print(f"\nChecking output files...")
        output_files = list(Path(test_out_dir).rglob("*"))
        print(f"Generated {len(output_files)} output files:")
        for file_path in sorted(output_files)[:10]:
            print(f"  - {file_path.relative_to(test_out_dir)}")
        if len(output_files) > 10:
            print(f"  ... and {len(output_files) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("🧪 COMPREHENSIVE EVALUATION PIPELINE DEBUG")
    print("="*70)
    
    # Create test configuration
    print("Creating test configuration...")
    cfg = create_minimal_config()
    set_seed(cfg.seed)
    
    # Check data availability
    if not check_data_availability(cfg):
        print("\n❌ Required data files not available. Cannot proceed with testing.")
        print("Please ensure the following files exist:")
        print("  - data/processed.pt")
        print("  - data/das_split.pt")  
        print("  - checkpoints/gRNAde_ARv1_1state_das.h5")
        sys.exit(1)
    
    # Test dataset loading
    dataset = test_dataset_loading(cfg)
    if dataset is None:
        print("\n❌ Dataset loading failed. Cannot proceed with evaluation tests.")
        sys.exit(1)
    
    # Limit dataset size for testing
    if len(dataset) > 5:
        print(f"📊 Limiting dataset to 5 samples for testing (was {len(dataset)})")
        dataset.items = dataset.items[:5]
    
    # Run progressive tests
    tests = [
        ("Basic Metrics", test_basic_metrics_only),
        ("Vienna Metrics", test_vienna_metrics),
        ("Comprehensive Pipeline", test_comprehensive_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            success = test_func(cfg, dataset)
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE TESTING SUMMARY")
    print("="*70)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All tests passed ({passed}/{total})! Pipeline is ready.")
    elif passed > 0:
        print(f"\n⚠ Some tests passed ({passed}/{total}). Partial functionality available.")
    else:
        print(f"\n❌ All tests failed ({passed}/{total}). Check your setup.")
    
    print("\nNext steps:")
    if passed > 0:
        print("  - Run the full evaluation with your actual config")
        print("  - Command: python -m dpo.bench.eval_benchmark --config dpo/configs/experiments/full_eval_v10_checkpoints.yaml")
    else:
        print("  - Check debug output above for specific errors")
        print("  - Run individual debug scripts (debug_vienna_metrics.py, etc.)")


if __name__ == "__main__":
    main()