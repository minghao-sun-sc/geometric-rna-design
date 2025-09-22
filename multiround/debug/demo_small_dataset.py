#!/usr/bin/env python3
"""
Demonstration script for the small_dataset feature in dpo.bench.eval_full.py

This script shows how to use the small_dataset option for fast testing vs full evaluation.

Usage:
    cd /mnt/rna01/smh/projects/ribopo
    python multiround/debug/demo_small_dataset.py
"""

import os
import yaml
import tempfile
import subprocess
import time

def create_test_config(small_dataset: bool = True, passk_enabled: bool = True):
    """Create a test config with the specified options."""
    
    config = {
        "seed": 42,
        "device": "cuda",
        "paths": {
            "processed_pt": "data/processed.pt",
            "split_pt": "data/das_split.pt",
            "split_name": "test",
            "small_dataset": small_dataset,
            "checkpoints": [
                {
                    "name": "BASE_gRNAde_1state_das_temp0.1_samp1",
                    "path": "checkpoints/gRNAde_ARv1_1state_das.h5"
                }
            ]
        },
        "featurizer": {
            "split": "test",
            "radius": 0.0,
            "top_k": 32,
            "num_rbf": 32,
            "num_posenc": 32,
            "max_num_conformers": 1,
            "noise_scale": 0.0,
            "distance_eps": 0.001,
            "device": "cpu"
        },
        "model": {
            "name": "AutoregressiveMultiGNNv1",
            "node_in_dim": [15, 4],
            "node_h_dim": [128, 16],
            "edge_in_dim": [131, 3],
            "edge_h_dim": [64, 4],
            "num_layers": 4,
            "drop_rate": 0.5,
            "out_dim": 4
        },
        "eval": {
            "batch_size": 24,
            "num_workers": 6,
            "pin_memory": True,
            "out_dir": "multiround/eval_multiround/",
            "n_samples": 8,
            "temperature": 0.1,
            "use_relax": False,
            "use_lddt": True,
            "metrics": [
                "recovery",
                "perplexity"
            ],
            "save_designs": True,
            "wandb": {
                "enable": False  # Disable for demo
            }
        }
    }
    
    if passk_enabled:
        config["eval"]["passk"] = {
            "enable": True,
            "k_values": [1, 2, 4, 8],  # Reduced for demo
            "n_samples_passk": 16,     # Reduced for demo
            "thresholds": {
                "tm_score": [0.4, 0.5],
                "rmsd": [8.0, 4.0],
                "mfe": [-15.0, -20.0]
            },
            "collect_individual_metrics": True,
            "plot_distributions": False,  # Disabled for demo
            "save_passk_results": True,
            "passk_output_dir": "passk_analysis"
        }
    
    return config

def run_evaluation(config, test_name):
    """Run evaluation with the given config."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print('='*60)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Run evaluation
        cmd = [
            "python", "-m", "dpo.bench.eval_full",
            "--config", config_path,
            "--metrics", "recovery", "perplexity"
        ]
        
        print(f"📋 Running: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Evaluation completed successfully in {duration:.1f} seconds")
            
            # Extract key metrics from output
            output = result.stdout
            if "Recovery:" in output:
                recovery_line = [line for line in output.split('\n') if "Recovery:" in line][0]
                print(f"📊 {recovery_line.strip()}")
            if "Perplexity:" in output:
                perplexity_line = [line for line in output.split('\n') if "Perplexity:" in line][0]
                print(f"📊 {perplexity_line.strip()}")
            
            # Show dataset size info
            if "Small dataset mode" in output:
                dataset_line = [line for line in output.split('\n') if "Small dataset mode" in line][0]
                print(f"🔬 {dataset_line.strip()}")
            elif "Full dataset mode" in output:
                dataset_line = [line for line in output.split('\n') if "Full dataset mode" in line][0]
                print(f"📊 {dataset_line.strip()}")
            
            # Show pass@k info if available
            if "Pass@k analysis enabled" in output:
                passk_line = [line for line in output.split('\n') if "Pass@k analysis enabled" in line][0]
                print(f"🎯 {passk_line.strip()}")
            
        else:
            print(f"❌ Evaluation failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            
        return result.returncode == 0
        
    finally:
        # Clean up temporary config
        os.unlink(config_path)

def main():
    """Run demonstration of small_dataset feature."""
    print("🚀 Small Dataset Feature Demonstration")
    print("This demo shows the difference between small_dataset=true (fast) vs small_dataset=false (full)")
    
    if not os.path.exists("data/processed.pt"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected files: data/processed.pt, checkpoints/gRNAde_ARv1_1state_das.h5")
        return False
    
    # Test 1: Small dataset (fast testing)
    config_small = create_test_config(small_dataset=True, passk_enabled=True)
    success1 = run_evaluation(config_small, "Small Dataset Test (4 structures, with pass@k)")
    
    # Test 2: Small dataset without pass@k (fastest)
    config_small_no_passk = create_test_config(small_dataset=True, passk_enabled=False)
    success2 = run_evaluation(config_small_no_passk, "Small Dataset Test (4 structures, no pass@k)")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEMONSTRATION SUMMARY")
    print('='*60)
    
    test_results = [
        ("Small dataset with pass@k", success1),
        ("Small dataset without pass@k", success2),
    ]
    
    for test_name, success in test_results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n💡 Usage Tips:")
    print(f"   🔬 small_dataset: true  → Use 4 structures for fast testing")
    print(f"   📊 small_dataset: false → Use all 98 structures for full evaluation")
    print(f"   🎯 passk.enable: true   → Enable pass@k analysis (slower but comprehensive)")
    print(f"   ⚡ passk.enable: false  → Disable pass@k analysis (faster)")
    print(f"\n✨ Example commands:")
    print(f"   # Fast testing:")
    print(f"   python -m dpo.bench.eval_full --config your_config.yaml --metrics recovery perplexity")
    print(f"   # Full evaluation:")
    print(f"   python -m dpo.bench.eval_full --config your_config.yaml")
    
    return all(success for _, success in test_results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)