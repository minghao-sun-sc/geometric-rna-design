#!/usr/bin/env python
"""Verify HPO setup is correct."""

import os
import sys
import yaml
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_config(config_path):
    """Check if config file is valid."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check essential fields
        assert 'model' in cfg, "Missing 'model' section"
        assert 'training' in cfg, "Missing 'training' section"
        assert 'hpo' in cfg, "Missing 'hpo' section"
        assert 'search_space' in cfg['hpo'], "Missing 'hpo.search_space' section"
        
        # Check algorithm
        algo = cfg.get('loss_type', 'unknown')
        print(f"  - Algorithm: {algo}")
        
        # Check search space
        ss = cfg['hpo']['search_space']
        if algo == 'simpo':
            print(f"  - Beta range: {ss['beta_min']:.3f} - {ss['beta_max']:.3f}")
            print(f"  - Gamma range: {ss['gamma_min']:.3f} - {ss['gamma_max']:.3f}")
        else:
            print(f"  - Beta range: {ss['beta_min']:.3f} - {ss['beta_max']:.3f}")
        print(f"  - LR range: {ss['lr_min']:.2e} - {ss['lr_max']:.2e}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False

def main():
    """Run verification checks."""
    print("="*60)
    print("HPO Setup Verification")
    print("="*60)
    
    all_good = True
    
    # Check main script
    print("\n1. Main HPO Script:")
    all_good &= check_file_exists("dpo/hpo/optuna_search.py", "Optuna search script")
    
    # Check configs
    print("\n2. HPO Configs:")
    for i in range(1, 7):
        config_path = f"dpo/hpo/optuna_{i}.yaml"
        if check_file_exists(config_path, f"Config {i}"):
            check_config(config_path)
        else:
            all_good = False
    
    # Check SLURM scripts
    print("\n3. SLURM Scripts:")
    all_good &= check_file_exists("dpo/scripts/hpo_optuna.slurm", "SLURM array job script")
    all_good &= check_file_exists("dpo/scripts/run_hpo_single.sh", "Single HPO test script")
    all_good &= check_file_exists("dpo/scripts/submit_all_hpo.sh", "Submit all script")
    
    # Check trainer modifications
    print("\n4. Trainer Support:")
    try:
        with open("dpo/trainer.py", 'r') as f:
            content = f.read()
        if 'max_steps' in content and 'best_metrics' in content:
            print("✓ Trainer has max_steps and best_metrics support")
        else:
            print("✗ Trainer missing required HPO support")
            all_good = False
    except:
        print("✗ Could not check trainer.py")
        all_good = False
    
    # Check directories
    print("\n5. Directories:")
    os.makedirs("dpo/hpo/optuna_results", exist_ok=True)
    print("✓ Results directory: dpo/hpo/optuna_results/")
    os.makedirs("logs", exist_ok=True)
    print("✓ Logs directory: logs/")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ All checks passed! HPO system is ready.")
        print("\nTo start HPO:")
        print("  - Test single config: bash dpo/scripts/run_hpo_single.sh 1")
        print("  - Submit all jobs: bash dpo/scripts/submit_all_hpo.sh")
    else:
        print("❌ Some checks failed. Please fix issues before running HPO.")
    print("="*60)

if __name__ == "__main__":
    main()