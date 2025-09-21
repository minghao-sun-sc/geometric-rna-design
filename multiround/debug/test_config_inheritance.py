#!/usr/bin/env python3
"""
Test script to validate configuration inheritance for multiround experiments.

This script tests the three representative configurations to ensure they can be loaded
without AttributeError exceptions and that all required fields are present.
"""

import os
import sys
import yaml
import traceback
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def load_yaml_with_inheritance(config_path):
    """
    Load a YAML config file with inheritance support.
    Mimics the actual config loading mechanism used in training.
    """
    def resolve_path(path_str, base_dir):
        """Resolve relative paths relative to project root for inheritance"""
        if os.path.isabs(path_str):
            return path_str
        # For inheritance paths, resolve relative to project root, not config dir
        project_root = "/mnt/rna01/smh/projects/ribopo"
        return os.path.join(project_root, path_str)
    
    def merge_configs(base_config, override_config):
        """Recursively merge override config into base config"""
        if not isinstance(base_config, dict) or not isinstance(override_config, dict):
            return override_config
        
        result = base_config.copy()
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def load_with_inheritance(config_path, visited=None):
        """Load config with inheritance resolution"""
        if visited is None:
            visited = set()
        
        config_path = os.path.abspath(config_path)
        if config_path in visited:
            raise ValueError(f"Circular inheritance detected: {config_path}")
        
        visited.add(config_path)
        base_dir = os.path.dirname(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Handle inheritance
        if 'inherit_from' in config:
            parent_path = resolve_path(config['inherit_from'], base_dir)
            parent_config = load_with_inheritance(parent_path, visited.copy())
            # Remove inherit_from from current config before merging
            config_without_inherit = {k: v for k, v in config.items() if k != 'inherit_from'}
            config = merge_configs(parent_config, config_without_inherit)
        
        return config
    
    return load_with_inheritance(config_path)

def validate_required_fields(config, config_name):
    """
    Validate that all required fields are present in the configuration.
    Returns a list of missing or invalid fields.
    """
    issues = []
    
    # Required top-level fields
    required_fields = [
        'seed', 'device', 'loss_type', 'wandb', 'paths', 'tools',
        'featurizer', 'model', 'optimizer', 'scheduler', 'training', 'dpo'
    ]
    
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    # Check critical nested fields
    if 'dpo' in config:
        dpo_required = ['beta', 'sft_lambda', 'max_len']
        for field in dpo_required:
            if field not in config['dpo']:
                issues.append(f"Missing dpo.{field}")
    
    if 'paths' in config:
        path_required = ['processed_pt', 'split_pt', 'base_checkpoint', 'pairs']
        for field in path_required:
            if field not in config['paths']:
                issues.append(f"Missing paths.{field}")
    
    if 'featurizer' in config:
        feat_required = ['split', 'radius', 'top_k', 'num_rbf', 'num_posenc', 'max_num_conformers']
        for field in feat_required:
            if field not in config['featurizer']:
                issues.append(f"Missing featurizer.{field}")
    
    if 'model' in config:
        model_required = ['name', 'node_in_dim', 'node_h_dim', 'edge_in_dim', 'edge_h_dim', 'num_layers', 'out_dim']
        for field in model_required:
            if field not in config['model']:
                issues.append(f"Missing model.{field}")
    
    if 'optimizer' in config:
        opt_required = ['name', 'lr', 'weight_decay']
        for field in opt_required:
            if field not in config['optimizer']:
                issues.append(f"Missing optimizer.{field}")
    
    if 'scheduler' in config:
        sched_required = ['name', 'warmup_steps']
        for field in sched_required:
            if field not in config['scheduler']:
                issues.append(f"Missing scheduler.{field}")
    
    if 'training' in config:
        train_required = ['epochs', 'batch_size', 'grad_accum_steps']
        for field in train_required:
            if field not in config['training']:
                issues.append(f"Missing training.{field}")
    
    # Check multiround-specific fields if present
    if 'multiround' in config:
        mr_required = ['num_rounds', 'epochs_per_round']
        for field in mr_required:
            if field not in config['multiround']:
                issues.append(f"Missing multiround.{field}")
    
    return issues

def validate_learning_rates(config, config_name):
    """Validate that learning rates match expected values for DPO vs SimPO"""
    issues = []
    
    if 'optimizer' not in config or 'lr' not in config['optimizer']:
        return issues
    
    lr = config['optimizer']['lr']
    loss_type = config.get('loss_type', 'unknown')
    
    if loss_type == 'dpo':
        expected_lr = 0.00018  # 1.8e-4
        if abs(lr - expected_lr) > 1e-6:
            issues.append(f"DPO learning rate should be {expected_lr}, got {lr}")
    elif loss_type == 'simpo':
        expected_lr = 0.0004  # 4e-4
        if abs(lr - expected_lr) > 1e-6:
            issues.append(f"SimPO learning rate should be {expected_lr}, got {lr}")
    
    return issues

def test_config_loading(config_path, config_name):
    """Test loading a single configuration file"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Path: {config_path}")
    print(f"{'='*60}")
    
    success = True
    
    try:
        # Test if file exists
        if not os.path.exists(config_path):
            print(f"❌ FAIL: Config file does not exist: {config_path}")
            return False
        
        # Test loading with inheritance
        print("Loading configuration with inheritance...")
        config = load_yaml_with_inheritance(config_path)
        print("✅ Configuration loaded successfully")
        
        # Validate required fields
        print("Validating required fields...")
        field_issues = validate_required_fields(config, config_name)
        if field_issues:
            print(f"❌ Missing/invalid fields found:")
            for issue in field_issues:
                print(f"   - {issue}")
            success = False
        else:
            print("✅ All required fields present")
        
        # Validate learning rates
        print("Validating learning rates...")
        lr_issues = validate_learning_rates(config, config_name)
        if lr_issues:
            print(f"❌ Learning rate issues:")
            for issue in lr_issues:
                print(f"   - {issue}")
            success = False
        else:
            print("✅ Learning rates correct")
        
        # Display key configuration details
        print("\nKey Configuration Details:")
        print(f"  - Loss type: {config.get('loss_type', 'MISSING')}")
        print(f"  - Learning rate: {config.get('optimizer', {}).get('lr', 'MISSING')}")
        print(f"  - DPO beta: {config.get('dpo', {}).get('beta', 'MISSING')}")
        print(f"  - SFT lambda: {config.get('dpo', {}).get('sft_lambda', 'MISSING')}")
        if config.get('loss_type') == 'simpo':
            print(f"  - SimPO beta: {config.get('simpo', {}).get('beta', 'MISSING')}")
            print(f"  - SimPO gamma: {config.get('simpo', {}).get('gamma', 'MISSING')}")
        print(f"  - Pair margin: {config.get('paths', {}).get('pair_margin', 'MISSING')}")
        if 'multiround' in config:
            print(f"  - Num rounds: {config.get('multiround', {}).get('num_rounds', 'MISSING')}")
            print(f"  - Epochs per round: {config.get('multiround', {}).get('epochs_per_round', 'MISSING')}")
            print(f"  - Update reference: {config.get('multiround', {}).get('update_reference', 'MISSING')}")
        
        # Check inheritance chain
        print("\nInheritance Chain:")
        current_path = config_path
        chain = [os.path.basename(current_path)]
        try:
            with open(current_path, 'r') as f:
                current_config = yaml.safe_load(f) or {}
            while 'inherit_from' in current_config:
                parent_path = current_config['inherit_from']
                if not os.path.isabs(parent_path):
                    # Use project root for inheritance paths
                    project_root = "/mnt/rna01/smh/projects/ribopo"
                    parent_path = os.path.join(project_root, parent_path)
                chain.append(os.path.basename(parent_path))
                current_path = parent_path
                with open(current_path, 'r') as f:
                    current_config = yaml.safe_load(f) or {}
            
            print(f"  {' → '.join(chain)}")
        except Exception as e:
            print(f"  Could not trace inheritance chain: {e}")
        
        return success
        
    except Exception as e:
        print(f"❌ FAIL: Exception during loading:")
        print(f"   {type(e).__name__}: {e}")
        print(f"   Traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                print(f"     {line}")
        return False

def main():
    """Main test function"""
    print("Configuration Inheritance Validation Test")
    print("========================================")
    
    # Test configurations
    test_configs = [
        ("04_dpo_m25.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/04_dpo_m25.yaml"),
        ("03_simpo_m125.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/03_simpo_m125.yaml"),
        ("15_dpo_dynamic_margins.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/15_dpo_dynamic_margins.yaml"),
    ]
    
    results = {}
    
    # Test each configuration
    for config_name, config_path in test_configs:
        results[config_name] = test_config_loading(config_path, config_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for config_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {config_name}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Configuration inheritance is working correctly!")
        print("   - All configs load without AttributeError exceptions")
        print("   - All required fields are present")
        print("   - Learning rates are correctly set for DPO vs SimPO")
        print("   - Inheritance chains are properly resolved")
    else:
        print("\n⚠️  Configuration inheritance has issues that need to be fixed.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())