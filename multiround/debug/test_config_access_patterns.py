#!/usr/bin/env python3
"""
Test specific config access patterns that were previously failing.

This script tests the actual field access patterns used in training code
to ensure that the configs can be used without AttributeError exceptions.
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
    """Load YAML config with inheritance (same as test_config_inheritance.py)"""
    def resolve_path(path_str, base_dir):
        if os.path.isabs(path_str):
            return path_str
        project_root = "/mnt/rna01/smh/projects/ribopo"
        return os.path.join(project_root, path_str)
    
    def merge_configs(base_config, override_config):
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
        if visited is None:
            visited = set()
        
        config_path = os.path.abspath(config_path)
        if config_path in visited:
            raise ValueError(f"Circular inheritance detected: {config_path}")
        
        visited.add(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        if 'inherit_from' in config:
            parent_path = resolve_path(config['inherit_from'], os.path.dirname(config_path))
            parent_config = load_with_inheritance(parent_path, visited.copy())
            config_without_inherit = {k: v for k, v in config.items() if k != 'inherit_from'}
            config = merge_configs(parent_config, config_without_inherit)
        
        return config
    
    return load_with_inheritance(config_path)

def test_training_access_patterns(config, config_name):
    """Test the specific field access patterns used in training code"""
    print(f"\nTesting training access patterns for {config_name}...")
    
    errors = []
    
    # Test patterns that were causing AttributeError before
    test_patterns = [
        # Basic config access
        ("config['training']['batch_size']", lambda c: c['training']['batch_size']),
        ("config['training']['epochs']", lambda c: c['training']['epochs']),
        ("config['training']['grad_accum_steps']", lambda c: c['training']['grad_accum_steps']),
        
        # Optimizer access
        ("config['optimizer']['lr']", lambda c: c['optimizer']['lr']),
        ("config['optimizer']['weight_decay']", lambda c: c['optimizer']['weight_decay']),
        
        # DPO parameters
        ("config['dpo']['beta']", lambda c: c['dpo']['beta']),
        ("config['dpo']['sft_lambda']", lambda c: c['dpo']['sft_lambda']),
        ("config['dpo']['max_len']", lambda c: c['dpo']['max_len']),
        
        # Model architecture
        ("config['model']['name']", lambda c: c['model']['name']),
        ("config['model']['node_in_dim']", lambda c: c['model']['node_in_dim']),
        ("config['model']['out_dim']", lambda c: c['model']['out_dim']),
        
        # Featurizer
        ("config['featurizer']['split']", lambda c: c['featurizer']['split']),
        ("config['featurizer']['max_num_conformers']", lambda c: c['featurizer']['max_num_conformers']),
        
        # Paths
        ("config['paths']['base_checkpoint']", lambda c: c['paths']['base_checkpoint']),
        ("config['paths']['pairs']['train']", lambda c: c['paths']['pairs']['train']),
        
        # Scheduler
        ("config['scheduler']['name']", lambda c: c['scheduler']['name']),
        ("config['scheduler']['warmup_steps']", lambda c: c['scheduler']['warmup_steps']),
        
        # Multiround specific
        ("config['multiround']['num_rounds']", lambda c: c['multiround']['num_rounds']),
        ("config['multiround']['epochs_per_round']", lambda c: c['multiround']['epochs_per_round']),
        ("config['multiround']['update_reference']", lambda c: c['multiround']['update_reference']),
    ]
    
    # Test SimPO-specific patterns if applicable
    if config.get('loss_type') == 'simpo':
        test_patterns.extend([
            ("config['simpo']['beta']", lambda c: c['simpo']['beta']),
            ("config['simpo']['gamma']", lambda c: c['simpo']['gamma']),
        ])
    
    for pattern_name, accessor_func in test_patterns:
        try:
            value = accessor_func(config)
            print(f"  ✅ {pattern_name} = {value}")
        except Exception as e:
            errors.append(f"❌ {pattern_name}: {type(e).__name__}: {e}")
            print(f"  ❌ {pattern_name}: {type(e).__name__}: {e}")
    
    return errors

def test_config_conversion_to_namespace(config, config_name):
    """Test converting config to namespace-like object (common pattern)"""
    print(f"\nTesting namespace conversion for {config_name}...")
    
    try:
        # Test creating a simple namespace object (like argparse.Namespace)
        class ConfigNamespace:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigNamespace(value))
                    else:
                        setattr(self, key, value)
        
        ns = ConfigNamespace(config)
        
        # Test accessing nested attributes
        test_accesses = [
            ("ns.training.batch_size", lambda: ns.training.batch_size),
            ("ns.optimizer.lr", lambda: ns.optimizer.lr),
            ("ns.dpo.beta", lambda: ns.dpo.beta),
            ("ns.model.name", lambda: ns.model.name),
            ("ns.multiround.num_rounds", lambda: ns.multiround.num_rounds),
        ]
        
        errors = []
        for pattern_name, accessor_func in test_accesses:
            try:
                value = accessor_func()
                print(f"  ✅ {pattern_name} = {value}")
            except Exception as e:
                errors.append(f"❌ {pattern_name}: {type(e).__name__}: {e}")
                print(f"  ❌ {pattern_name}: {type(e).__name__}: {e}")
        
        return errors
        
    except Exception as e:
        error_msg = f"Failed to create namespace: {type(e).__name__}: {e}"
        print(f"  ❌ {error_msg}")
        return [error_msg]

def test_practical_training_initialization(config, config_name):
    """Test patterns used when initializing training components"""
    print(f"\nTesting training initialization patterns for {config_name}...")
    
    errors = []
    
    try:
        # Simulate optimizer creation
        optimizer_config = config['optimizer']
        optimizer_params = {
            'lr': optimizer_config['lr'],
            'weight_decay': optimizer_config['weight_decay'],
            'betas': optimizer_config.get('betas', [0.9, 0.999]),
            'eps': optimizer_config.get('eps', 1e-8),
        }
        print(f"  ✅ Optimizer params: {optimizer_params}")
        
        # Simulate DPO loss configuration
        dpo_config = config['dpo']
        loss_params = {
            'beta': dpo_config['beta'],
            'sft_lambda': dpo_config['sft_lambda'],
            'max_len': dpo_config.get('max_len'),
        }
        print(f"  ✅ DPO loss params: {loss_params}")
        
        # Simulate model creation
        model_config = config['model']
        model_params = {
            'name': model_config['name'],
            'node_in_dim': model_config['node_in_dim'],
            'node_h_dim': model_config['node_h_dim'],
            'edge_in_dim': model_config['edge_in_dim'],
            'edge_h_dim': model_config['edge_h_dim'],
            'num_layers': model_config['num_layers'],
            'out_dim': model_config['out_dim'],
        }
        print(f"  ✅ Model params: {model_params}")
        
        # Simulate multiround configuration
        multiround_config = config['multiround']
        multiround_params = {
            'num_rounds': multiround_config['num_rounds'],
            'epochs_per_round': multiround_config['epochs_per_round'],
            'update_reference': multiround_config['update_reference'],
            'eval_temperature': multiround_config.get('eval_temperature', 0.5),
        }
        print(f"  ✅ Multiround params: {multiround_params}")
        
        return []
        
    except Exception as e:
        error_msg = f"Training initialization failed: {type(e).__name__}: {e}"
        print(f"  ❌ {error_msg}")
        return [error_msg]

def test_single_config(config_path, config_name):
    """Test a single configuration comprehensively"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TEST: {config_name}")
    print(f"Path: {config_path}")
    print(f"{'='*80}")
    
    try:
        # Load config
        config = load_yaml_with_inheritance(config_path)
        print("✅ Configuration loaded successfully")
        
        all_errors = []
        
        # Test 1: Basic field access patterns
        errors1 = test_training_access_patterns(config, config_name)
        all_errors.extend(errors1)
        
        # Test 2: Namespace conversion
        errors2 = test_config_conversion_to_namespace(config, config_name)
        all_errors.extend(errors2)
        
        # Test 3: Practical initialization patterns
        errors3 = test_practical_training_initialization(config, config_name)
        all_errors.extend(errors3)
        
        # Summary for this config
        if all_errors:
            print(f"\n❌ {config_name} has {len(all_errors)} errors:")
            for error in all_errors:
                print(f"   {error}")
            return False
        else:
            print(f"\n✅ {config_name} passed all tests!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to load config {config_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Configuration Access Patterns Validation")
    print("========================================")
    
    test_configs = [
        ("04_dpo_m25.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/04_dpo_m25.yaml"),
        ("03_simpo_m125.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/03_simpo_m125.yaml"),
        ("15_dpo_dynamic_margins.yaml", "/mnt/rna01/smh/projects/ribopo/multiround/config/experiments/15_dpo_dynamic_margins.yaml"),
    ]
    
    results = {}
    
    for config_name, config_path in test_configs:
        results[config_name] = test_single_config(config_path, config_name)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for config_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {config_name}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL CONFIGS READY FOR TRAINING' if all_passed else '❌ CONFIGS NEED FIXES'}")
    
    if all_passed:
        print("\n🎉 All configurations are properly set up!")
        print("   - All field access patterns work")
        print("   - Namespace conversion works")
        print("   - Training initialization patterns work")
        print("   - No AttributeError exceptions expected during training")
    else:
        print("\n⚠️ Some configurations have issues that will cause training failures.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())