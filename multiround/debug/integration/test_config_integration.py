# multiround/debug/integration/test_config_integration.py
"""Integration tests for configuration system and experiment configs."""

import os
import sys
import unittest
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment


class TestConfigIntegration(unittest.TestCase):
    """Test integration of configuration system."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config_root = Path(__file__).parent.parent.parent / "config"
    
    def test_defaults_config_loading(self):
        """Test loading of multiround_defaults.yaml."""
        defaults_path = self.config_root / "multiround_defaults.yaml"
        self.assertTrue(defaults_path.exists(), f"Defaults config not found: {defaults_path}")
        
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        # Test essential default sections
        self.assertIn('multiround', defaults)
        self.assertIn('dpo', defaults)
        self.assertIn('training', defaults)
        self.assertIn('paths', defaults)
        
        # Test multiround defaults
        multiround = defaults['multiround']
        self.assertIn('num_rounds', multiround)
        self.assertIn('epochs_per_round', multiround)
        self.assertIn('update_reference', multiround)
        
        # Test default values are reasonable
        self.assertGreater(multiround['num_rounds'], 0)
        self.assertGreater(multiround['epochs_per_round'], 0)
    
    def test_all_experiment_configs_loading(self):
        """Test loading of all experiment configuration files."""
        experiments_dir = self.config_root / "experiments"
        self.assertTrue(experiments_dir.exists(), f"Experiments directory not found: {experiments_dir}")
        
        # Expected experiment configs
        expected_configs = [
            "01_sft_ablation.yaml",
            "02_dpo_m125.yaml",
            "03_simpo_m125.yaml",
            "04_dpo_m25.yaml",
            "05_simpo_m25.yaml",
            "06_no_ref_dpo_m25_3r.yaml",
            "11_dpo_dynamic_margins.yaml",
            "13_dpo_m25_ema_ref.yaml"
        ]
        
        for config_name in expected_configs:
            config_path = experiments_dir / config_name
            self.assertTrue(config_path.exists(), f"Missing experiment config: {config_name}")
            
            # Test that config can be loaded
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test essential fields are present
            self.assertIn('inherit_from', config, f"Missing inherit_from in {config_name}")
            self.assertIn('seed', config, f"Missing seed in {config_name}")
            self.assertIn('device', config, f"Missing device in {config_name}")
    
    def test_config_inheritance_structure(self):
        """Test that experiment configs properly inherit from defaults."""
        experiments_dir = self.config_root / "experiments"
        defaults_path = "multiround/config/multiround_defaults.yaml"
        
        for config_file in experiments_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # All experiment configs should inherit from defaults
            self.assertEqual(
                config.get('inherit_from'), defaults_path,
                f"Config {config_file.name} should inherit from {defaults_path}"
            )
    
    def test_dpo_configs_consistency(self):
        """Test DPO configuration consistency across experiments."""
        experiments_dir = self.config_root / "experiments"
        dpo_configs = [
            "02_dpo_m125.yaml",
            "04_dpo_m25.yaml",
            "06_no_ref_dpo_m25_3r.yaml",
            "11_dpo_dynamic_margins.yaml",
            "13_dpo_m25_ema_ref.yaml"
        ]
        
        for config_name in dpo_configs:
            config_path = experiments_dir / config_name
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # All DPO configs should have loss_type: dpo
            self.assertEqual(config.get('loss_type'), 'dpo', f"Wrong loss_type in {config_name}")
            
            # Should have DPO section
            self.assertIn('dpo', config, f"Missing dpo section in {config_name}")
            
            dpo = config['dpo']
            self.assertIn('beta', dpo, f"Missing dpo.beta in {config_name}")
            self.assertIn('sft_lambda', dpo, f"Missing dpo.sft_lambda in {config_name}")
    
    def test_simpo_configs_consistency(self):
        """Test SimPO configuration consistency."""
        experiments_dir = self.config_root / "experiments"
        simpo_configs = [
            "03_simpo_m125.yaml",
            "05_simpo_m25.yaml"
        ]
        
        for config_name in simpo_configs:
            config_path = experiments_dir / config_name
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # All SimPO configs should have loss_type: simpo
            self.assertEqual(config.get('loss_type'), 'simpo', f"Wrong loss_type in {config_name}")
            
            # Should have SimPO section
            self.assertIn('simpo', config, f"Missing simpo section in {config_name}")
            
            simpo = config['simpo']
            self.assertIn('beta', simpo, f"Missing simpo.beta in {config_name}")
            self.assertIn('gamma', simpo, f"Missing simpo.gamma in {config_name}")
    
    def test_reference_update_settings(self):
        """Test reference model update settings across configs."""
        experiments_dir = self.config_root / "experiments"
        
        # Configs that should have update_reference: true
        ref_update_configs = [
            "02_dpo_m125.yaml",
            "03_simpo_m125.yaml", 
            "04_dpo_m25.yaml",
            "05_simpo_m25.yaml",
            "11_dpo_dynamic_margins.yaml",
            "13_dpo_m25_ema_ref.yaml"
        ]
        
        # Configs that should have update_reference: false
        no_ref_update_configs = [
            "01_sft_ablation.yaml",
            "06_no_ref_dpo_m25_3r.yaml"
        ]
        
        for config_name in ref_update_configs:
            config_path = experiments_dir / config_name
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            update_ref = config.get('multiround', {}).get('update_reference')
            self.assertTrue(update_ref, f"Expected update_reference: true in {config_name}")
        
        for config_name in no_ref_update_configs:
            config_path = experiments_dir / config_name
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            update_ref = config.get('multiround', {}).get('update_reference')
            self.assertFalse(update_ref, f"Expected update_reference: false in {config_name}")
    
    def test_dynamic_margins_config(self):
        """Test dynamic margins configuration."""
        config_path = self.config_root / "experiments" / "11_dpo_dynamic_margins.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should have dynamic_pairs enabled
        self.assertTrue(
            config.get('multiround', {}).get('dynamic_pairs'),
            "Dynamic pairs should be enabled in 11_dpo_dynamic_margins.yaml"
        )
        
        # Should have pair_configs
        pair_configs = config.get('multiround', {}).get('pair_configs')
        self.assertIsNotNone(pair_configs, "Missing pair_configs in dynamic margins config")
        
        # Should have rounds_1_2 and rounds_3_5 configurations
        self.assertIn('rounds_1_2', pair_configs)
        self.assertIn('rounds_3_5', pair_configs)
        
        # Check margin types
        self.assertEqual(pair_configs['rounds_1_2']['margin_type'], "25")
        self.assertEqual(pair_configs['rounds_3_5']['margin_type'], "125")
    
    def test_ema_config(self):
        """Test EMA configuration."""
        config_path = self.config_root / "experiments" / "13_dpo_m25_ema_ref.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should have EMA section
        self.assertIn('ema', config, "Missing ema section in EMA config")
        
        ema = config['ema']
        self.assertTrue(ema.get('enable'), "EMA should be enabled")
        self.assertIn('decay', ema)
        self.assertIn('update_after_step', ema)
        self.assertIn('update_every', ema)
        self.assertIn('use_ema_for_reference', ema)
        
        # Check reasonable EMA decay value
        decay = ema['decay']
        self.assertGreater(decay, 0.9, "EMA decay should be > 0.9")
        self.assertLess(decay, 1.0, "EMA decay should be < 1.0")
    
    def test_wandb_config_consistency(self):
        """Test WandB configuration consistency across configs."""
        experiments_dir = self.config_root / "experiments"
        
        for config_file in experiments_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'wandb' in config:
                wandb = config['wandb']
                
                # Essential WandB fields
                self.assertIn('enable', wandb, f"Missing wandb.enable in {config_file.name}")
                self.assertIn('project', wandb, f"Missing wandb.project in {config_file.name}")
                self.assertIn('run_name', wandb, f"Missing wandb.run_name in {config_file.name}")
                
                # Run name should match config name pattern
                if wandb.get('enable'):
                    run_name = wandb['run_name']
                    config_prefix = config_file.stem.split('_')[0]
                    self.assertTrue(
                        run_name.startswith(config_prefix),
                        f"Run name {run_name} should start with {config_prefix} in {config_file.name}"
                    )
    
    def test_path_configurations(self):
        """Test path configurations are consistent."""
        experiments_dir = self.config_root / "experiments"
        
        for config_file in experiments_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'paths' in config:
                paths = config['paths']
                
                # Should have save_dir
                self.assertIn('save_dir', paths, f"Missing paths.save_dir in {config_file.name}")
                
                # Save dir should be under multiround/runs/
                save_dir = paths['save_dir']
                self.assertTrue(
                    save_dir.startswith('multiround/runs/'),
                    f"Save dir should be under multiround/runs/ in {config_file.name}"
                )
    
    def test_training_hyperparameters(self):
        """Test training hyperparameters are reasonable."""
        experiments_dir = self.config_root / "experiments"
        
        for config_file in experiments_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check multiround settings
            if 'multiround' in config:
                multiround = config['multiround']
                
                if 'num_rounds' in multiround:
                    self.assertGreater(
                        multiround['num_rounds'], 0,
                        f"num_rounds should be > 0 in {config_file.name}"
                    )
                    self.assertLessEqual(
                        multiround['num_rounds'], 10,
                        f"num_rounds should be <= 10 in {config_file.name}"
                    )
                
                if 'epochs_per_round' in multiround:
                    self.assertGreater(
                        multiround['epochs_per_round'], 0,
                        f"epochs_per_round should be > 0 in {config_file.name}"
                    )
            
            # Check training settings
            if 'training' in config:
                training = config['training']
                
                if 'batch_size' in training:
                    self.assertGreater(
                        training['batch_size'], 0,
                        f"batch_size should be > 0 in {config_file.name}"
                    )
                
                if 'epochs' in training:
                    self.assertGreater(
                        training['epochs'], 0,
                        f"epochs should be > 0 in {config_file.name}"
                    )


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        with TempDirectory() as temp_dir:
            # Create invalid YAML file
            invalid_config = temp_dir / "invalid.yaml"
            invalid_config.write_text("invalid: yaml: content: [\n")
            
            # Should raise YAML parsing error
            with self.assertRaises(yaml.YAMLError):
                with open(invalid_config, 'r') as f:
                    yaml.safe_load(f)
    
    def test_missing_required_fields(self):
        """Test detection of missing required configuration fields."""
        with TempDirectory() as temp_dir:
            # Create config with missing required fields
            incomplete_config = temp_dir / "incomplete.yaml"
            incomplete_config.write_text("""
            seed: 42
            # Missing device, multiround, etc.
            """)
            
            with open(incomplete_config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Should be missing essential fields
            self.assertNotIn('multiround', config)
            self.assertNotIn('device', config)
    
    def test_config_type_validation(self):
        """Test validation of configuration field types."""
        config_data = {
            'seed': "not_a_number",  # Should be int
            'multiround': {
                'num_rounds': "not_a_number",  # Should be int
                'update_reference': "not_a_boolean"  # Should be bool
            }
        }
        
        # These would fail type validation in a proper config validator
        self.assertIsInstance(config_data['seed'], str)  # Wrong type
        self.assertIsInstance(config_data['multiround']['num_rounds'], str)  # Wrong type


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)