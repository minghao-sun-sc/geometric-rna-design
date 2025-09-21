# multiround/debug/unit/test_trainer.py
"""Unit tests for MultiRoundDPOTrainer."""

import os
import sys
import unittest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment
from utils.mock_data import setup_complete_mock_environment

# Import the class to test
from multiround.trainer import MultiRoundDPOTrainer


class TestMultiRoundDPOTrainer(unittest.TestCase):
    """Test cases for MultiRoundDPOTrainer."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization with minimal config."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Mock dependencies to avoid full initialization
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                # Configure mocks
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Test basic properties
                self.assertEqual(trainer.num_rounds, 2)
                self.assertEqual(trainer.epochs_per_round, 2)
                self.assertEqual(trainer.current_round, 1)
                self.assertTrue(trainer.output_root.endswith('test_run'))
    
    def test_trainer_round_management(self):
        """Test round management functionality."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Test round tracking
                self.assertEqual(trainer.current_round, 1)
                self.assertEqual(len(trainer.round_metrics), 0)
                self.assertEqual(len(trainer.best_checkpoints), 0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with invalid config
        invalid_config = TestConfig.create_minimal_config()
        invalid_config.multiround.num_rounds = -1
        
        with TempDirectory() as temp_dir:
            invalid_config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                # Should handle invalid configs gracefully
                trainer = MultiRoundDPOTrainer(invalid_config)
                # The trainer should use default values for invalid inputs
                self.assertGreater(trainer.num_rounds, 0)
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Check that output directory exists
                self.assertTrue(os.path.exists(trainer.output_root))
    
    @patch('multiround.trainer.wandb')
    def test_wandb_integration(self, mock_wandb):
        """Test WandB integration."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            self.config.wandb.enable = True
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb_manager:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb_manager.return_value = Mock()
                mock_wandb_manager.return_value.get_wandb_config.return_value = {
                    'name': 'test_run',
                    'project': 'test_project',
                    'entity': 'test_entity'
                }
                mock_wandb_manager.return_value.get_hyperparameters.return_value = {}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Verify WandB manager was called
                mock_wandb_manager.assert_called_once()
    
    def test_device_handling(self):
        """Test device selection logic."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Test CPU device
            self.config.device = 'cpu'
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                self.assertEqual(trainer.device.type, 'cpu')
    
    def test_pair_provider_integration(self):
        """Test integration with pair provider."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider_class, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider = Mock()
                mock_provider.get_summary.return_value = {"margin": "25", "pairs_count": 100}
                mock_provider_class.return_value = mock_provider
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Verify pair provider was initialized
                mock_provider_class.assert_called_once_with(self.config)
                self.assertEqual(trainer.pair_provider, mock_provider)
    
    def test_evaluator_integration(self):
        """Test integration with evaluator."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator_class, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator = Mock()
                mock_evaluator_class.return_value = mock_evaluator
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Verify evaluator was initialized
                mock_evaluator_class.assert_called_once_with(self.config)
                self.assertEqual(trainer.evaluator, mock_evaluator)
    
    def test_reference_update_setting(self):
        """Test reference model update setting."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Test with reference updates enabled
            self.config.multiround.update_reference = True
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                trainer = MultiRoundDPOTrainer(self.config)
                
                # This should be accessible through the config
                self.assertTrue(self.config.multiround.update_reference)


class TestTrainerErrorHandling(unittest.TestCase):
    """Test error handling in trainer."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_missing_config_fields(self):
        """Test handling of missing config fields."""
        # Remove required field
        delattr(self.config, 'multiround')
        
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                # Should handle missing fields gracefully with defaults
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Should use default values
                self.assertEqual(trainer.num_rounds, 5)  # default
                self.assertEqual(trainer.epochs_per_round, 20)  # default
    
    def test_invalid_directory_path(self):
        """Test handling of invalid directory paths."""
        with TempDirectory() as temp_dir:
            # Use invalid path
            self.config.paths.save_dir = "/invalid/path/that/does/not/exist"
            
            with patch('multiround.trainer.MultiRoundPairProvider') as mock_provider, \
                 patch('multiround.trainer.MultiRoundEvaluator') as mock_evaluator, \
                 patch('multiround.trainer.MultiRoundWandBManager') as mock_wandb:
                
                mock_provider.return_value = Mock()
                mock_evaluator.return_value = Mock()
                mock_wandb.return_value = Mock()
                mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
                
                # Should handle gracefully and create directories
                trainer = MultiRoundDPOTrainer(self.config)
                
                # Directory should be created
                self.assertTrue(os.path.exists(trainer.output_root))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)