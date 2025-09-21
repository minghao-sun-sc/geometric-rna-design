# multiround/debug/unit/test_wandb_manager.py
"""Unit tests for MultiRoundWandBManager."""

import os
import sys
import unittest
import tempfile
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

# Import the class to test
from multiround.wandb_manager import MultiRoundWandBManager


class TestMultiRoundWandBManager(unittest.TestCase):
    """Test cases for MultiRoundWandBManager."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    @patch('multiround.wandb_manager.wandb')
    def test_wandb_manager_initialization_enabled(self, mock_wandb):
        """Test WandB manager initialization when enabled."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Test basic properties
        self.assertTrue(manager.enabled)
        self.assertEqual(manager.project, "test")
        self.assertEqual(manager.entity, "test")
        self.assertEqual(manager.run_name, "test_run")
    
    def test_wandb_manager_initialization_disabled(self):
        """Test WandB manager initialization when disabled."""
        self.config.wandb.enable = False
        
        manager = MultiRoundWandBManager(self.config)
        
        # Test disabled state
        self.assertFalse(manager.enabled)
    
    @patch('multiround.wandb_manager.wandb')
    def test_run_initialization(self, mock_wandb):
        """Test WandB run initialization."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Mock wandb.init return value
        mock_run = Mock()
        mock_run.name = "test_run_123"
        mock_run.id = "abc123"
        mock_wandb.init.return_value = mock_run
        
        # Initialize run
        manager.init_run(round_num=1)
        
        # Verify wandb.init was called
        mock_wandb.init.assert_called_once()
        
        # Check call arguments
        call_args = mock_wandb.init.call_args
        self.assertEqual(call_args[1]['project'], "test")
        self.assertEqual(call_args[1]['entity'], "test")
        self.assertIn('name', call_args[1])
    
    @patch('multiround.wandb_manager.wandb')
    def test_hyperparameters_logging(self, mock_wandb):
        """Test hyperparameters configuration for WandB."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Get hyperparameters
        hyperparams = manager.get_hyperparameters()
        
        # Check essential hyperparameters are included
        self.assertIn('device', hyperparams)
        self.assertIn('seed', hyperparams)
        self.assertIn('multiround', hyperparams)
        self.assertIn('dpo', hyperparams)
        self.assertIn('training', hyperparams)
        
        # Check multiround specific parameters
        self.assertIn('num_rounds', hyperparams['multiround'])
        self.assertIn('epochs_per_round', hyperparams['multiround'])
        self.assertIn('update_reference', hyperparams['multiround'])
    
    @patch('multiround.wandb_manager.wandb')
    def test_config_generation(self, mock_wandb):
        """Test WandB config generation."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        wandb_config = manager.get_wandb_config()
        
        # Check required config fields
        self.assertIn('name', wandb_config)
        self.assertIn('project', wandb_config)
        self.assertIn('entity', wandb_config)
        self.assertIn('tags', wandb_config)
        self.assertIn('notes', wandb_config)
        
        # Check tags include multiround identifier
        self.assertIn('multiround', wandb_config['tags'])
    
    @patch('multiround.wandb_manager.wandb')
    def test_metrics_logging(self, mock_wandb):
        """Test metrics logging functionality."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Mock wandb run
        mock_run = Mock()
        mock_wandb.run = mock_run
        
        # Test logging training metrics
        train_metrics = {
            'train_loss': 0.5,
            'train_pref_acc': 0.75,
            'learning_rate': 0.001
        }
        
        manager.log_metrics(train_metrics, step=100, prefix='train')
        
        # Verify log was called when enabled
        if manager.enabled:
            mock_wandb.log.assert_called()
    
    @patch('multiround.wandb_manager.wandb')
    def test_round_metrics_logging(self, mock_wandb):
        """Test round-specific metrics logging."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Mock wandb run
        mock_run = Mock()
        mock_wandb.run = mock_run
        
        # Test logging round evaluation metrics
        eval_metrics = {
            'recovery_mean': 0.65,
            'tm_score_mean': 0.42,
            'rmsd_mean': 3.5,
            'pass@8_tm_0.45': 0.3
        }
        
        manager.log_round_metrics(
            round_num=2,
            eval_metrics=eval_metrics,
            train_metrics={'final_loss': 0.4}
        )
        
        # Should have logged with round prefix
        if manager.enabled:
            mock_wandb.log.assert_called()
    
    @patch('multiround.wandb_manager.wandb')
    def test_artifact_logging(self, mock_wandb):
        """Test artifact logging (checkpoints, configs, etc.)."""
        self.config.wandb.enable = True
        
        with TempDirectory() as temp_dir:
            manager = MultiRoundWandBManager(self.config)
            
            # Create mock artifact file
            artifact_path = temp_dir / "checkpoint.pt"
            artifact_path.write_text("mock checkpoint data")
            
            # Mock wandb artifact
            mock_artifact = Mock()
            mock_wandb.Artifact.return_value = mock_artifact
            
            # Test artifact logging
            manager.log_artifact(
                file_path=str(artifact_path),
                artifact_name="checkpoint_round_2",
                artifact_type="model"
            )
            
            if manager.enabled:
                mock_wandb.Artifact.assert_called()
                mock_artifact.add_file.assert_called_with(str(artifact_path))
    
    def test_wandb_disabled_behavior(self):
        """Test that operations work correctly when WandB is disabled."""
        self.config.wandb.enable = False
        
        manager = MultiRoundWandBManager(self.config)
        
        # All operations should work without errors when disabled
        manager.init_run(round_num=1)
        manager.log_metrics({'test': 1.0}, step=1)
        manager.log_round_metrics(1, {'test': 1.0}, {'test': 1.0})
        
        # Should not raise any exceptions
        self.assertFalse(manager.enabled)
    
    @patch('multiround.wandb_manager.wandb')
    def test_run_naming_convention(self, mock_wandb):
        """Test run naming convention for different rounds."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Test round-specific naming
        config1 = manager.get_wandb_config(round_num=1)
        config3 = manager.get_wandb_config(round_num=3)
        
        # Names should be different for different rounds
        self.assertNotEqual(config1['name'], config3['name'])
        
        # Both should contain round information
        self.assertIn('round', config1['name'].lower())
        self.assertIn('round', config3['name'].lower())
    
    @patch('multiround.wandb_manager.wandb')
    def test_tags_and_metadata(self, mock_wandb):
        """Test tags and metadata generation."""
        self.config.wandb.enable = True
        
        # Add some specific config for testing
        self.config.multiround.dynamic_pairs = True
        self.config.loss_type = "dpo"
        
        manager = MultiRoundWandBManager(self.config)
        
        wandb_config = manager.get_wandb_config()
        
        # Check that relevant tags are included
        tags = wandb_config['tags']
        self.assertIn('multiround', tags)
        self.assertIn('dpo', tags)
        
        # Check dynamic pairs tag if enabled
        if hasattr(self.config.multiround, 'dynamic_pairs') and self.config.multiround.dynamic_pairs:
            self.assertIn('dynamic_pairs', tags)


class TestWandBManagerErrorHandling(unittest.TestCase):
    """Test error handling in WandB manager."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_missing_config_fields(self):
        """Test handling of missing WandB config fields."""
        # Remove wandb config
        delattr(self.config, 'wandb')
        
        # Should handle gracefully with defaults
        manager = MultiRoundWandBManager(self.config)
        
        # Should default to disabled
        self.assertFalse(manager.enabled)
    
    @patch('multiround.wandb_manager.wandb')
    def test_wandb_initialization_failure(self, mock_wandb):
        """Test handling of WandB initialization failures."""
        self.config.wandb.enable = True
        
        # Make wandb.init raise an exception
        mock_wandb.init.side_effect = Exception("WandB init failed")
        
        manager = MultiRoundWandBManager(self.config)
        
        # Should handle init failure gracefully
        with self.assertLogs() as log:
            manager.init_run(round_num=1)
        
        # Should log the error but not crash
        self.assertTrue(any("WandB" in msg for msg in log.output))
    
    @patch('multiround.wandb_manager.wandb')
    def test_logging_with_connection_issues(self, mock_wandb):
        """Test logging behavior with connection issues."""
        self.config.wandb.enable = True
        
        manager = MultiRoundWandBManager(self.config)
        
        # Make wandb.log raise an exception
        mock_wandb.log.side_effect = Exception("Connection error")
        
        # Should handle logging errors gracefully
        metrics = {'test_metric': 1.0}
        manager.log_metrics(metrics, step=1)
        
        # Should not crash the program
        self.assertTrue(manager.enabled)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)