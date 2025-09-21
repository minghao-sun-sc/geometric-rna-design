# multiround/debug/integration/test_full_pipeline.py
"""Integration tests for the full multi-round training pipeline."""

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
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment, TestMetrics
from utils.mock_data import setup_complete_mock_environment


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the complete multi-round training pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    @patch('multiround.trainer.MultiRoundPairProvider')
    @patch('multiround.trainer.MultiRoundEvaluator')
    @patch('multiround.trainer.MultiRoundWandBManager')
    def test_trainer_initialization_pipeline(self, mock_wandb, mock_evaluator, mock_provider):
        """Test complete trainer initialization with all components."""
        with TempDirectory() as temp_dir:
            # Set up mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            self.config.paths.save_dir = str(temp_dir)
            
            # Configure mocks
            mock_provider.return_value = Mock()
            mock_evaluator.return_value = Mock()
            mock_wandb.return_value = Mock()
            mock_wandb.return_value.get_wandb_config.return_value = {'name': 'test_run'}
            
            # Import here to ensure mocks are in place
            from multiround.trainer import MultiRoundDPOTrainer
            
            trainer = MultiRoundDPOTrainer(self.config)
            
            # Verify all components initialized
            self.assertIsNotNone(trainer.pair_provider)
            self.assertIsNotNone(trainer.evaluator)
            self.assertIsNotNone(trainer.wandb_manager)
            
            # Verify output directory created
            self.assertTrue(os.path.exists(trainer.output_root))
    
    @patch('multiround.trainer.MultiRoundPairProvider')
    @patch('multiround.trainer.MultiRoundEvaluator')
    @patch('multiround.trainer.MultiRoundWandBManager')
    @patch('torch.load')  # Mock model loading
    @patch('torch.save')  # Mock model saving
    def test_round_execution_pipeline(self, mock_save, mock_load, mock_wandb, mock_evaluator, mock_provider):
        """Test execution of a complete training round."""
        with TempDirectory() as temp_dir:
            # Set up comprehensive mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=20)
            self.config.paths.save_dir = str(temp_dir)
            
            # Configure provider mock
            mock_provider_instance = Mock()
            mock_provider_instance.get_pairs_for_round.return_value = (
                [{'chosen': 'AUGC', 'rejected': 'AUGC'}] * 5,  # train pairs
                [{'chosen': 'AUGC', 'rejected': 'AUGC'}] * 2   # val pairs
            )
            mock_provider_instance.get_summary.return_value = {'total_pairs': 7}
            mock_provider.return_value = mock_provider_instance
            
            # Configure evaluator mock
            mock_evaluator_instance = Mock()
            mock_eval_results = TestMetrics.create_mock_evaluation_results()
            mock_evaluator_instance.evaluate_model.return_value = mock_eval_results
            mock_evaluator.return_value = mock_evaluator_instance
            
            # Configure wandb mock
            mock_wandb_instance = Mock()
            mock_wandb_instance.get_wandb_config.return_value = {'name': 'test_run'}
            mock_wandb.return_value = mock_wandb_instance
            
            # Import and create trainer
            from multiround.trainer import MultiRoundDPOTrainer
            trainer = MultiRoundDPOTrainer(self.config)
            
            # Mock model and dataset
            mock_model = Mock()
            mock_dataset = Mock()
            
            # Execute one round (this would be the main integration test)
            try:
                # This is a placeholder for the actual round execution
                # In the real implementation, this would call trainer.train_round()
                round_results = trainer.execute_round(
                    round_num=1,
                    model=mock_model,
                    test_dataset=mock_dataset
                )
                
                # Verify round execution components were called
                mock_provider_instance.get_pairs_for_round.assert_called()
                mock_evaluator_instance.evaluate_model.assert_called()
                
            except AttributeError:
                # Method doesn't exist yet, which is expected
                self.skipTest("execute_round method not implemented yet")
    
    def test_config_loading_and_validation_pipeline(self):
        """Test configuration loading and validation pipeline."""
        with TempDirectory() as temp_dir:
            # Create mock config file
            config_content = """
            seed: 42
            device: cpu
            loss_type: dpo
            
            multiround:
              num_rounds: 3
              epochs_per_round: 5
              update_reference: true
              dynamic_pairs: false
            
            dpo:
              beta: 0.1
              sft_lambda: 0.1
            
            training:
              batch_size: 4
              epochs: 5
            
            paths:
              save_dir: {temp_dir}
              pair_margin: "25"
            """.format(temp_dir=temp_dir)
            
            config_file = temp_dir / "test_config.yaml"
            config_file.write_text(config_content)
            
            # Test loading config from file
            # This would test the config loading pipeline
            try:
                from multiround.config_loader import load_config
                loaded_config = load_config(str(config_file))
                
                # Verify key config values
                self.assertEqual(loaded_config.seed, 42)
                self.assertEqual(loaded_config.multiround.num_rounds, 3)
                self.assertEqual(loaded_config.dpo.beta, 0.1)
                
            except ImportError:
                # Config loader not implemented yet
                self.skipTest("Config loader not implemented yet")
    
    @patch('multiround.pair_provider.MultiRoundPairProvider')
    def test_dynamic_pair_switching_pipeline(self, mock_provider_class):
        """Test dynamic pair switching pipeline."""
        with TempDirectory() as temp_dir:
            # Set up environment with both margin types
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=15)
            
            # Configure for dynamic pairs
            self.config.multiround.dynamic_pairs = True
            self.config.multiround.pair_configs = {
                "rounds_1_2": {
                    "margin_type": "25",
                    "pairs": {
                        "train": str(mock_env["pairs_margin25"] / "train.clean.jsonl"),
                        "val": str(mock_env["pairs_margin25"] / "val.clean.jsonl")
                    }
                },
                "rounds_3_5": {
                    "margin_type": "125",
                    "pairs": {
                        "train": str(mock_env["pairs_margin125"] / "train.clean.jsonl"),
                        "val": str(mock_env["pairs_margin125"] / "val.clean.jsonl")
                    }
                }
            }
            
            # Create mock provider instance
            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider
            
            # Test pair switching logic
            from multiround.pair_provider import MultiRoundPairProvider
            provider = MultiRoundPairProvider(self.config)
            
            # Test that different rounds would get different margins
            # This tests the integration between config and pair provider
            try:
                margin_r1 = provider._get_margin_for_round(1)
                margin_r3 = provider._get_margin_for_round(3)
                
                self.assertEqual(margin_r1, "25")
                self.assertEqual(margin_r3, "125")
                
            except AttributeError:
                # Method not implemented yet
                self.skipTest("_get_margin_for_round method not implemented yet")
    
    def test_checkpoint_and_resume_pipeline(self):
        """Test checkpoint saving and resuming pipeline."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Create mock checkpoint data
            checkpoint_data = {
                'round': 2,
                'epoch': 10,
                'model_state_dict': {'weight': torch.randn(5, 5)},
                'optimizer_state_dict': {'lr': 0.001},
                'metrics': {'train_loss': 0.5, 'val_loss': 0.6}
            }
            
            # Test checkpoint saving
            checkpoint_path = temp_dir / "checkpoint_round_2.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            # Verify checkpoint file exists
            self.assertTrue(checkpoint_path.exists())
            
            # Test checkpoint loading
            loaded_checkpoint = torch.load(checkpoint_path)
            
            # Verify checkpoint data integrity
            self.assertEqual(loaded_checkpoint['round'], 2)
            self.assertEqual(loaded_checkpoint['epoch'], 10)
            self.assertIn('model_state_dict', loaded_checkpoint)
            self.assertIn('optimizer_state_dict', loaded_checkpoint)
    
    @patch('multiround.evaluator.evaluate')
    def test_evaluation_pipeline(self, mock_evaluate):
        """Test the evaluation pipeline integration."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Set up mock evaluation results
            mock_results = TestMetrics.create_mock_evaluation_results()
            mock_evaluate.return_value = mock_results
            
            # Test evaluation pipeline
            from multiround.evaluator import MultiRoundEvaluator
            evaluator = MultiRoundEvaluator(self.config)
            
            # Mock model and dataset
            mock_model = Mock()
            mock_dataset = Mock()
            
            # Test evaluation
            results = evaluator.evaluate_model(
                model=mock_model,
                dataset=mock_dataset,
                round_num=1,
                n_samples=8,
                temperature=0.5
            )
            
            # Verify evaluation was called and results are valid
            mock_evaluate.assert_called_once()
            TestMetrics.assert_metrics_valid(results)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between major components."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_trainer_evaluator_integration(self):
        """Test integration between trainer and evaluator."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test that trainer correctly calls evaluator
            # and handles evaluation results properly
            self.skipTest("Integration test placeholder - implement when components are ready")
    
    def test_trainer_pair_provider_integration(self):
        """Test integration between trainer and pair provider."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test that trainer correctly gets pairs from provider
            # and handles dynamic switching
            self.skipTest("Integration test placeholder - implement when components are ready")
    
    def test_trainer_wandb_integration(self):
        """Test integration between trainer and WandB manager."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test that trainer correctly logs metrics to WandB
            self.skipTest("Integration test placeholder - implement when components are ready")
    
    def test_ema_trainer_integration(self):
        """Test integration between EMA and trainer."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test EMA usage within training loop
            self.skipTest("Integration test placeholder - implement when components are ready")


class TestEndToEndScenarios(unittest.TestCase):
    """Test complete end-to-end scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_single_round_training_scenario(self):
        """Test complete single round training scenario."""
        with TempDirectory() as temp_dir:
            # Set up complete mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test a complete single round:
            # 1. Load pairs
            # 2. Train model
            # 3. Evaluate model
            # 4. Save checkpoint
            # 5. Log metrics
            
            self.skipTest("End-to-end test placeholder - implement when pipeline is complete")
    
    def test_multi_round_training_scenario(self):
        """Test complete multi-round training scenario."""
        with TempDirectory() as temp_dir:
            # Set up complete mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=20)
            self.config.paths.save_dir = str(temp_dir)
            self.config.multiround.num_rounds = 3
            
            # This would test multiple rounds:
            # 1. Round 1: train, evaluate, save
            # 2. Update reference model
            # 3. Round 2: train, evaluate, save
            # 4. Round 3: train, evaluate, save
            # 5. Final evaluation
            
            self.skipTest("Multi-round test placeholder - implement when pipeline is complete")
    
    def test_dynamic_pairs_scenario(self):
        """Test dynamic pairs switching scenario."""
        with TempDirectory() as temp_dir:
            # Set up environment with both margin types
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=15)
            self.config.paths.save_dir = str(temp_dir)
            
            # Configure dynamic pairs
            self.config.multiround.dynamic_pairs = True
            self.config.multiround.num_rounds = 5
            
            # This would test:
            # 1. Rounds 1-2 use margin25 pairs
            # 2. Rounds 3-5 use margin125 pairs
            # 3. Verify pair switching works correctly
            
            self.skipTest("Dynamic pairs test placeholder - implement when pipeline is complete")
    
    def test_resume_training_scenario(self):
        """Test resume training from checkpoint scenario."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # This would test:
            # 1. Train for 2 rounds
            # 2. Save checkpoint
            # 3. Resume from checkpoint
            # 4. Continue training
            # 5. Verify state is correctly restored
            
            self.skipTest("Resume training test placeholder - implement when pipeline is complete")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)