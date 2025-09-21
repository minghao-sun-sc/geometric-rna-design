# multiround/debug/unit/test_evaluator.py
"""Unit tests for MultiRoundEvaluator."""

import os
import sys
import unittest
import tempfile
import numpy as np
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

# Import the class to test
from multiround.evaluator import MultiRoundEvaluator


class TestMultiRoundEvaluator(unittest.TestCase):
    """Test cases for MultiRoundEvaluator."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization with minimal config."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Test basic properties
            self.assertEqual(evaluator.n_samples_eval, 2)
            self.assertEqual(evaluator.n_samples_final_eval, 4)
            self.assertEqual(evaluator.eval_temperature, 0.5)
            self.assertEqual(evaluator.final_eval_temperature, 0.1)
            self.assertFalse(evaluator.save_eval_data)
    
    def test_temperature_switching(self):
        """Test temperature switching logic for final rounds."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Test regular round temperature
            temp1 = evaluator.get_temperature_for_round(1, is_final=False)
            self.assertEqual(temp1, 0.5)
            
            temp2 = evaluator.get_temperature_for_round(3, is_final=False)
            self.assertEqual(temp2, 0.5)
            
            # Test final round temperature
            final_temp = evaluator.get_temperature_for_round(5, is_final=True)
            self.assertEqual(final_temp, 0.1)
    
    def test_sample_count_for_round(self):
        """Test sample count logic for different rounds."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Test regular round samples
            samples1 = evaluator.get_samples_for_round(1)
            self.assertEqual(samples1, 2)
            
            samples2 = evaluator.get_samples_for_round(2)
            self.assertEqual(samples2, 2)
            
            # Test final round samples (should use final eval setting)
            samples_final = evaluator.get_samples_for_round(5, is_final=True)
            self.assertEqual(samples_final, 4)
    
    @patch('multiround.evaluator.evaluate')
    def test_model_evaluation(self, mock_evaluate):
        """Test model evaluation with mocked evaluator."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Set up mock evaluation results
            mock_results = TestMetrics.create_mock_evaluation_results()
            mock_evaluate.return_value = mock_results
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Create mock model and dataset
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
            
            # Verify evaluation was called correctly
            mock_evaluate.assert_called_once()
            
            # Verify results structure
            TestMetrics.assert_metrics_valid(results)
    
    def test_pass_k_evaluation(self):
        """Test pass@k evaluation logic."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Create mock results for pass@k
            mock_results = {
                'sc_score_tm_list': [0.3, 0.5, 0.6, 0.4, 0.7],
                'sc_score_rmsd_list': [3.0, 1.5, 9.0, 6.0, 2.0]
            }
            
            # Test pass@k calculation
            pass_k_results = evaluator.calculate_pass_k_metrics(
                mock_results, 
                k_values=[1, 2, 4],
                tm_thresholds=[0.45],
                rmsd_thresholds=[8.0, 2.0]
            )
            
            # Verify pass@k results structure
            self.assertIn('pass@k_tm', pass_k_results)
            self.assertIn('pass@k_rmsd', pass_k_results)
            
            # Test specific threshold results
            tm_45_results = pass_k_results['pass@k_tm']['threshold_0.45']
            self.assertIn('k_1', tm_45_results)
            self.assertIn('k_2', tm_45_results)
    
    def test_round_evaluation_mode(self):
        """Test evaluation mode determination for different rounds."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Test which rounds should use full pass@k (k=64)
            self.assertTrue(evaluator.should_use_full_pass_k(1))
            self.assertFalse(evaluator.should_use_full_pass_k(2))
            self.assertTrue(evaluator.should_use_full_pass_k(3))
            self.assertFalse(evaluator.should_use_full_pass_k(4))
            self.assertTrue(evaluator.should_use_full_pass_k(5))
    
    def test_evaluation_metrics_selection(self):
        """Test metric selection for evaluation."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Configure specific metrics
            self.config.evaluation = TestConfig.create_minimal_config().evaluation if hasattr(TestConfig.create_minimal_config(), 'evaluation') else type('obj', (object,), {})()
            self.config.evaluation.metrics = ['recovery', 'perplexity', 'sc_score_rhofold']
            
            evaluator = MultiRoundEvaluator(self.config)
            
            metrics = evaluator.get_evaluation_metrics()
            self.assertIn('recovery', metrics)
            self.assertIn('perplexity', metrics)
            self.assertIn('sc_score_rhofold', metrics)
    
    def test_results_aggregation(self):
        """Test aggregation of evaluation results."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Create multiple mock results
            results_list = [
                {'recovery_list': [0.7, 0.8], 'perplexity_list': [2.0, 1.8]},
                {'recovery_list': [0.6, 0.9], 'perplexity_list': [2.2, 1.6]},
                {'recovery_list': [0.8, 0.7], 'perplexity_list': [1.9, 2.1]}
            ]
            
            # Test aggregation
            aggregated = evaluator.aggregate_results(results_list)
            
            # Check aggregated statistics
            self.assertIn('mean_recovery', aggregated)
            self.assertIn('std_recovery', aggregated)
            self.assertIn('mean_perplexity', aggregated)
            self.assertIn('std_perplexity', aggregated)
            
            # Verify calculations
            all_recoveries = [0.7, 0.8, 0.6, 0.9, 0.8, 0.7]
            expected_mean_recovery = np.mean(all_recoveries)
            self.assertAlmostEqual(aggregated['mean_recovery'], expected_mean_recovery, places=4)
    
    def test_save_evaluation_data(self):
        """Test saving evaluation data to files."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            self.config.multiround.save_eval_data = True
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Create mock results
            mock_results = TestMetrics.create_mock_evaluation_results()
            
            # Test saving
            save_path = evaluator.save_evaluation_results(
                results=mock_results,
                round_num=1,
                output_dir=temp_dir
            )
            
            # Verify file was created
            self.assertTrue(os.path.exists(save_path))


class TestEvaluatorErrorHandling(unittest.TestCase):
    """Test error handling in evaluator."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_missing_config_fields(self):
        """Test handling of missing config fields."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Remove multiround config
            delattr(self.config, 'multiround')
            
            # Should handle gracefully with defaults
            evaluator = MultiRoundEvaluator(self.config)
            
            # Should use default values
            self.assertEqual(evaluator.n_samples_eval, 8)  # default
            self.assertEqual(evaluator.eval_temperature, 0.5)  # default
    
    def test_invalid_pass_k_parameters(self):
        """Test handling of invalid pass@k parameters."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            evaluator = MultiRoundEvaluator(self.config)
            
            # Test with invalid results (empty)
            empty_results = {'sc_score_tm_list': []}
            
            pass_k_results = evaluator.calculate_pass_k_metrics(
                empty_results,
                k_values=[1, 2],
                tm_thresholds=[0.45]
            )
            
            # Should handle gracefully
            self.assertIsInstance(pass_k_results, dict)
    
    @patch('multiround.evaluator.evaluate')
    def test_evaluation_failure_handling(self, mock_evaluate):
        """Test handling of evaluation failures."""
        with TempDirectory() as temp_dir:
            self.config.paths.save_dir = str(temp_dir)
            
            # Make evaluate function raise an exception
            mock_evaluate.side_effect = Exception("Evaluation failed")
            
            evaluator = MultiRoundEvaluator(self.config)
            
            mock_model = Mock()
            mock_dataset = Mock()
            
            # Should handle evaluation failure gracefully
            with self.assertLogs() as log:
                results = evaluator.evaluate_model(
                    model=mock_model,
                    dataset=mock_dataset,
                    round_num=1
                )
            
            # Should return None or empty results on failure
            self.assertIsNone(results)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)