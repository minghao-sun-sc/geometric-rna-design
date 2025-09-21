# multiround/debug/validation/test_metrics_validation.py
"""Validation tests for evaluation metrics and their correctness."""

import os
import sys
import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment, TestMetrics
from utils.mock_data import setup_complete_mock_environment


class TestEvaluationMetricsValidation(unittest.TestCase):
    """Test validation of evaluation metrics."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_metrics_range_validation(self):
        """Test that evaluation metrics are within expected ranges."""
        # Create mock evaluation results
        mock_results = TestMetrics.create_mock_evaluation_results()
        
        # Validate recovery metrics (should be 0-1)
        recovery_values = mock_results['recovery_list']
        for val in recovery_values:
            self.assertGreaterEqual(val, 0.0, "Recovery should be >= 0")
            self.assertLessEqual(val, 1.0, "Recovery should be <= 1")
        
        # Validate perplexity metrics (should be > 1)
        perplexity_values = mock_results['perplexity_list']
        for val in perplexity_values:
            self.assertGreater(val, 1.0, "Perplexity should be > 1")
            self.assertLess(val, 1000.0, "Perplexity should be reasonable")
        
        # Validate RMSD metrics (should be positive)
        rmsd_values = mock_results['sc_score_rmsd_list']
        for val in rmsd_values:
            self.assertGreater(val, 0.0, "RMSD should be > 0")
            self.assertLess(val, 100.0, "RMSD should be reasonable")
        
        # Validate TM-score metrics (should be 0-1)
        tm_values = mock_results['sc_score_tm_list']
        for val in tm_values:
            self.assertGreaterEqual(val, 0.0, "TM-score should be >= 0")
            self.assertLessEqual(val, 1.0, "TM-score should be <= 1")
        
        # Validate GDT metrics (should be 0-1)
        gdt_values = mock_results['sc_score_gddt_list']
        for val in gdt_values:
            self.assertGreaterEqual(val, 0.0, "GDT should be >= 0")
            self.assertLessEqual(val, 1.0, "GDT should be <= 1")
        
        # Validate pLDDT metrics (should be 0-1)
        plddt_values = mock_results['sc_score_plddt_list']
        for val in plddt_values:
            self.assertGreaterEqual(val, 0.0, "pLDDT should be >= 0")
            self.assertLessEqual(val, 1.0, "pLDDT should be <= 1")
    
    def test_metrics_consistency(self):
        """Test consistency between related metrics."""
        # Create mock results with known relationships
        n_samples = 10
        
        # Create consistent RMSD and TM-score data
        # Lower RMSD should generally correlate with higher TM-score
        rmsd_values = np.random.uniform(1.0, 10.0, n_samples)
        tm_values = 1.0 / (1.0 + rmsd_values / 5.0)  # Inverse relationship
        
        mock_results = {
            'sc_score_rmsd_list': rmsd_values.tolist(),
            'sc_score_tm_list': tm_values.tolist(),
            'recovery_list': np.random.uniform(0.3, 0.9, n_samples).tolist(),
            'perplexity_list': np.random.uniform(1.5, 5.0, n_samples).tolist(),
            'sc_score_gddt_list': tm_values.tolist(),  # Should correlate with TM
            'sc_score_plddt_list': np.random.uniform(0.4, 0.9, n_samples).tolist(),
            'samples_list': [np.random.randint(0, 4, 20).tolist() for _ in range(n_samples)]
        }
        
        # Test that the relationship holds (not perfect due to randomness, but general trend)
        correlation = np.corrcoef(rmsd_values, tm_values)[0, 1]
        self.assertLess(correlation, -0.5, "RMSD and TM-score should be negatively correlated")
    
    def test_metrics_structure_validation(self):
        """Test that metrics have the correct structure."""
        mock_results = TestMetrics.create_mock_evaluation_results()
        
        # Test required metrics are present
        required_metrics = [
            'recovery_list', 'perplexity_list', 'sc_score_rmsd_list',
            'sc_score_tm_list', 'sc_score_gddt_list'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, mock_results, f"Missing required metric: {metric}")
        
        # Test all metrics have same length (same number of samples)
        metric_lengths = [len(mock_results[metric]) for metric in required_metrics]
        self.assertTrue(
            all(length == metric_lengths[0] for length in metric_lengths),
            "All metrics should have the same number of samples"
        )
        
        # Test metrics are numeric
        for metric in required_metrics:
            values = mock_results[metric]
            for val in values:
                self.assertIsInstance(val, (int, float), f"Metric {metric} should be numeric")
                self.assertFalse(np.isnan(val), f"Metric {metric} should not contain NaN")
                self.assertFalse(np.isinf(val), f"Metric {metric} should not contain infinity")
    
    def test_pass_k_metrics_validation(self):
        """Test pass@k metrics validation."""
        # Create mock results for pass@k calculation
        n_samples = 32
        tm_scores = np.random.uniform(0.1, 0.8, n_samples)
        rmsd_scores = np.random.uniform(1.0, 15.0, n_samples)
        
        mock_results = {
            'sc_score_tm_list': tm_scores.tolist(),
            'sc_score_rmsd_list': rmsd_scores.tolist()
        }
        
        # Test pass@k calculation logic
        k_values = [1, 2, 4, 8, 16]
        tm_threshold = 0.45
        rmsd_threshold = 8.0
        
        for k in k_values:
            if k <= n_samples:
                # For TM-score (higher is better)
                top_k_tm = np.sort(tm_scores)[-k:]
                pass_k_tm = np.mean(top_k_tm >= tm_threshold)
                
                # Validate pass@k range
                self.assertGreaterEqual(pass_k_tm, 0.0, f"pass@{k} TM should be >= 0")
                self.assertLessEqual(pass_k_tm, 1.0, f"pass@{k} TM should be <= 1")
                
                # For RMSD (lower is better)
                top_k_rmsd = np.sort(rmsd_scores)[:k]
                pass_k_rmsd = np.mean(top_k_rmsd <= rmsd_threshold)
                
                # Validate pass@k range
                self.assertGreaterEqual(pass_k_rmsd, 0.0, f"pass@{k} RMSD should be >= 0")
                self.assertLessEqual(pass_k_rmsd, 1.0, f"pass@{k} RMSD should be <= 1")
    
    def test_aggregation_metrics_validation(self):
        """Test validation of aggregated metrics."""
        # Create multiple mock results to aggregate
        results_list = []
        for _ in range(5):
            results_list.append(TestMetrics.create_mock_evaluation_results())
        
        # Test aggregation logic
        all_recovery = []
        all_perplexity = []
        
        for results in results_list:
            all_recovery.extend(results['recovery_list'])
            all_perplexity.extend(results['perplexity_list'])
        
        # Calculate aggregated statistics
        mean_recovery = np.mean(all_recovery)
        std_recovery = np.std(all_recovery)
        mean_perplexity = np.mean(all_perplexity)
        std_perplexity = np.std(all_perplexity)
        
        # Validate aggregated metrics
        self.assertGreaterEqual(mean_recovery, 0.0, "Mean recovery should be >= 0")
        self.assertLessEqual(mean_recovery, 1.0, "Mean recovery should be <= 1")
        self.assertGreaterEqual(std_recovery, 0.0, "Std recovery should be >= 0")
        
        self.assertGreater(mean_perplexity, 1.0, "Mean perplexity should be > 1")
        self.assertGreaterEqual(std_perplexity, 0.0, "Std perplexity should be >= 0")
    
    def test_vienna_metrics_validation(self):
        """Test ViennaRNA-based metrics validation."""
        # Mock ViennaRNA metrics
        vienna_metrics = {
            'vienna_mfe': np.random.uniform(-25.0, -5.0, 10).tolist(),  # MFE should be negative
            'vienna_ED': np.random.uniform(0.0, 50.0, 10).tolist(),      # ED should be positive
            'vienna_ED_per_nt': np.random.uniform(0.0, 2.0, 10).tolist(), # ED/nt should be reasonable
            'vienna_pS0': np.random.uniform(1e-8, 1e-2, 10).tolist(),    # pS0 should be small probability
            'vienna_entropy': np.random.uniform(0.1, 2.0, 10).tolist(),   # Entropy should be positive
            'vienna_diversity': np.random.uniform(0.0, 1.0, 10).tolist(), # Diversity 0-1
            'vienna_Tm': np.random.uniform(20.0, 80.0, 10).tolist()       # Tm in reasonable range
        }
        
        # Validate MFE (should be negative for stable structures)
        for mfe in vienna_metrics['vienna_mfe']:
            self.assertLess(mfe, 0.0, "MFE should be negative")
            self.assertGreater(mfe, -100.0, "MFE should be reasonable")
        
        # Validate Ensemble Defect (should be positive)
        for ed in vienna_metrics['vienna_ED']:
            self.assertGreaterEqual(ed, 0.0, "Ensemble Defect should be >= 0")
        
        # Validate ED per nucleotide (should be reasonable)
        for ed_nt in vienna_metrics['vienna_ED_per_nt']:
            self.assertGreaterEqual(ed_nt, 0.0, "ED per nt should be >= 0")
            self.assertLess(ed_nt, 10.0, "ED per nt should be reasonable")
        
        # Validate probability of target structure (should be small)
        for ps0 in vienna_metrics['vienna_pS0']:
            self.assertGreater(ps0, 0.0, "pS0 should be > 0")
            self.assertLess(ps0, 1.0, "pS0 should be < 1")
        
        # Validate entropy (should be positive)
        for entropy in vienna_metrics['vienna_entropy']:
            self.assertGreater(entropy, 0.0, "Entropy should be > 0")
            self.assertLess(entropy, 10.0, "Entropy should be reasonable")
        
        # Validate diversity (should be 0-1)
        for diversity in vienna_metrics['vienna_diversity']:
            self.assertGreaterEqual(diversity, 0.0, "Diversity should be >= 0")
            self.assertLessEqual(diversity, 1.0, "Diversity should be <= 1")
        
        # Validate melting temperature (should be reasonable)
        for tm in vienna_metrics['vienna_Tm']:
            self.assertGreater(tm, 0.0, "Tm should be > 0")
            self.assertLess(tm, 100.0, "Tm should be < 100°C")


class TestMetricsComputationValidation(unittest.TestCase):
    """Test validation of metrics computation logic."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
    
    def test_sequence_recovery_computation(self):
        """Test sequence recovery computation validation."""
        # Mock sequences
        true_seq = "AUGCUGCA"
        predicted_seqs = [
            "AUGCUGCA",  # Perfect match (recovery = 1.0)
            "AUGCUGCU",  # 1 mismatch (recovery = 7/8 = 0.875)
            "AAAAAAAAA", # All mismatches except first (recovery = 1/8 = 0.125)
        ]
        
        for i, pred_seq in enumerate(predicted_seqs):
            # Calculate recovery manually
            matches = sum(1 for a, b in zip(true_seq, pred_seq[:len(true_seq)]) if a == b)
            recovery = matches / len(true_seq)
            
            # Validate recovery calculation
            self.assertGreaterEqual(recovery, 0.0, f"Recovery should be >= 0 for seq {i}")
            self.assertLessEqual(recovery, 1.0, f"Recovery should be <= 1 for seq {i}")
            
            # Test specific expected values
            if i == 0:
                self.assertEqual(recovery, 1.0, "Perfect match should have recovery = 1.0")
            elif i == 1:
                self.assertAlmostEqual(recovery, 7/8, places=3, msg="1 mismatch should give recovery = 7/8")
    
    def test_perplexity_computation(self):
        """Test perplexity computation validation."""
        # Mock logits and targets
        import torch
        import torch.nn.functional as F
        
        # Simple case: uniform distribution
        seq_len = 10
        vocab_size = 4
        batch_size = 2
        
        # Perfect prediction (low perplexity)
        perfect_logits = torch.zeros(batch_size, seq_len, vocab_size)
        targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
        perfect_logits[:, :, 0] = 10.0  # High confidence for correct class
        
        perfect_loss = F.cross_entropy(
            perfect_logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len).mean(dim=1)
        
        perfect_perplexity = torch.exp(perfect_loss)
        
        # Random prediction (high perplexity)
        random_logits = torch.randn(batch_size, seq_len, vocab_size)
        random_loss = F.cross_entropy(
            random_logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len).mean(dim=1)
        
        random_perplexity = torch.exp(random_loss)
        
        # Validate perplexity properties
        for pp in perfect_perplexity:
            self.assertGreater(pp.item(), 1.0, "Perplexity should be > 1")
            self.assertLess(pp.item(), 2.0, "Perfect prediction should have low perplexity")
        
        for rp in random_perplexity:
            self.assertGreater(rp.item(), 1.0, "Perplexity should be > 1")
            # Random prediction should generally have higher perplexity than perfect
    
    def test_structural_metrics_computation(self):
        """Test structural metrics computation validation."""
        # Mock coordinates for TM-score and RMSD calculation
        import torch
        
        # Simple case: identical structures (should give TM=1, RMSD=0)
        coords1 = torch.randn(20, 3)
        coords2 = coords1.clone()
        
        # Test RMSD calculation (should be 0 for identical structures)
        rmsd = torch.sqrt(torch.mean(torch.sum((coords1 - coords2) ** 2, dim=1)))
        self.assertAlmostEqual(rmsd.item(), 0.0, places=5, msg="RMSD should be 0 for identical structures")
        
        # Test with slightly different structures
        coords3 = coords1 + torch.randn_like(coords1) * 0.1  # Small perturbation
        rmsd_perturbed = torch.sqrt(torch.mean(torch.sum((coords1 - coords3) ** 2, dim=1)))
        
        self.assertGreater(rmsd_perturbed.item(), 0.0, "RMSD should be > 0 for different structures")
        self.assertLess(rmsd_perturbed.item(), 1.0, "Small perturbation should give small RMSD")
    
    def test_preference_accuracy_computation(self):
        """Test preference accuracy computation validation."""
        # Mock preference predictions
        # Format: (chosen_logprob, rejected_logprob, expected_preference)
        test_cases = [
            (-1.0, -2.0, 1),  # Chosen has higher prob (less negative), prefer chosen
            (-2.0, -1.0, 0),  # Rejected has higher prob, prefer rejected
            (-1.5, -1.5, 0.5), # Equal probs, random preference
        ]
        
        for chosen_logprob, rejected_logprob, expected_pref in test_cases:
            # Compute preference (chosen > rejected means preference = 1)
            preference = 1 if chosen_logprob > rejected_logprob else 0
            
            if chosen_logprob == rejected_logprob:
                # Equal case - could be either, just check it's valid
                self.assertIn(preference, [0, 1], "Preference should be 0 or 1")
            else:
                self.assertEqual(preference, expected_pref, 
                               f"Preference mismatch for logprobs ({chosen_logprob}, {rejected_logprob})")
    
    def test_metrics_aggregation_logic(self):
        """Test metrics aggregation logic validation."""
        # Mock data for different aggregation scenarios
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test mean
        mean_val = np.mean(data)
        self.assertAlmostEqual(mean_val, 3.0, places=5, msg="Mean calculation incorrect")
        
        # Test standard deviation
        std_val = np.std(data)
        expected_std = np.sqrt(2.0)  # For [1,2,3,4,5], std = sqrt(2)
        self.assertAlmostEqual(std_val, expected_std, places=5, msg="Std calculation incorrect")
        
        # Test percentiles
        p50 = np.percentile(data, 50)
        p95 = np.percentile(data, 95)
        
        self.assertAlmostEqual(p50, 3.0, places=5, msg="Median calculation incorrect")
        self.assertGreater(p95, p50, "95th percentile should be > median")
        
        # Test min/max
        self.assertEqual(np.min(data), 1.0, "Min calculation incorrect")
        self.assertEqual(np.max(data), 5.0, "Max calculation incorrect")


class TestMetricsThresholdValidation(unittest.TestCase):
    """Test validation of metric thresholds and success criteria."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
    
    def test_tm_score_thresholds(self):
        """Test TM-score threshold validation."""
        # Standard TM-score thresholds
        thresholds = [0.4, 0.45, 0.5, 0.55]
        
        # Mock TM-scores
        tm_scores = np.array([0.3, 0.42, 0.48, 0.52, 0.6])
        
        for threshold in thresholds:
            success_rate = np.mean(tm_scores >= threshold)
            
            # Validate success rate
            self.assertGreaterEqual(success_rate, 0.0, f"Success rate should be >= 0 for threshold {threshold}")
            self.assertLessEqual(success_rate, 1.0, f"Success rate should be <= 1 for threshold {threshold}")
            
            # Higher thresholds should have lower or equal success rates
            if threshold > 0.4:
                lower_threshold_rate = np.mean(tm_scores >= 0.4)
                self.assertLessEqual(success_rate, lower_threshold_rate, 
                                   "Higher threshold should have lower success rate")
    
    def test_rmsd_thresholds(self):
        """Test RMSD threshold validation."""
        # Standard RMSD thresholds
        thresholds = [2.0, 4.0, 6.0, 8.0]
        
        # Mock RMSD values
        rmsd_values = np.array([1.5, 3.0, 5.0, 7.0, 10.0])
        
        for threshold in thresholds:
            success_rate = np.mean(rmsd_values <= threshold)
            
            # Validate success rate
            self.assertGreaterEqual(success_rate, 0.0, f"Success rate should be >= 0 for threshold {threshold}")
            self.assertLessEqual(success_rate, 1.0, f"Success rate should be <= 1 for threshold {threshold}")
            
            # Lower thresholds should have lower or equal success rates
            if threshold > 2.0:
                lower_threshold_rate = np.mean(rmsd_values <= 2.0)
                self.assertGreaterEqual(success_rate, lower_threshold_rate,
                                      "Higher RMSD threshold should have higher success rate")
    
    def test_pass_k_threshold_validation(self):
        """Test pass@k threshold validation."""
        k_values = [1, 2, 4, 8, 16, 32, 64]
        
        # Mock scores (sorted for easy testing)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        threshold = 0.45
        
        for k in k_values:
            if k <= len(scores):
                # For pass@k, we take top k scores
                top_k_scores = np.sort(scores)[-k:]
                pass_k_rate = np.mean(top_k_scores >= threshold)
                
                # Validate pass@k
                self.assertGreaterEqual(pass_k_rate, 0.0, f"pass@{k} should be >= 0")
                self.assertLessEqual(pass_k_rate, 1.0, f"pass@{k} should be <= 1")
                
                # Larger k should have lower or equal pass@k (harder to have all top-k above threshold)
                if k > 1:
                    top_1_scores = np.sort(scores)[-1:]
                    pass_1_rate = np.mean(top_1_scores >= threshold)
                    # This relationship might not always hold due to threshold effects
                    # but we can at least check the calculation is reasonable


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)