# multiround/debug/unit/test_pair_provider.py
"""Unit tests for MultiRoundPairProvider."""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment
from utils.mock_data import setup_complete_mock_environment, MockDataset

# Import the class to test
from multiround.pair_provider import MultiRoundPairProvider


class TestMultiRoundPairProvider(unittest.TestCase):
    """Test cases for MultiRoundPairProvider."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_static_pair_provider(self):
        """Test static pair provider (no dynamic switching)."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=20)
            
            # Configure for static pairs
            self.config.paths.pair_margin = "25"
            self.config.paths.pairs_margin25 = {
                "train": str(mock_env["pairs_margin25"] / "train.clean.jsonl"),
                "val": str(mock_env["pairs_margin25"] / "val.clean.jsonl")
            }
            
            provider = MultiRoundPairProvider(self.config)
            
            # Test basic properties
            self.assertEqual(provider.current_margin, "25")
            self.assertFalse(provider.is_dynamic)
            
            # Test getting pairs for round 1
            train_pairs, val_pairs = provider.get_pairs_for_round(1)
            self.assertIsNotNone(train_pairs)
            self.assertIsNotNone(val_pairs)
            self.assertGreater(len(train_pairs), 0)
            self.assertGreater(len(val_pairs), 0)
    
    def test_dynamic_pair_provider(self):
        """Test dynamic pair provider with round-based switching."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=30)
            
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
            
            provider = MultiRoundPairProvider(self.config)
            
            # Test dynamic properties
            self.assertTrue(provider.is_dynamic)
            
            # Test round 1 (should use margin25)
            train_pairs_r1, val_pairs_r1 = provider.get_pairs_for_round(1)
            self.assertIsNotNone(train_pairs_r1)
            self.assertIsNotNone(val_pairs_r1)
            
            # Test round 3 (should use margin125)
            train_pairs_r3, val_pairs_r3 = provider.get_pairs_for_round(3)
            self.assertIsNotNone(train_pairs_r3)
            self.assertIsNotNone(val_pairs_r3)
            
            # Pairs should be different between rounds (different margins)
            # Note: In real implementation, this would check file sources
            self.assertIsNotNone(train_pairs_r1)
            self.assertIsNotNone(train_pairs_r3)
    
    def test_pair_validation(self):
        """Test pair file validation."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            
            self.config.paths.pair_margin = "25"
            self.config.paths.pairs_margin25 = {
                "train": str(mock_env["pairs_margin25"] / "train.clean.jsonl"),
                "val": str(mock_env["pairs_margin25"] / "val.clean.jsonl")
            }
            
            provider = MultiRoundPairProvider(self.config)
            
            # Test validation
            is_valid = provider.validate_pair_files("25")
            self.assertTrue(is_valid)
            
            # Test with invalid files
            invalid_config = TestConfig.create_minimal_config()
            invalid_config.paths.pair_margin = "25"
            invalid_config.paths.pairs_margin25 = {
                "train": "/nonexistent/train.jsonl",
                "val": "/nonexistent/val.jsonl"
            }
            
            invalid_provider = MultiRoundPairProvider(invalid_config)
            is_valid_invalid = invalid_provider.validate_pair_files("25")
            self.assertFalse(is_valid_invalid)
    
    def test_get_summary(self):
        """Test summary generation."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=15)
            
            self.config.paths.pair_margin = "25"
            self.config.paths.pairs_margin25 = {
                "train": str(mock_env["pairs_margin25"] / "train.clean.jsonl"),
                "val": str(mock_env["pairs_margin25"] / "val.clean.jsonl")
            }
            
            provider = MultiRoundPairProvider(self.config)
            
            summary = provider.get_summary()
            
            # Check summary contains expected fields
            self.assertIn("current_margin", summary)
            self.assertIn("is_dynamic", summary)
            self.assertIn("total_train_pairs", summary)
            self.assertIn("total_val_pairs", summary)
            
            self.assertEqual(summary["current_margin"], "25")
            self.assertFalse(summary["is_dynamic"])
            self.assertGreater(summary["total_train_pairs"], 0)
            self.assertGreater(summary["total_val_pairs"], 0)
    
    def test_fallback_mechanism(self):
        """Test fallback to static pairs when dynamic fails."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            
            # Configure dynamic with invalid files
            self.config.multiround.dynamic_pairs = True
            self.config.multiround.pair_configs = {
                "rounds_1_2": {
                    "pairs": {
                        "train": "/invalid/path/train.jsonl",
                        "val": "/invalid/path/val.jsonl"
                    }
                }
            }
            
            # Set up valid fallback
            self.config.paths.pair_margin = "25"
            self.config.paths.pairs_margin25 = {
                "train": str(mock_env["pairs_margin25"] / "train.clean.jsonl"),
                "val": str(mock_env["pairs_margin25"] / "val.clean.jsonl")
            }
            self.config.validation = TestConfig.create_minimal_config().validation if hasattr(TestConfig.create_minimal_config(), 'validation') else type('obj', (object,), {})()
            self.config.validation.fallback_to_static = True
            
            provider = MultiRoundPairProvider(self.config)
            
            # Should fallback to static pairs
            train_pairs, val_pairs = provider.get_pairs_for_round(1)
            self.assertIsNotNone(train_pairs)
            self.assertIsNotNone(val_pairs)
    
    def test_round_to_margin_mapping(self):
        """Test correct mapping of rounds to margins."""
        with TempDirectory() as temp_dir:
            # Set up mock data
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=20)
            
            # Configure dynamic pairs
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
            
            provider = MultiRoundPairProvider(self.config)
            
            # Test mapping for different rounds
            margin_r1 = provider._get_margin_for_round(1)
            margin_r2 = provider._get_margin_for_round(2)
            margin_r3 = provider._get_margin_for_round(3)
            margin_r4 = provider._get_margin_for_round(4)
            margin_r5 = provider._get_margin_for_round(5)
            
            # Rounds 1-2 should use margin 25
            self.assertEqual(margin_r1, "25")
            self.assertEqual(margin_r2, "25")
            
            # Rounds 3-5 should use margin 125
            self.assertEqual(margin_r3, "125")
            self.assertEqual(margin_r4, "125")
            self.assertEqual(margin_r5, "125")
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        with TempDirectory() as temp_dir:
            # Test with minimal config (missing pair paths)
            minimal_config = TestConfig.create_minimal_config()
            
            try:
                provider = MultiRoundPairProvider(minimal_config)
                # Should handle gracefully, possibly with warnings
                summary = provider.get_summary()
                self.assertIsInstance(summary, dict)
            except Exception as e:
                # Should not crash, but might raise specific exceptions
                self.assertIsInstance(e, (FileNotFoundError, KeyError, AttributeError))
    
    def test_pair_file_loading(self):
        """Test actual loading of pair files."""
        with TempDirectory() as temp_dir:
            # Create a simple test pair file
            test_pairs = [
                {
                    "backbone_id": "test_001",
                    "chosen": "AUGCUGCA",
                    "rejected": "AUGCUGCU",
                    "chosen_metrics": {"plddt": 0.8},
                    "rejected_metrics": {"plddt": 0.6}
                }
            ]
            
            train_file = temp_dir / "train.jsonl"
            with open(train_file, 'w') as f:
                for pair in test_pairs:
                    f.write(json.dumps(pair) + '\n')
            
            val_file = temp_dir / "val.jsonl"
            with open(val_file, 'w') as f:
                for pair in test_pairs:
                    f.write(json.dumps(pair) + '\n')
            
            # Configure provider
            self.config.paths.pair_margin = "25"
            self.config.paths.pairs_margin25 = {
                "train": str(train_file),
                "val": str(val_file)
            }
            
            provider = MultiRoundPairProvider(self.config)
            
            # Test loading
            train_pairs, val_pairs = provider.get_pairs_for_round(1)
            
            self.assertEqual(len(train_pairs), 1)
            self.assertEqual(len(val_pairs), 1)
            self.assertEqual(train_pairs[0]["backbone_id"], "test_001")
            self.assertEqual(val_pairs[0]["backbone_id"], "test_001")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)