# multiround/debug/validation/test_data_validation.py
"""Validation tests for data integrity and format compliance."""

import os
import sys
import unittest
import json
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment
from utils.mock_data import setup_complete_mock_environment, MockDataset


class TestDataFormatValidation(unittest.TestCase):
    """Test data format validation and compliance."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_preference_pairs_format_validation(self):
        """Test preference pairs JSON format validation."""
        with TempDirectory() as temp_dir:
            # Create mock dataset
            dataset = MockDataset(temp_dir)
            pairs_dir = dataset.create_preference_pairs(n_pairs=5, margin_type="25")
            
            # Test train pairs format
            train_file = pairs_dir / "train.clean.jsonl"
            self.assertTrue(train_file.exists())
            
            # Validate JSONL format
            with open(train_file, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0, "Train file should not be empty")
                
                for i, line in enumerate(lines):
                    # Each line should be valid JSON
                    try:
                        pair = json.loads(line.strip())
                    except json.JSONDecodeError:
                        self.fail(f"Invalid JSON at line {i+1} in train file")
                    
                    # Validate required fields
                    required_fields = [
                        'backbone_id', 'chosen', 'rejected',
                        'chosen_metrics', 'rejected_metrics'
                    ]
                    
                    for field in required_fields:
                        self.assertIn(field, pair, f"Missing field '{field}' in pair {i+1}")
                    
                    # Validate field types
                    self.assertIsInstance(pair['backbone_id'], str)
                    self.assertIsInstance(pair['chosen'], str)
                    self.assertIsInstance(pair['rejected'], str)
                    self.assertIsInstance(pair['chosen_metrics'], dict)
                    self.assertIsInstance(pair['rejected_metrics'], dict)
                    
                    # Validate sequence format (RNA nucleotides)
                    valid_nucleotides = set('AUGC')
                    chosen_set = set(pair['chosen'])
                    rejected_set = set(pair['rejected'])
                    
                    self.assertTrue(
                        chosen_set.issubset(valid_nucleotides),
                        f"Invalid nucleotides in chosen sequence: {chosen_set - valid_nucleotides}"
                    )
                    self.assertTrue(
                        rejected_set.issubset(valid_nucleotides),
                        f"Invalid nucleotides in rejected sequence: {rejected_set - valid_nucleotides}"
                    )
                    
                    # Validate metrics format
                    for metrics_name, metrics in [('chosen_metrics', pair['chosen_metrics']), 
                                                   ('rejected_metrics', pair['rejected_metrics'])]:
                        expected_metrics = ['plddt', 'rmsd', 'mfe']
                        
                        for metric in expected_metrics:
                            self.assertIn(metric, metrics, f"Missing metric '{metric}' in {metrics_name}")
                            self.assertIsInstance(metrics[metric], (int, float), 
                                                f"Metric '{metric}' should be numeric")
    
    def test_test_dataset_format_validation(self):
        """Test test dataset format validation."""
        with TempDirectory() as temp_dir:
            # Create mock dataset
            dataset = MockDataset(temp_dir)
            test_file = dataset.create_test_dataset(n_structures=3)
            
            self.assertTrue(test_file.exists())
            
            # Validate JSON format
            with open(test_file, 'r') as f:
                try:
                    test_data = json.load(f)
                except json.JSONDecodeError:
                    self.fail("Test dataset file is not valid JSON")
            
            # Should be a list
            self.assertIsInstance(test_data, list)
            self.assertGreater(len(test_data), 0)
            
            # Validate each structure
            for i, structure in enumerate(test_data):
                required_fields = [
                    'id_list', 'sequence', 'coords_list', 'sec_struct_list',
                    'mask_coords', 'sasa_list', 'rfam_list', 'eq_class_list'
                ]
                
                for field in required_fields:
                    self.assertIn(field, structure, f"Missing field '{field}' in structure {i}")
                
                # Validate field types and formats
                self.assertIsInstance(structure['id_list'], list)
                self.assertIsInstance(structure['sequence'], str)
                self.assertIsInstance(structure['coords_list'], list)
                self.assertIsInstance(structure['sec_struct_list'], list)
                self.assertIsInstance(structure['mask_coords'], list)
                
                # Validate sequence format
                valid_nucleotides = set('AUGC')
                seq_set = set(structure['sequence'])
                self.assertTrue(
                    seq_set.issubset(valid_nucleotides),
                    f"Invalid nucleotides in sequence: {seq_set - valid_nucleotides}"
                )
                
                # Validate coordinates format
                for j, coords in enumerate(structure['coords_list']):
                    self.assertIsInstance(coords, list, f"Coords {j} should be a list")
                    self.assertEqual(len(coords), len(structure['sequence']),
                                   f"Coords length should match sequence length")
                    
                    for k, coord in enumerate(coords):
                        self.assertIsInstance(coord, list, f"Coord {k} should be a list")
                        self.assertEqual(len(coord), 3, f"Coord {k} should have 3 dimensions")
                        
                        for dim in coord:
                            self.assertIsInstance(dim, (int, float),
                                                f"Coordinate dimension should be numeric")
    
    def test_checkpoint_format_validation(self):
        """Test checkpoint format validation."""
        with TempDirectory() as temp_dir:
            # Create mock dataset
            dataset = MockDataset(temp_dir)
            checkpoint_file = dataset.create_model_checkpoint(round_num=1)
            
            self.assertTrue(checkpoint_file.exists())
            
            # Load checkpoint
            import torch
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
            except Exception as e:
                self.fail(f"Failed to load checkpoint: {e}")
            
            # Validate required fields
            required_fields = [
                'model_state_dict', 'optimizer_state_dict', 'epoch', 'round',
                'step', 'train_loss', 'val_loss', 'metrics'
            ]
            
            for field in required_fields:
                self.assertIn(field, checkpoint, f"Missing field '{field}' in checkpoint")
            
            # Validate field types
            self.assertIsInstance(checkpoint['model_state_dict'], dict)
            self.assertIsInstance(checkpoint['optimizer_state_dict'], dict)
            self.assertIsInstance(checkpoint['epoch'], int)
            self.assertIsInstance(checkpoint['round'], int)
            self.assertIsInstance(checkpoint['step'], int)
            self.assertIsInstance(checkpoint['metrics'], dict)
            
            # Validate numeric fields
            self.assertIsInstance(checkpoint['train_loss'], (int, float))
            self.assertIsInstance(checkpoint['val_loss'], (int, float))
            
            # Validate metrics
            expected_metrics = ['train_pref_acc', 'val_pref_acc', 'tm_mean', 'rmsd_mean', 'mfe_mean']
            for metric in expected_metrics:
                self.assertIn(metric, checkpoint['metrics'])
                self.assertIsInstance(checkpoint['metrics'][metric], (int, float))
    
    def test_data_path_validation(self):
        """Test data path validation and accessibility."""
        # Standard data paths that should exist in a complete setup
        expected_paths = [
            "data/pairs_margin25/by_das/clean/train.clean.jsonl",
            "data/pairs_margin25/by_das/clean/val.clean.jsonl",
            "data/pairs_margin125/by_das/clean/train.clean.jsonl",
            "data/pairs_margin125/by_das/clean/val.clean.jsonl"
        ]
        
        project_root = Path(__file__).parent.parent.parent.parent
        
        for path in expected_paths:
            full_path = project_root / path
            # Note: In testing environment, these files might not exist
            # This test documents the expected structure
            if full_path.exists():
                self.assertTrue(full_path.is_file(), f"Path should be a file: {path}")
                self.assertGreater(full_path.stat().st_size, 0, f"File should not be empty: {path}")


class TestDataIntegrityValidation(unittest.TestCase):
    """Test data integrity and consistency validation."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
    
    def test_preference_pairs_integrity(self):
        """Test preference pairs data integrity."""
        with TempDirectory() as temp_dir:
            # Create mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            
            # Test margin25 pairs
            train_file = mock_env["pairs_margin25"] / "train.clean.jsonl"
            val_file = mock_env["pairs_margin25"] / "val.clean.jsonl"
            
            # Load and validate pairs
            train_pairs = []
            with open(train_file, 'r') as f:
                for line in f:
                    train_pairs.append(json.loads(line.strip()))
            
            val_pairs = []
            with open(val_file, 'r') as f:
                for line in f:
                    val_pairs.append(json.loads(line.strip()))
            
            # Test data integrity
            self.assertGreater(len(train_pairs), 0, "Should have training pairs")
            self.assertGreater(len(val_pairs), 0, "Should have validation pairs")
            
            # Test that train and val don't overlap (different backbone IDs)
            train_ids = {pair['backbone_id'] for pair in train_pairs}
            val_ids = {pair['backbone_id'] for pair in val_pairs}
            
            overlap = train_ids & val_ids
            self.assertEqual(len(overlap), 0, f"Train and val should not overlap: {overlap}")
            
            # Test preference consistency (chosen should be better than rejected)
            for pair in train_pairs + val_pairs:
                chosen_metrics = pair['chosen_metrics']
                rejected_metrics = pair['rejected_metrics']
                
                # pLDDT: higher is better
                self.assertGreaterEqual(
                    chosen_metrics['plddt'], rejected_metrics['plddt'],
                    f"Chosen pLDDT should be >= rejected for {pair['backbone_id']}"
                )
                
                # RMSD: lower is better
                self.assertLessEqual(
                    chosen_metrics['rmsd'], rejected_metrics['rmsd'],
                    f"Chosen RMSD should be <= rejected for {pair['backbone_id']}"
                )
                
                # MFE: lower (more negative) is better
                self.assertLessEqual(
                    chosen_metrics['mfe'], rejected_metrics['mfe'],
                    f"Chosen MFE should be <= rejected for {pair['backbone_id']}"
                )
    
    def test_sequence_length_consistency(self):
        """Test sequence length consistency across data."""
        with TempDirectory() as temp_dir:
            # Create mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=5)
            
            # Load test dataset
            with open(mock_env["test_dataset"], 'r') as f:
                test_data = json.load(f)
            
            for structure in test_data:
                seq_len = len(structure['sequence'])
                
                # Coordinates should match sequence length
                for coords in structure['coords_list']:
                    self.assertEqual(
                        len(coords), seq_len,
                        f"Coords length should match sequence length for {structure['id_list'][0]}"
                    )
                
                # Mask should match sequence length
                self.assertEqual(
                    len(structure['mask_coords']), seq_len,
                    f"Mask length should match sequence length for {structure['id_list'][0]}"
                )
                
                # SASA should match sequence length
                for sasa in structure['sasa_list']:
                    self.assertEqual(
                        len(sasa), seq_len,
                        f"SASA length should match sequence length for {structure['id_list'][0]}"
                    )
    
    def test_metrics_range_validation(self):
        """Test that metrics are within expected ranges."""
        with TempDirectory() as temp_dir:
            # Create mock environment
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=5)
            
            train_file = mock_env["pairs_margin25"] / "train.clean.jsonl"
            
            with open(train_file, 'r') as f:
                for line in f:
                    pair = json.loads(line.strip())
                    
                    for metrics_name in ['chosen_metrics', 'rejected_metrics']:
                        metrics = pair[metrics_name]
                        
                        # pLDDT should be between 0 and 1
                        plddt = metrics['plddt']
                        self.assertGreaterEqual(plddt, 0.0, f"pLDDT should be >= 0")
                        self.assertLessEqual(plddt, 1.0, f"pLDDT should be <= 1")
                        
                        # RMSD should be positive
                        rmsd = metrics['rmsd']
                        self.assertGreater(rmsd, 0.0, f"RMSD should be > 0")
                        
                        # MFE should be negative (stable structures)
                        mfe = metrics['mfe']
                        self.assertLess(mfe, 0.0, f"MFE should be < 0 for stable structures")
    
    def test_margin_type_consistency(self):
        """Test consistency between different margin types."""
        with TempDirectory() as temp_dir:
            # Create mock environment with both margin types
            mock_env = setup_complete_mock_environment(temp_dir, n_pairs=10)
            
            # Load both margin types
            train25_file = mock_env["pairs_margin25"] / "train.clean.jsonl"
            train125_file = mock_env["pairs_margin125"] / "train.clean.jsonl"
            
            pairs25 = []
            with open(train25_file, 'r') as f:
                for line in f:
                    pairs25.append(json.loads(line.strip()))
            
            pairs125 = []
            with open(train125_file, 'r') as f:
                for line in f:
                    pairs125.append(json.loads(line.strip()))
            
            # Both should have similar structure
            self.assertGreater(len(pairs25), 0)
            self.assertGreater(len(pairs125), 0)
            
            # Sample pairs for comparison
            sample25 = pairs25[0]
            sample125 = pairs125[0]
            
            # Should have same required fields
            required_fields = ['backbone_id', 'chosen', 'rejected', 'chosen_metrics', 'rejected_metrics']
            
            for field in required_fields:
                self.assertIn(field, sample25)
                self.assertIn(field, sample125)
            
            # Metrics should have same structure
            for metrics_name in ['chosen_metrics', 'rejected_metrics']:
                metrics25_keys = set(sample25[metrics_name].keys())
                metrics125_keys = set(sample125[metrics_name].keys())
                self.assertEqual(
                    metrics25_keys, metrics125_keys,
                    f"Metrics structure should be consistent between margin types"
                )


class TestConfigDataConsistency(unittest.TestCase):
    """Test consistency between configuration and data paths."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.config_root = Path(__file__).parent.parent.parent / "config"
    
    def test_config_data_path_consistency(self):
        """Test that config files reference valid data paths."""
        experiments_dir = self.config_root / "experiments"
        
        for config_file in experiments_dir.glob("*.yaml"):
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check paths section
            if 'paths' in config:
                paths = config['paths']
                
                # Check pair paths
                for margin in ['pairs_margin25', 'pairs_margin125']:
                    if margin in paths:
                        pair_paths = paths[margin]
                        
                        for split in ['train', 'val']:
                            if split in pair_paths:
                                path = pair_paths[split]
                                # Path should be reasonable format
                                self.assertTrue(
                                    path.endswith('.jsonl'),
                                    f"Pair file should be .jsonl format: {path} in {config_file.name}"
                                )
                                self.assertTrue(
                                    'pairs_margin' in path,
                                    f"Path should contain margin info: {path} in {config_file.name}"
                                )
            
            # Check dynamic pair configs
            if 'multiround' in config and 'pair_configs' in config['multiround']:
                pair_configs = config['multiround']['pair_configs']
                
                for round_config in pair_configs.values():
                    if 'pairs' in round_config:
                        pairs = round_config['pairs']
                        
                        for split in ['train', 'val']:
                            if split in pairs:
                                path = pairs[split]
                                self.assertTrue(
                                    path.endswith('.jsonl'),
                                    f"Dynamic pair file should be .jsonl: {path} in {config_file.name}"
                                )


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)