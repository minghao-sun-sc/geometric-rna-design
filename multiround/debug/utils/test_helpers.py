# multiround/debug/utils/test_helpers.py
"""Common testing utilities and helper functions for multiround testing."""

import os
import sys
import tempfile
import shutil
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()


class TestConfig:
    """Test configuration management."""
    
    @staticmethod
    def create_minimal_config():
        """Create minimal config for testing."""
        return SimpleNamespace(
            device='cpu',
            seed=42,
            multiround=SimpleNamespace(
                num_rounds=2,
                epochs_per_round=2,
                update_reference=True,
                eval_temperature=0.5,
                final_eval_temperature=0.1,
                n_samples_eval=2,
                n_samples_final_eval=4,
                save_eval_data=False,
                reference_update=SimpleNamespace(
                    enabled=True,
                    strategy='pass_k_selection',
                    pass_k_threshold=0.45,
                    candidates=SimpleNamespace(
                        top_train_pref_acc=2,
                        top_val_pref_acc=2,
                        last_epochs=5
                    ),
                    dev_test_size=24,
                    pass_k=8,
                    random_seed=42
                ),
                ema=SimpleNamespace(
                    enabled=False,
                    decay=0.9999,
                    update_after_step=100,
                    update_every=10,
                    reference_strategy='none'
                )
            ),
            wandb=SimpleNamespace(
                enable=False,
                project="test",
                entity="test",
                run_name="test_run"
            ),
            dpo=SimpleNamespace(
                beta=0.1,
                sft_lambda=0.1,
                label_smoothing=0.0
            ),
            training=SimpleNamespace(
                epochs=2,
                batch_size=2,
                grad_accum_steps=1,
                learning_rate=1e-4,
                warmup_steps=100
            ),
            paths=SimpleNamespace(
                pair_margin="25",
                save_dir="/tmp/test_ribopo"
            ),
            pairs=SimpleNamespace(
                train_path="/tmp/test_pairs.jsonl",
                eval_path="/tmp/test_pairs_eval.jsonl",
                margin="25"
            ),
            evaluation=SimpleNamespace(
                metrics=[
                    'recovery', 'perplexity', 'sc_score_rhofold'
                ],
                n_samples=4,
                temperature=0.5,
                pass_k=SimpleNamespace(
                    k_values=[1, 2, 4, 8],
                    thresholds=SimpleNamespace(
                        tm_score=[0.4, 0.45, 0.5],
                        rmsd=[8.0, 6.0, 4.0, 2.0],
                        mfe=[-10.0, -15.0, -20.0]
                    )
                ),
                distribution_analysis=SimpleNamespace(
                    enabled=True,
                    metrics=['plddt', 'rmsd', 'mfe', 'tm_score'],
                    statistical_tests=['ks_test', 'mannwhitney'],
                    plot_types=['pdf', 'cdf', 'boxplot']
                )
            ),
            checkpoint=SimpleNamespace(
                save_every_n_steps=50,
                keep_best_n=3,
                keep_last_n=2,
                selection_metric='val_pref_acc'
            )
        )
    
    @staticmethod
    def load_test_config(config_path: str):
        """Load test config from file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SimpleNamespace(**config_dict)


class MockData:
    """Generate mock data for testing."""
    
    @staticmethod
    def create_mock_preference_pair():
        """Create a mock preference pair."""
        return {
            "backbone_id": "test_backbone_1",
            "chosen": "AUGCUGCAUGCA",
            "rejected": "AUGCUGCAUGCU",
            "chosen_metrics": {
                "plddt": 0.8,
                "rmsd": 2.0,
                "mfe": -15.0
            },
            "rejected_metrics": {
                "plddt": 0.6,
                "rmsd": 5.0,
                "mfe": -10.0
            }
        }
    
    @staticmethod
    def create_mock_dataset(size=10):
        """Create mock dataset for testing."""
        return [MockData.create_mock_preference_pair() for _ in range(size)]
    
    @staticmethod
    def create_mock_backbone():
        """Create mock backbone structure."""
        return {
            "id": "test_backbone",
            "sequence": "AUGCUGCAAUGCACGU",
            "coordinates": torch.randn(16, 3),
            "mask": torch.ones(16, dtype=torch.bool)
        }


class TempDirectory:
    """Context manager for temporary directories."""
    
    def __init__(self, prefix="test_ribopo_"):
        self.prefix = prefix
        self.path = None
    
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return Path(self.path)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path)


@contextmanager
def capture_logs():
    """Capture log output for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)


class TestMetrics:
    """Helper functions for testing metrics."""
    
    @staticmethod
    def assert_metrics_valid(metrics_dict):
        """Assert that metrics dictionary contains valid values."""
        required_metrics = [
            'recovery_list', 'perplexity_list', 'sc_score_rmsd_list',
            'sc_score_tm_list', 'sc_score_gddt_list'
        ]
        
        for metric in required_metrics:
            assert metric in metrics_dict, f"Missing required metric: {metric}"
            values = metrics_dict[metric]
            assert isinstance(values, (list, np.ndarray)), f"Metric {metric} should be list/array"
            assert len(values) > 0, f"Metric {metric} should not be empty"
            assert all(not np.isnan(v) for v in values), f"Metric {metric} contains NaN values"
    
    @staticmethod
    def create_mock_evaluation_results():
        """Create mock evaluation results."""
        n_samples = 5
        return {
            'recovery_list': np.random.uniform(0.3, 0.8, n_samples),
            'perplexity_list': np.random.uniform(1.0, 5.0, n_samples),
            'sc_score_rmsd_list': np.random.uniform(1.0, 10.0, n_samples),
            'sc_score_tm_list': np.random.uniform(0.2, 0.8, n_samples),
            'sc_score_gddt_list': np.random.uniform(0.2, 0.8, n_samples),
            'sc_score_plddt_list': np.random.uniform(0.4, 0.9, n_samples),
            'samples_list': [np.random.randint(0, 4, 20) for _ in range(n_samples)]
        }


def setup_test_environment():
    """Set up test environment variables and paths."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for testing
    os.environ['WANDB_MODE'] = 'disabled'   # Disable wandb for testing
    

def cleanup_test_environment():
    """Clean up test environment."""
    # Remove any temporary files or reset environment
    pass


def assert_config_valid(config, required_fields=None):
    """Assert that config contains required fields."""
    if required_fields is None:
        required_fields = ['device', 'multiround', 'dpo', 'training', 'paths']
    
    for field in required_fields:
        assert hasattr(config, field), f"Config missing required field: {field}"


def mock_model_weights():
    """Generate mock model weights for testing."""
    return {
        'linear.weight': torch.randn(4, 128),
        'linear.bias': torch.randn(4),
        'embedding.weight': torch.randn(1000, 128)
    }


def assert_checkpoint_valid(checkpoint_path):
    """Assert that checkpoint file is valid."""
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'round']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint missing key: {key}"
    except Exception as e:
        raise AssertionError(f"Invalid checkpoint format: {e}")


if __name__ == "__main__":
    # Test the helper functions
    print("Testing helper functions...")
    
    # Test config creation
    config = TestConfig.create_minimal_config()
    assert_config_valid(config)
    print("✓ Config creation works")
    
    # Test mock data
    pair = MockData.create_mock_preference_pair()
    assert "chosen" in pair
    assert "rejected" in pair
    print("✓ Mock data creation works")
    
    # Test temporary directory
    with TempDirectory() as temp_dir:
        assert temp_dir.exists()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
    print("✓ Temporary directory works")
    
    print("All helper functions working correctly!")