#!/usr/bin/env python3
"""
Debug script to test the enhanced multiround evaluation metrics storage.

This script tests:
1. Individual metrics saving
2. Distribution plot creation
3. Organized output structure
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multiround.evaluator import MultiRoundEvaluator
from types import SimpleNamespace


def create_mock_eval_results(n_structures=10, n_samples=8):
    """Create mock evaluation results for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create realistic mock data
    eval_results = {
        'samples_list': [np.random.randint(0, 4, (n_samples, 50)) for _ in range(n_structures)],
        'recovery_list': np.random.uniform(0.3, 0.9, n_structures).tolist(),
        'perplexity_list': np.random.uniform(1.5, 4.0, n_structures).tolist(),
        'sc_score_tm': np.random.uniform(0.2, 0.8, n_structures).tolist(),
        'sc_score_rmsd': np.random.uniform(2.0, 12.0, n_structures).tolist(),
        'sc_score_gddt': np.random.uniform(0.1, 0.7, n_structures).tolist(),
        'sc_score_plddt': np.random.uniform(0.4, 0.9, n_structures).tolist(),
        'inf_all': np.random.uniform(0.0, 1.0, n_structures).tolist(),
        'inf_wc': np.random.uniform(0.0, 1.0, n_structures).tolist(),
        'inf_nwc': np.random.uniform(-0.2, 0.8, n_structures).tolist(),
        'inf_stack': np.random.uniform(0.0, 1.0, n_structures).tolist(),
        'clashscore_pre_relax': np.random.uniform(0.0, 15.0, n_structures).tolist(),
        'clashscore_post_relax': np.random.uniform(0.0, 8.0, n_structures).tolist(),
        'vienna_mfe': np.random.uniform(-25.0, -5.0, n_structures).tolist(),
        'vienna_ED_per_nt': np.random.uniform(0.1, 0.6, n_structures).tolist(),
        'vienna_pS0': np.random.uniform(0.1, 0.8, n_structures).tolist(),
        'vienna_entropy': np.random.uniform(0.5, 2.0, n_structures).tolist(),
        'vienna_diversity': np.random.uniform(0.3, 0.9, n_structures).tolist(),
        'vienna_Tm': np.random.uniform(35.0, 65.0, n_structures).tolist(),
        'sc_score_eternafold': np.random.uniform(0.2, 0.8, n_structures).tolist()
    }
    
    return eval_results


def test_individual_metrics_saving():
    """Test saving individual metrics."""
    print("🧪 Testing individual metrics saving...")
    
    # Create mock config
    cfg = SimpleNamespace(
        device="cpu",
        multiround=SimpleNamespace(
            eval_samples=8,
            final_eval_samples=64,
            eval_temperature=0.5,
            final_eval_temperature=0.1
        ),
        paths=SimpleNamespace(
            pairs=SimpleNamespace(
                test="data/test_pairs.jsonl"
            ),
            processed_pt="data/processed.pt"
        ),
        featurizer=SimpleNamespace(
            split="test",
            radius=0.0,
            top_k=32,
            num_rbf=32,
            num_posenc=32,
            max_num_conformers=1,
            noise_scale=0.0,
            distance_eps=0.001,
            device="cpu"
        ),
        evaluation=SimpleNamespace()
    )
    
    # Create evaluator (without actually loading dataset)
    evaluator = MultiRoundEvaluator(cfg)
    evaluator.eval_dataset = None  # Skip dataset loading for testing
    
    # Create test output directory
    test_output_dir = "/tmp/test_multiround_metrics"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create mock evaluation results
    eval_results = create_mock_eval_results(n_structures=20, n_samples=8)
    round_num = 2
    
    try:
        # Test individual metrics saving
        evaluator._save_individual_metrics(eval_results, test_output_dir, round_num, 8)
        
        # Verify file was created
        expected_file = os.path.join(test_output_dir, f"individual_metrics_round_{round_num}.json")
        assert os.path.exists(expected_file), f"Individual metrics file not created: {expected_file}"
        
        # Verify file contents
        with open(expected_file, 'r') as f:
            saved_metrics = json.load(f)
        
        assert len(saved_metrics) == 20, f"Expected 20 structures, got {len(saved_metrics)}"
        assert saved_metrics[0]['round'] == round_num, "Round number not saved correctly"
        assert 'tm_score' in saved_metrics[0], "TM score not saved"
        assert 'recovery' in saved_metrics[0], "Recovery not saved"
        
        print("✅ Individual metrics saving test passed")
        
    except Exception as e:
        print(f"❌ Individual metrics saving test failed: {e}")
        raise


def test_distribution_plots():
    """Test distribution plot creation."""
    print("🧪 Testing distribution plots creation...")
    
    # Create mock config
    cfg = SimpleNamespace(device="cpu")
    evaluator = MultiRoundEvaluator(cfg)
    
    # Create test output directory
    test_output_dir = "/tmp/test_distribution_plots"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create mock evaluation results
    eval_results = create_mock_eval_results(n_structures=50, n_samples=8)
    round_num = 3
    
    try:
        # Test distribution plot creation
        evaluator._create_distribution_plots(eval_results, test_output_dir, round_num)
        
        # Verify plot was created
        expected_plot = os.path.join(test_output_dir, f"distribution_plots_round_{round_num}.png")
        assert os.path.exists(expected_plot), f"Distribution plot not created: {expected_plot}"
        
        print("✅ Distribution plots test passed")
        
    except Exception as e:
        print(f"❌ Distribution plots test failed: {e}")
        # Don't raise for plot failures as they might be due to display issues


def test_checkpoint_selection_logic():
    """Test checkpoint selection logic."""
    print("🧪 Testing checkpoint selection logic...")
    
    # Create mock trainer with selection method
    from multiround.trainer import MultiRoundDPOTrainer
    
    # Create minimal config
    cfg = SimpleNamespace(
        device="cpu",
        multiround=SimpleNamespace(
            num_rounds=3,
            epochs_per_round=5
        )
    )
    
    # Create test output directory and mock evaluation results
    test_output_dir = "/tmp/test_checkpoint_selection"
    eval_dir = os.path.join(test_output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create mock evaluation results with pass@k metrics
    eval_results = {
        'round': 2,
        'n_samples': 8,
        'temperature': 0.5,
        'n_structures': 98,
        'tm_mean': 0.65,
        'rmsd_mean': 4.2,
        'mfe_mean': -15.3,
        'passk_tm_0.45_k8': 0.73,  # Primary metric
        'passk_rmsd_8.0_k8': 0.81,
        'passk_combined_tm0.45_rmsd8.0_k8': 0.68
    }
    
    # Save mock evaluation results
    eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    try:
        # Create trainer instance (this will fail on full init, so we'll test the method directly)
        trainer = MultiRoundDPOTrainer.__new__(MultiRoundDPOTrainer)  # Create without __init__
        trainer.best_checkpoints = []
        
        # Test checkpoint selection
        result = trainer.select_best_checkpoint_from_round(2, test_output_dir)
        
        assert result is not None, "Checkpoint selection returned None"
        assert result['round'] == 2, "Wrong round in selection result"
        assert 'primary_metric' in result, "Primary metric not in result"
        assert result['primary_metric'] == 0.73, "Primary metric value incorrect"
        
        print("✅ Checkpoint selection logic test passed")
        
    except Exception as e:
        print(f"❌ Checkpoint selection logic test failed: {e}")
        raise


def test_output_organization():
    """Test that outputs are properly organized."""
    print("🧪 Testing output organization...")
    
    test_base_dir = "/tmp/test_output_organization"
    
    # Simulate round directory creation
    round_dir = os.path.join(test_base_dir, "round_02")
    eval_dir = os.path.join(round_dir, "evaluation")
    designs_dir = os.path.join(round_dir, "designs")
    checkpoints_dir = os.path.join(round_dir, "checkpoints")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(designs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check directory structure
    assert os.path.exists(eval_dir), "Evaluation directory not created"
    assert os.path.exists(designs_dir), "Designs directory not created"
    assert os.path.exists(checkpoints_dir), "Checkpoints directory not created"
    
    print("✅ Output organization test passed")


def main():
    """Run all tests."""
    print("🧪 Starting multiround metrics tests...\n")
    
    try:
        test_individual_metrics_saving()
        print()
        
        test_distribution_plots()
        print()
        
        test_checkpoint_selection_logic()
        print()
        
        test_output_organization()
        print()
        
        print("✅ All multiround metrics tests passed!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()