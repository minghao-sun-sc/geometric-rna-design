#!/usr/bin/env python3
"""
Debug script to test checkpoint selection logic in multiround training.

This script tests:
1. Checkpoint selection based on pass@k metrics
2. Tie-breaking with MFE values
3. Selection rationale logging
4. Comparison across rounds
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multiround.trainer import MultiRoundDPOTrainer
from types import SimpleNamespace


def create_mock_evaluation_results(round_num, passk_score, mfe_score, tm_score=0.65, rmsd_score=4.2):
    """Create mock evaluation results for testing."""
    return {
        'round': round_num,
        'n_samples': 8,
        'temperature': 0.5,
        'n_structures': 98,
        'timestamp': '2024-01-01T12:00:00',
        'tm_mean': tm_score,
        'rmsd_mean': rmsd_score,
        'mfe_mean': mfe_score,
        'recovery': 0.75,
        'perplexity': 2.3,
        'passk_tm_0.45_k8': passk_score,
        'passk_rmsd_8.0_k8': 0.81,
        'passk_combined_tm0.45_rmsd8.0_k8': passk_score * 0.9,  # Slightly lower combined score
        'plddt_mean': 0.72,
        'gdt_mean': 0.58
    }


def setup_mock_trainer():
    """Setup a mock trainer for testing."""
    trainer = MultiRoundDPOTrainer.__new__(MultiRoundDPOTrainer)  # Create without __init__
    trainer.best_checkpoints = []
    return trainer


def test_first_round_selection():
    """Test checkpoint selection for the first round."""
    print("🧪 Testing first round checkpoint selection...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create evaluation results for round 1
        eval_results = create_mock_evaluation_results(1, 0.65, -15.0)
        eval_file = os.path.join(eval_dir, "eval_results_round_1.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Add a checkpoint to the list (simulating that one was saved)
        trainer.best_checkpoints.append({
            'round': 1,
            'path': os.path.join(temp_dir, "checkpoints", "round_1_best.pt"),
            'metric_value': 0.65
        })
        
        # Test selection
        result = trainer.select_best_checkpoint_from_round(1, temp_dir)
        
        assert result is not None, "First round selection returned None"
        assert result['round'] == 1, "Wrong round in result"
        assert 'selection_reason' in result, "No selection reason provided"
        
        print("✅ First round selection test passed")


def test_improvement_detection():
    """Test detection of improved checkpoints."""
    print("🧪 Testing improvement detection...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Setup previous best checkpoint (round 1)
        trainer.best_checkpoints.append({
            'round': 1,
            'path': os.path.join(temp_dir, "checkpoints", "round_1_best.pt"),
            'primary_metric': 0.60,  # Lower pass@k score
            'tiebreaker_metric': -12.0,  # Worse MFE
            'is_best': True
        })
        
        # Create better evaluation results for round 2
        eval_results = create_mock_evaluation_results(2, 0.75, -18.0)  # Better pass@k and MFE
        eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Test selection
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        assert result is not None, "Selection returned None"
        assert result['is_best'] is True, "Better checkpoint not marked as best"
        assert result['primary_metric'] == 0.75, "Primary metric not correct"
        assert "Higher pass@k" in result['selection_reason'], "Improvement reason not detected"
        
        print("✅ Improvement detection test passed")


def test_tie_breaking():
    """Test tie-breaking with MFE values."""
    print("🧪 Testing tie-breaking with MFE...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Setup previous best checkpoint with same pass@k score
        trainer.best_checkpoints.append({
            'round': 1,
            'path': os.path.join(temp_dir, "checkpoints", "round_1_best.pt"),
            'primary_metric': 0.70,  # Same pass@k score
            'tiebreaker_metric': -12.0,  # Worse MFE (higher value)
            'is_best': True
        })
        
        # Create evaluation results with same pass@k but better MFE
        eval_results = create_mock_evaluation_results(2, 0.70, -18.0)  # Same pass@k, better MFE
        eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Test selection
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        assert result is not None, "Selection returned None"
        assert result['is_best'] is True, "Better MFE not recognized"
        assert "better MFE" in result['selection_reason'], "MFE tie-breaking not detected"
        assert result['tiebreaker_metric'] == -18.0, "MFE value not correct"
        
        print("✅ Tie-breaking test passed")


def test_no_improvement():
    """Test when current round is worse than previous."""
    print("🧪 Testing no improvement detection...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Setup previous best checkpoint with better scores
        trainer.best_checkpoints.append({
            'round': 1,
            'path': os.path.join(temp_dir, "checkpoints", "round_1_best.pt"),
            'primary_metric': 0.80,  # Better pass@k score
            'tiebreaker_metric': -20.0,  # Better MFE
            'is_best': True
        })
        
        # Create worse evaluation results for round 2
        eval_results = create_mock_evaluation_results(2, 0.65, -15.0)  # Worse scores
        eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Test selection
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        assert result is not None, "Selection returned None"
        assert result['round'] == 1, "Should keep previous best checkpoint"
        assert result['is_best'] is True, "Previous best not maintained"
        
        print("✅ No improvement detection test passed")


def test_selection_logging():
    """Test that selection rationale is properly logged."""
    print("🧪 Testing selection rationale logging...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create evaluation results
        eval_results = create_mock_evaluation_results(2, 0.73, -16.5)
        eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Test selection
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        assert result is not None, "Selection returned None"
        assert 'selection_criteria' in result, "Selection criteria not logged"
        assert 'primary' in result['selection_criteria'], "Primary criteria not logged"
        assert 'tiebreaker' in result['selection_criteria'], "Tiebreaker criteria not logged"
        
        # Check that the criteria contain the expected values
        primary_criteria = result['selection_criteria']['primary']
        assert "0.73" in primary_criteria, "Primary metric value not in criteria"
        
        tiebreaker_criteria = result['selection_criteria']['tiebreaker']
        assert "-16.5" in tiebreaker_criteria, "Tiebreaker metric value not in criteria"
        
        print("✅ Selection rationale logging test passed")


def test_missing_evaluation_file():
    """Test handling of missing evaluation results."""
    print("🧪 Testing missing evaluation file handling...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Don't create evaluation results file
        
        # Test selection with missing file
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        # Should handle gracefully and return None or previous best
        assert result is None or 'round' in result, "Should handle missing file gracefully"
        
        print("✅ Missing evaluation file handling test passed")


def test_edge_cases():
    """Test edge cases in checkpoint selection."""
    print("🧪 Testing edge cases...")
    
    trainer = setup_mock_trainer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_dir = os.path.join(temp_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Test with missing pass@k metric
        eval_results = {
            'round': 2,
            'n_samples': 8,
            'mfe_mean': -15.0,
            'tm_mean': 0.65
            # Missing 'passk_tm_0.45_k8'
        }
        eval_file = os.path.join(eval_dir, "eval_results_round_2.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Test selection with missing metric
        result = trainer.select_best_checkpoint_from_round(2, temp_dir)
        
        # Should handle gracefully
        assert result is not None, "Should handle missing metrics gracefully"
        assert result['primary_metric'] == 0.0, "Should default to 0.0 for missing metric"
        
        print("✅ Edge cases test passed")


def main():
    """Run all checkpoint selection tests."""
    print("🧪 Starting checkpoint selection tests...\n")
    
    tests = [
        test_first_round_selection,
        test_improvement_detection,
        test_tie_breaking,
        test_no_improvement,
        test_selection_logging,
        test_missing_evaluation_file,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"✅ Checkpoint selection tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()