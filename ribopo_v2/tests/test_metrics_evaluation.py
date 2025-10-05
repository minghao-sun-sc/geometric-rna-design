"""
Test module for RiboPO v2 metrics evaluation
Validates the real metrics calculation pipeline.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ribopo_v2.metrics_evaluation import RealMetricsEvaluator, MetricsResult


class TestMetricsEvaluation:
    """Test suite for real metrics evaluation"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing"""
        return RealMetricsEvaluator(cache_dir="ribopo_v2/cache/metrics_test")
    
    @pytest.fixture
    def test_sequence(self):
        """Provide test RNA sequence"""
        return "GGAAAGCUGAAGCUGGCCCUGAUGGAGCUGAGAACUGGGGCUCC"
    
    @pytest.fixture
    def test_backbone_data(self, test_sequence):
        """Provide test backbone data"""
        return {
            'id_list': ['1Y26_1_X'],
            'sequence': test_sequence,
            'coords_list': [torch.zeros((len(test_sequence), 27, 3))],
            'sec_struct_list': ['(((((.......))..)))......(((((.....)))))'[:len(test_sequence)]]
        }
    
    def test_cache_key_generation(self, evaluator, test_sequence):
        """Test cache key generation is deterministic"""
        key1 = evaluator._get_cache_key(test_sequence, "backbone1")
        key2 = evaluator._get_cache_key(test_sequence, "backbone1")
        key3 = evaluator._get_cache_key(test_sequence, "backbone2")
        
        assert key1 == key2  # Same inputs produce same key
        assert key1 != key3  # Different backbone produces different key
        assert len(key1) == 32  # MD5 hash length
    
    def test_thermodynamic_metrics(self, evaluator, test_sequence):
        """Test Vienna thermodynamic metrics calculation"""
        # Test with and without target structure
        metrics_no_target = evaluator.calculate_thermodynamic_metrics(test_sequence)
        
        assert 'mfe' in metrics_no_target
        assert 'ensemble_defect' in metrics_no_target
        assert 'ed_per_nt' in metrics_no_target
        assert 'shannon_entropy' in metrics_no_target
        
        # MFE should be negative for stable structures
        if not np.isnan(metrics_no_target['mfe']):
            assert metrics_no_target['mfe'] < 0
        
        # ED/nt should be between 0 and 1
        if not np.isnan(metrics_no_target['ed_per_nt']):
            assert 0 <= metrics_no_target['ed_per_nt'] <= 1
        
        # Test with target structure
        target_ss = "(((((.......))..)))......(((((.....)))))"[:len(test_sequence)]
        metrics_with_target = evaluator.calculate_thermodynamic_metrics(test_sequence, target_ss)
        
        assert 'p_target' in metrics_with_target
        # P(target) should be between 0 and 1
        if not np.isnan(metrics_with_target['p_target']):
            assert 0 <= metrics_with_target['p_target'] <= 1
    
    def test_metrics_result_dataclass(self, test_sequence):
        """Test MetricsResult dataclass initialization"""
        result = MetricsResult(
            sequence=test_sequence,
            backbone_id="test_backbone",
            tm_score=0.5,
            inf_all=0.3,
            ed_per_nt=0.4
        )
        
        assert result.sequence == test_sequence
        assert result.backbone_id == "test_backbone"
        assert result.tm_score == 0.5
        assert result.inf_all == 0.3
        assert result.ed_per_nt == 0.4
        assert np.isnan(result.rmsd)  # Default NaN values
    
    def test_cache_operations(self, evaluator, test_sequence):
        """Test cache save and load operations"""
        # Create test result
        result = MetricsResult(
            sequence=test_sequence,
            backbone_id="test_backbone",
            tm_score=0.5,
            inf_all=0.3,
            ed_per_nt=0.4,
            cache_key="test_cache_key"
        )
        
        # Save to cache
        evaluator._save_to_cache(result)
        
        # Load from cache
        loaded = evaluator._load_from_cache("test_cache_key")
        
        assert loaded is not None
        assert loaded.sequence == result.sequence
        assert loaded.tm_score == result.tm_score
        assert loaded.inf_all == result.inf_all
        
        # Clean up test cache
        cache_file = evaluator.cache_dir / "test_cache_key.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    @pytest.mark.slow
    def test_full_evaluation_pipeline(self, evaluator, test_sequence, test_backbone_data):
        """Test complete evaluation pipeline (requires RhoFold)"""
        # This test requires RhoFold model to be available
        # Mark as slow since it involves model loading and prediction
        
        result = evaluator.evaluate_sequence(
            test_sequence,
            test_backbone_data,
            "test_backbone",
            use_cache=False
        )
        
        assert isinstance(result, MetricsResult)
        assert result.sequence == test_sequence
        assert result.backbone_id == "test_backbone"
        
        # Check that metrics were calculated (may be NaN if models not available)
        assert hasattr(result, 'tm_score')
        assert hasattr(result, 'inf_all')
        assert hasattr(result, 'mfe')
        assert hasattr(result, 'ed_per_nt')
    
    def test_batch_evaluation(self, evaluator, test_backbone_data):
        """Test batch evaluation of multiple sequences"""
        sequences = [
            "GGAAAGCUGAAGCUGGCCCUGAUGGAGCUGAGAACUGGGGCUCC",
            "AUCGGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA",
            "GCGCAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUGC"
        ]
        
        results = evaluator.evaluate_batch(
            sequences[:2],  # Test with 2 sequences for speed
            test_backbone_data,
            "test_backbone_batch",
            use_cache=True
        )
        
        assert len(results) == 2
        assert all(isinstance(r, MetricsResult) for r in results)
        assert results[0].sequence == sequences[0]
        assert results[1].sequence == sequences[1]


class TestIntegrationWithCandidateEvaluation:
    """Test integration of metrics evaluation with candidate evaluation pipeline"""
    
    def test_mock_to_real_metrics_replacement(self):
        """Verify real metrics can replace mock evaluation"""
        from ribopo_v2.candidate_evaluation import RiboPOv2CandidateEvaluator
        
        # This test would verify that the real metrics evaluator
        # can be integrated into the candidate evaluation pipeline
        # For now, we just check the interface compatibility
        
        evaluator = RealMetricsEvaluator(cache_dir="ribopo_v2/cache/metrics_test")
        
        # Check that evaluator has the required methods
        assert hasattr(evaluator, 'evaluate_sequence')
        assert hasattr(evaluator, 'evaluate_batch')
        
        # Test sequence evaluation returns proper format
        test_seq = "GGAAAGCUGAAGCUGGCCCUGAUGGAGCUGAGAACUGGGGCUCC"
        test_backbone = {
            'id_list': ['test'],
            'coords_list': [torch.zeros((len(test_seq), 27, 3))],
            'sec_struct_list': ['.' * len(test_seq)]
        }
        
        result = evaluator.evaluate_sequence(
            test_seq,
            test_backbone,
            "test_backbone",
            use_cache=False
        )
        
        # Check result has required metrics for S_total calculation
        assert hasattr(result, 'tm_score')
        assert hasattr(result, 'inf_all')
        assert hasattr(result, 'ed_per_nt')


def run_basic_tests():
    """Run basic tests without pytest"""
    print("Running basic metrics evaluation tests...")
    
    # Test 1: Initialize evaluator
    print("\n1. Testing evaluator initialization...")
    evaluator = RealMetricsEvaluator(cache_dir="ribopo_v2/cache/metrics_test")
    print("   ✓ Evaluator initialized")
    
    # Test 2: Cache key generation
    print("\n2. Testing cache key generation...")
    test_seq = "GGAAAGCUGAAGCUGGCCCUGAUGGAGCUGAGAACUGGGGCUCC"
    key = evaluator._get_cache_key(test_seq, "test_backbone")
    assert len(key) == 32
    print(f"   ✓ Cache key generated: {key[:8]}...")
    
    # Test 3: Thermodynamic metrics
    print("\n3. Testing thermodynamic metrics calculation...")
    metrics = evaluator.calculate_thermodynamic_metrics(test_seq)
    print(f"   ✓ MFE: {metrics['mfe']:.2f} kcal/mol")
    print(f"   ✓ ED/nt: {metrics['ed_per_nt']:.3f}")
    print(f"   ✓ Shannon entropy: {metrics['shannon_entropy']:.3f}")
    
    # Test 4: MetricsResult
    print("\n4. Testing MetricsResult dataclass...")
    result = MetricsResult(
        sequence=test_seq,
        backbone_id="test",
        tm_score=0.5,
        inf_all=0.3
    )
    assert result.tm_score == 0.5
    print("   ✓ MetricsResult created successfully")
    
    # Test 5: Cache operations
    print("\n5. Testing cache operations...")
    result.cache_key = "test_cache"
    evaluator._save_to_cache(result)
    loaded = evaluator._load_from_cache("test_cache")
    assert loaded is not None
    assert loaded.tm_score == 0.5
    print("   ✓ Cache save/load working")
    
    # Clean up
    cache_file = evaluator.cache_dir / "test_cache.pkl"
    if cache_file.exists():
        cache_file.unlink()
    
    print("\n✅ All basic tests passed!")
    return True


if __name__ == "__main__":
    # Run basic tests without pytest
    success = run_basic_tests()
    
    if success:
        print("\n" + "="*50)
        print("To run comprehensive tests with pytest:")
        print("  pytest ribopo_v2/tests/test_metrics_evaluation.py -v")
        print("  pytest ribopo_v2/tests/test_metrics_evaluation.py -v -m 'not slow'  # Skip slow tests")
        print("="*50)