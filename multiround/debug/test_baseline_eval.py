#!/usr/bin/env python3
"""
Debug script to test the baseline evaluation pipeline.

This script tests:
1. Loading sequences from different formats
2. Matching sequences with test dataset
3. Structure prediction (mock)
4. Metrics computation
5. Results aggregation and saving
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multiround.eval_baseline import BaselineEvaluator
from types import SimpleNamespace


def create_test_sequences():
    """Create test sequences for different formats."""
    # Sample RNA sequences (these should exist in the test dataset)
    test_sequences = {
        '1DDY_1_A': 'GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAUGGAUGCCGACUGGCUCUUAAACACGGGUGAUACCGUCACGCACU',
        '1Y26_1_X': 'CGCCGGGUAGCGCUGGGCUUCCGGGGACGGGCGUAGAGCGCACCAUGGUCGGCAGCGGUUCCGCACGGAGCUUU',
        '2HOK_1_A': 'GAGUAGCUGACCUAAGAGCGGACUGGCCAACUAAGUCAGCCACGGUGCUUUCUGCUUAGGUCAGCCUGCCUUUCUGCU'
    }
    return test_sequences


def create_test_fasta_file(sequences, filepath):
    """Create a test FASTA file."""
    with open(filepath, 'w') as f:
        for structure_id, sequence in sequences.items():
            f.write(f">{structure_id}\n{sequence}\n")


def create_test_json_file(sequences, filepath):
    """Create a test JSON file."""
    with open(filepath, 'w') as f:
        json.dump(sequences, f, indent=2)


def create_test_csv_file(sequences, filepath):
    """Create a test CSV file."""
    import pandas as pd
    
    data = []
    for structure_id, sequence in sequences.items():
        data.append({
            'structure_id': structure_id,
            'sequence': sequence,
            'chain': 'A',
            'description': f'Test sequence for {structure_id}'
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def test_sequence_loading():
    """Test loading sequences from different formats."""
    print("🧪 Testing sequence loading...")
    
    sequences = create_test_sequences()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test FASTA loading
        fasta_file = os.path.join(temp_dir, "test_sequences.fasta")
        create_test_fasta_file(sequences, fasta_file)
        
        # Test JSON loading
        json_file = os.path.join(temp_dir, "test_sequences.json")
        create_test_json_file(sequences, json_file)
        
        # Test CSV loading
        csv_file = os.path.join(temp_dir, "test_sequences.csv")
        create_test_csv_file(sequences, csv_file)
        
        # Create mock config
        cfg = SimpleNamespace(
            device="cpu",
            test_dataset=SimpleNamespace(
                processed_pt="data/processed.pt",
                split_pt="data/das_split.pt"
            )
        )
        
        # Create evaluator (this will fail if actual dataset files don't exist)
        try:
            evaluator = BaselineEvaluator(cfg)
            
            # Test FASTA loading
            fasta_sequences = evaluator._load_sequences(fasta_file, "fasta")
            assert len(fasta_sequences) == len(sequences), f"FASTA: Expected {len(sequences)}, got {len(fasta_sequences)}"
            
            # Test JSON loading
            json_sequences = evaluator._load_sequences(json_file, "json")
            assert len(json_sequences) == len(sequences), f"JSON: Expected {len(sequences)}, got {len(json_sequences)}"
            
            # Test CSV loading
            csv_sequences = evaluator._load_sequences(csv_file, "csv")
            assert len(csv_sequences) == len(sequences), f"CSV: Expected {len(sequences)}, got {len(csv_sequences)}"
            
            # Test auto-detection
            auto_fasta = evaluator._load_sequences(fasta_file, "auto")
            assert len(auto_fasta) == len(sequences), f"Auto FASTA: Expected {len(sequences)}, got {len(auto_fasta)}"
            
            print("✅ Sequence loading tests passed")
            return True
            
        except Exception as e:
            print(f"⚠️ Sequence loading test skipped (dataset files not available): {e}")
            return False


def test_metrics_computation():
    """Test individual metrics computation methods."""
    print("🧪 Testing metrics computation...")
    
    try:
        # Create mock config
        cfg = SimpleNamespace(device="cpu")
        
        # Create evaluator instance without dataset loading
        evaluator = BaselineEvaluator.__new__(BaselineEvaluator)
        evaluator.cfg = cfg
        evaluator.device = torch.device("cpu") if 'torch' in globals() else "cpu"
        
        # Test sequence recovery computation
        designed_seq = "AUGCGU"
        native_seq = "AAGCGU"
        recovery = evaluator._compute_sequence_recovery(designed_seq, native_seq)
        expected_recovery = 5/6  # 5 out of 6 bases match
        assert abs(recovery - expected_recovery) < 1e-6, f"Recovery: expected {expected_recovery}, got {recovery}"
        
        # Test length mismatch handling
        recovery_mismatch = evaluator._compute_sequence_recovery("AUGC", "AAGCGU")
        expected_mismatch = 3/6  # Compare first 4 bases, 3 match out of 6 total
        assert abs(recovery_mismatch - 3/4) < 1e-6, f"Recovery mismatch: expected 0.75, got {recovery_mismatch}"
        
        print("✅ Metrics computation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics computation test failed: {e}")
        return False


def test_results_aggregation():
    """Test results aggregation."""
    print("🧪 Testing results aggregation...")
    
    try:
        # Create mock results
        results = [
            {
                'structure_id': 'test1',
                'sequence_recovery': 0.8,
                'tm_score': 0.6,
                'rmsd': 3.2,
                'mfe': -15.5
            },
            {
                'structure_id': 'test2',
                'sequence_recovery': 0.7,
                'tm_score': 0.7,
                'rmsd': 2.8,
                'mfe': -18.2
            },
            {
                'structure_id': 'test3',
                'sequence_recovery': 0.9,
                'tm_score': 0.5,
                'rmsd': 4.1,
                'mfe': -12.3
            }
        ]
        
        # Create evaluator instance
        evaluator = BaselineEvaluator.__new__(BaselineEvaluator)
        
        # Test aggregation
        aggregated = evaluator._aggregate_baseline_results(results, "test_model")
        
        assert aggregated['model_name'] == "test_model", "Model name not set correctly"
        assert aggregated['n_structures'] == 3, "Number of structures incorrect"
        
        # Check aggregated metrics
        assert abs(aggregated['sequence_recovery_mean'] - 0.8) < 1e-6, "Recovery mean incorrect"
        assert abs(aggregated['tm_score_mean'] - 0.6) < 1e-6, "TM score mean incorrect"
        assert abs(aggregated['rmsd_mean'] - (3.2 + 2.8 + 4.1)/3) < 1e-6, "RMSD mean incorrect"
        
        print("✅ Results aggregation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Results aggregation test failed: {e}")
        return False


def test_file_output():
    """Test file output functionality."""
    print("🧪 Testing file output...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data
            aggregated_results = {
                'model_name': 'test_model',
                'n_structures': 2,
                'sequence_recovery_mean': 0.75,
                'tm_score_mean': 0.65
            }
            
            structure_metrics = [
                {'structure_id': 'test1', 'sequence_recovery': 0.8, 'tm_score': 0.6},
                {'structure_id': 'test2', 'sequence_recovery': 0.7, 'tm_score': 0.7}
            ]
            
            # Create evaluator instance
            evaluator = BaselineEvaluator.__new__(BaselineEvaluator)
            
            # Test saving
            evaluator._save_baseline_results(
                aggregated_results, structure_metrics, temp_dir, "test_model"
            )
            
            # Check files were created
            expected_files = [
                "test_model_aggregated_results.json",
                "test_model_individual_results.json",
                "test_model_summary.json"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath), f"File not created: {filename}"
                
                # Check file content
                with open(filepath, 'r') as f:
                    data = json.load(f)
                assert data, f"File {filename} is empty"
            
            print("✅ File output tests passed")
            return True
            
    except Exception as e:
        print(f"❌ File output test failed: {e}")
        return False


def test_end_to_end_mock():
    """Test end-to-end evaluation with mocked components."""
    print("🧪 Testing end-to-end evaluation (mocked)...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test sequences file
            sequences = create_test_sequences()
            sequences_file = os.path.join(temp_dir, "test_sequences.fasta")
            create_test_fasta_file(sequences, sequences_file)
            
            # Mock the evaluator to skip actual structure prediction and dataset loading
            class MockBaselineEvaluator(BaselineEvaluator):
                def __init__(self, cfg):
                    self.cfg = cfg
                    self.device = "cpu"
                    # Mock test dataset
                    self.test_dataset = []
                    self.id_to_test_idx = {}
                    for i, structure_id in enumerate(sequences.keys()):
                        self.id_to_test_idx[structure_id] = i
                    
                def _match_sequences_with_dataset(self, sequences_data):
                    # Mock matching - return all sequences as matched
                    matched = []
                    for seq_data in sequences_data:
                        test_item = {
                            'sequence': seq_data['sequence'],
                            'sec_struct_list': ['.' * len(seq_data['sequence'])],
                            'coords_list': [],
                            'id_list': [seq_data['structure_id']]
                        }
                        matched.append((seq_data, test_item, 0))
                    return matched
                
                def _predict_structure(self, sequence, structure_id, output_dir):
                    # Mock structure prediction - create empty PDB
                    pdb_path = os.path.join(output_dir, f"{structure_id}_predicted.pdb")
                    with open(pdb_path, 'w') as f:
                        f.write("HEADER    MOCK PDB\\n")
                    return pdb_path
                
                def _compute_all_metrics(self, sequence, predicted_pdb_path, test_item, output_dir):
                    # Return mock metrics
                    return {
                        'sequence_recovery': 0.8,
                        'edit_distance': 5,
                        'sc_eternafold': 0.6,
                        'rmsd': 3.5,
                        'tm_score': 0.65,
                        'gdt': 0.55,
                        'mfe': -15.0
                    }
            
            # Create mock config
            cfg = SimpleNamespace(device="cpu")
            
            # Run mock evaluation
            evaluator = MockBaselineEvaluator(cfg)
            results = evaluator.evaluate_baseline_sequences(
                sequences_file=sequences_file,
                model_name="mock_model",
                output_dir=temp_dir
            )
            
            # Check results
            assert 'error' not in results, f"Evaluation failed: {results.get('error')}"
            assert results['model_name'] == "mock_model", "Model name incorrect"
            assert results['n_structures'] == len(sequences), f"Expected {len(sequences)} structures"
            
            print("✅ End-to-end mock test passed")
            return True
            
    except Exception as e:
        print(f"❌ End-to-end mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all baseline evaluation tests."""
    print("🧪 Starting baseline evaluation tests...\n")
    
    tests = [
        test_sequence_loading,
        test_metrics_computation, 
        test_results_aggregation,
        test_file_output,
        test_end_to_end_mock
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            success = test()
            if success:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"✅ Baseline evaluation tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed or were skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()