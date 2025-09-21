# multiround/debug/utils/mock_data.py
"""Mock data generation for comprehensive testing."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()


class MockRNA:
    """Generate realistic mock RNA data."""
    
    NUCLEOTIDES = ['A', 'U', 'G', 'C']
    NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    @classmethod
    def random_sequence(cls, length: int) -> str:
        """Generate random RNA sequence."""
        return ''.join(np.random.choice(cls.NUCLEOTIDES, length))
    
    @classmethod
    def sequence_to_tensor(cls, sequence: str) -> torch.Tensor:
        """Convert sequence to tensor."""
        return torch.tensor([cls.NUCLEOTIDE_MAP[nt] for nt in sequence])
    
    @classmethod
    def create_coordinates(cls, length: int) -> torch.Tensor:
        """Create realistic 3D coordinates."""
        # Simulate RNA backbone with realistic distances
        coords = torch.zeros(length, 3)
        for i in range(1, length):
            # Add some structure with phosphate-phosphate distance ~6Å
            coords[i] = coords[i-1] + torch.normal(0, 1, (3,)) * 2.0
        return coords
    
    @classmethod
    def create_secondary_structure(cls, length: int) -> str:
        """Create realistic secondary structure in dot-bracket notation."""
        # Simple stem-loop structures
        structure = ['.'] * length
        
        # Add some base pairs
        for i in range(0, length - 10, 15):
            stem_length = min(4, (length - i) // 3)
            for j in range(stem_length):
                if i + j < length and i + 10 + j < length:
                    structure[i + j] = '('
                    structure[i + 10 + j] = ')'
        
        return ''.join(structure)


class MockDataset:
    """Generate comprehensive mock datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_preference_pairs(self, n_pairs: int = 100, margin_type: str = "25") -> Path:
        """Create mock preference pairs dataset."""
        pairs = []
        
        for i in range(n_pairs):
            seq_length = np.random.randint(15, 50)
            
            # Generate backbone structure
            backbone_seq = MockRNA.random_sequence(seq_length)
            backbone_coords = MockRNA.create_coordinates(seq_length)
            backbone_ss = MockRNA.create_secondary_structure(seq_length)
            
            # Generate chosen and rejected sequences (similar to backbone)
            chosen_seq = self._mutate_sequence(backbone_seq, mutation_rate=0.1)
            rejected_seq = self._mutate_sequence(backbone_seq, mutation_rate=0.3)
            
            # Create metrics with appropriate preference gap
            margin_factor = 0.25 if margin_type == "25" else 0.125
            
            chosen_metrics = self._generate_metrics(quality="good", margin=margin_factor)
            rejected_metrics = self._generate_metrics(quality="poor", margin=margin_factor)
            
            pair = {
                "backbone_id": f"test_backbone_{i:04d}",
                "backbone_sequence": backbone_seq,
                "backbone_coordinates": backbone_coords.tolist(),
                "backbone_secondary_structure": backbone_ss,
                "chosen": chosen_seq,
                "rejected": rejected_seq,
                "chosen_metrics": chosen_metrics,
                "rejected_metrics": rejected_metrics,
                "margin_type": margin_type,
                "split": "train" if i < n_pairs * 0.8 else "val"
            }
            pairs.append(pair)
        
        # Split into train/val
        train_pairs = [p for p in pairs if p["split"] == "train"]
        val_pairs = [p for p in pairs if p["split"] == "val"]
        
        # Save to files
        margin_dir = self.data_dir / f"pairs_margin{margin_type}"
        margin_dir.mkdir(exist_ok=True)
        
        train_file = margin_dir / "train.clean.jsonl"
        val_file = margin_dir / "val.clean.jsonl"
        
        self._save_jsonl(train_pairs, train_file)
        self._save_jsonl(val_pairs, val_file)
        
        return margin_dir
    
    def create_test_dataset(self, n_structures: int = 20) -> Path:
        """Create mock test dataset for evaluation."""
        test_data = []
        
        for i in range(n_structures):
            seq_length = np.random.randint(20, 60)
            
            # Create multiple conformations for same structure
            n_conformations = np.random.randint(1, 4)
            coords_list = [MockRNA.create_coordinates(seq_length) for _ in range(n_conformations)]
            
            structure = {
                "id_list": [f"test_pdb_{i:04d}"],
                "sequence": MockRNA.random_sequence(seq_length),
                "coords_list": [coords.tolist() for coords in coords_list],
                "sec_struct_list": [MockRNA.create_secondary_structure(seq_length)],
                "mask_coords": [True] * seq_length,
                "sasa_list": [np.random.uniform(0, 100, seq_length).tolist()],
                "rfam_list": [f"RF{i:05d}"],
                "eq_class_list": [f"eq_class_{i % 10}"],
                "cluster_structsim0.45": f"cluster_{i % 5}",
                "split": "test"
            }
            test_data.append(structure)
        
        # Save test dataset
        test_file = self.data_dir / "test_dataset.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        return test_file
    
    def create_model_checkpoint(self, round_num: int = 1) -> Path:
        """Create mock model checkpoint."""
        checkpoint = {
            "model_state_dict": {
                "embedding.weight": torch.randn(1000, 128).tolist(),
                "gnn.layers.0.weight": torch.randn(128, 128).tolist(),
                "output.weight": torch.randn(4, 128).tolist(),
                "output.bias": torch.randn(4).tolist()
            },
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [{"lr": 0.001, "weight_decay": 0.01}]
            },
            "epoch": 20,
            "round": round_num,
            "step": 1000,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "metrics": {
                "train_pref_acc": 0.75,
                "val_pref_acc": 0.70,
                "tm_mean": 0.45,
                "rmsd_mean": 3.2,
                "mfe_mean": -12.5
            }
        }
        
        checkpoint_file = self.data_dir / f"checkpoint_round_{round_num}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        return checkpoint_file
    
    def _mutate_sequence(self, sequence: str, mutation_rate: float) -> str:
        """Introduce mutations in sequence."""
        mutated = list(sequence)
        n_mutations = int(len(sequence) * mutation_rate)
        
        positions = np.random.choice(len(sequence), n_mutations, replace=False)
        for pos in positions:
            # Replace with random nucleotide
            current_nt = mutated[pos]
            available_nts = [nt for nt in MockRNA.NUCLEOTIDES if nt != current_nt]
            mutated[pos] = np.random.choice(available_nts)
        
        return ''.join(mutated)
    
    def _generate_metrics(self, quality: str, margin: float) -> Dict[str, float]:
        """Generate realistic metrics with appropriate quality."""
        if quality == "good":
            base_plddt = np.random.uniform(0.7, 0.9)
            base_rmsd = np.random.uniform(1.0, 4.0)
            base_mfe = np.random.uniform(-20.0, -10.0)
        else:  # poor
            base_plddt = np.random.uniform(0.3, 0.6)
            base_rmsd = np.random.uniform(6.0, 15.0)
            base_mfe = np.random.uniform(-8.0, -2.0)
        
        # Add some noise but ensure margin
        noise_factor = margin * 0.5
        plddt = base_plddt + np.random.normal(0, noise_factor)
        rmsd = base_rmsd + np.random.normal(0, noise_factor * 2)
        mfe = base_mfe + np.random.normal(0, noise_factor * 3)
        
        return {
            "plddt": max(0.0, min(1.0, plddt)),
            "rmsd": max(0.5, rmsd),
            "mfe": mfe,
            "tm_score": np.random.uniform(0.2, 0.8),
            "gdt": np.random.uniform(0.2, 0.8)
        }
    
    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """Save data as JSONL format."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


class QuickMockData:
    """Quick mock data for simple tests."""
    
    @staticmethod
    def minimal_config():
        """Minimal working configuration."""
        return {
            "device": "cpu",
            "seed": 42,
            "multiround": {
                "num_rounds": 2,
                "epochs_per_round": 2,
                "update_reference": True
            },
            "dpo": {"beta": 0.1, "sft_lambda": 0.1},
            "training": {"batch_size": 2, "epochs": 2},
            "paths": {"save_dir": "/tmp/test"}
        }
    
    @staticmethod
    def sample_preference_pair():
        """Single preference pair for testing."""
        return {
            "backbone_id": "test_001",
            "chosen": "AUGCUGCAUGCA",
            "rejected": "AUGCUGCAUGCU",
            "chosen_metrics": {"plddt": 0.8, "rmsd": 2.0, "mfe": -15.0},
            "rejected_metrics": {"plddt": 0.6, "rmsd": 5.0, "mfe": -10.0}
        }


def setup_complete_mock_environment(temp_dir: Path, n_pairs: int = 50) -> Dict[str, Path]:
    """Set up complete mock environment for testing."""
    dataset = MockDataset(temp_dir)
    
    # Create preference pairs for both margins
    pairs_25_dir = dataset.create_preference_pairs(n_pairs, "25")
    pairs_125_dir = dataset.create_preference_pairs(n_pairs, "125")
    
    # Create test dataset
    test_file = dataset.create_test_dataset(20)
    
    # Create some checkpoints
    checkpoints = []
    for round_num in [1, 2, 3]:
        ckpt = dataset.create_model_checkpoint(round_num)
        checkpoints.append(ckpt)
    
    return {
        "pairs_margin25": pairs_25_dir,
        "pairs_margin125": pairs_125_dir,
        "test_dataset": test_file,
        "checkpoints": checkpoints,
        "data_dir": temp_dir
    }


if __name__ == "__main__":
    # Test mock data generation
    print("Testing mock data generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test complete setup
        mock_env = setup_complete_mock_environment(temp_path, n_pairs=10)
        
        # Verify files exist
        assert mock_env["pairs_margin25"].exists()
        assert mock_env["pairs_margin125"].exists()
        assert mock_env["test_dataset"].exists()
        assert len(mock_env["checkpoints"]) == 3
        
        print("✓ Mock environment setup successful")
        
        # Test individual components
        dataset = MockDataset(temp_path / "individual_test")
        
        # Test sequence generation
        seq = MockRNA.random_sequence(20)
        assert len(seq) == 20
        assert all(nt in MockRNA.NUCLEOTIDES for nt in seq)
        print("✓ Sequence generation works")
        
        # Test coordinates
        coords = MockRNA.create_coordinates(15)
        assert coords.shape == (15, 3)
        print("✓ Coordinate generation works")
        
        print("All mock data generation working correctly!")