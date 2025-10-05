#!/usr/bin/env python3
"""
RiboPO v2 Candidate Evaluation Pipeline

Integrates candidate generation with winner selection system.
Handles the complete pipeline from sequence generation to preference pair creation.
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

# Add multiround to path
multiround_path = project_root / "multiround"
sys.path.insert(0, str(multiround_path))

from multiround.evaluator import MultiRoundEvaluator
from multiround.utils import load_config_with_inheritance
from ribopo_v2.winner_selection import (
    WinnerSelector, WinnerConfig, CandidateMetrics, 
    create_winner_summary
)
from types import SimpleNamespace as SN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

class RiboPOv2CandidateEvaluator:
    """
    Complete RiboPO v2 candidate evaluation pipeline.
    
    Workflow:
    1. Load backbone structures from test set
    2. Generate candidate sequences using gRNAde (multiple temperatures)
    3. Evaluate candidates using comprehensive metrics
    4. Apply winner selection with S_total calculation
    5. Create preference pairs for DPO training
    """
    
    def __init__(self, config):
        """Initialize evaluator with configuration"""
        if isinstance(config, str):
            # If config_path is passed, load it
            self.config_path = config
            self.config = self._load_config(config)
        else:
            # If config object is passed, use it directly
            self.config = config
            self.config_path = getattr(config, 'config_path', None)
        
        # Initialize multiround evaluator for sequence generation
        self.evaluator = MultiRoundEvaluator(self.config)
        
        # Initialize winner selector
        winner_config = self._create_winner_config()
        self.winner_selector = WinnerSelector(winner_config)
        
        logger.info(f"Initialized RiboPO v2 evaluator with {len(self.evaluator.eval_dataset)} backbones")
    
    def _load_config(self, config_path: str) -> SN:
        """Load configuration with inheritance support"""
        config_dict = load_config_with_inheritance(config_path)
        return _to_sn(config_dict)
    
    def _create_winner_config(self) -> WinnerConfig:
        """Create winner configuration from main config"""
        ribopo_cfg = self.config.ribopo_v2
        
        return WinnerConfig(
            tm_weight=getattr(ribopo_cfg.reward_formula, 'tm_weight', 0.50),
            inf_weight=getattr(ribopo_cfg.reward_formula, 'inf_weight', 0.30),
            ed_weight=getattr(ribopo_cfg.reward_formula, 'ed_weight', 0.20),
            candidates_per_backbone=getattr(ribopo_cfg, 'candidate_pool_size', 200),
            winners_per_backbone=getattr(ribopo_cfg, 'winner_pool_size', 10),
            negatives_per_winner=getattr(ribopo_cfg, 'negatives_per_winner', 3),
            # Adjusted quality gates for mock data distributions
            min_tm_threshold=0.10,    # Lower threshold for mock TM-scores (mean 0.3, std 0.15)
            max_rmsd_threshold=30.0,  # Higher threshold for mock RMSD lognormal(2.0, 0.8) 
            min_plddt_threshold=0.50  # Lower threshold for mock pLDDT (mean 0.75, std 0.15)
        )
    
    def generate_candidates_for_backbone(self, 
                                       backbone_data: Dict[str, Any],
                                       backbone_idx: int) -> List[str]:
        """
        Generate candidate sequences for a single backbone using multi-temperature sampling.
        
        Args:
            backbone_data: Raw backbone structure data
            backbone_idx: Index in the evaluation dataset
            
        Returns:
            List of candidate sequences (empty list if base sequence is empty)
        """
        # Check if base sequence is valid
        base_sequence = backbone_data.get('sequence', '')
        if not base_sequence or len(base_sequence.strip()) == 0:
            logger.warning(f"Skipping backbone {backbone_idx} due to empty base sequence")
            return []
        
        # Get temperature diversity from config
        temperatures = getattr(self.config.ribopo_v2, 'temperature_diversity', [0.5])
        candidates_per_temp = self.config.ribopo_v2.candidate_pool_size // len(temperatures)
        
        all_candidates = []
        
        # Generate candidates at each temperature
        for temp in temperatures:
            logger.info(f"Generating {candidates_per_temp} candidates at T={temp} for backbone {backbone_idx}")
            
            # Note: This is a placeholder for actual sequence generation
            # In practice, this would use the model to generate sequences
            # For now, we'll create mock sequences for testing
            for i in range(candidates_per_temp):
                # Mock sequence generation (replace with actual model.sample() call)
                mock_sequence = self._create_mock_variant(base_sequence, i, temp)
                if mock_sequence and len(mock_sequence) > 0:  # Only add non-empty sequences
                    all_candidates.append(mock_sequence)
        
        logger.info(f"Generated {len(all_candidates)} total candidates for backbone {backbone_idx}")
        return all_candidates
    
    def _create_mock_variant(self, base_sequence: str, variant_idx: int, temperature: float) -> str:
        """Create mock sequence variant for testing (replace with actual generation)"""
        import random
        random.seed(variant_idx + int(temperature * 100))  # Deterministic for testing
        
        sequence = list(base_sequence)
        # Make small random changes based on temperature
        n_changes = max(1, int(len(sequence) * temperature * 0.1))
        
        for _ in range(n_changes):
            if len(sequence) > 0:
                pos = random.randint(0, len(sequence) - 1)
                sequence[pos] = random.choice(['A', 'U', 'G', 'C'])
        
        return ''.join(sequence)
    
    def evaluate_candidates_comprehensive(self, 
                                        candidates: List[str],
                                        backbone_data: Dict[str, Any],
                                        backbone_idx: int) -> List[CandidateMetrics]:
        """
        Evaluate candidates using comprehensive metrics pipeline.
        
        Note: This is a simplified version. In practice, this would:
        1. Use RhoFold for structure prediction
        2. Calculate TM-score, INF, ED using the evaluation pipeline
        3. Apply all the metrics from src/evaluator.py
        """
        backbone_id = f"backbone_{backbone_idx}_{backbone_data.get('id_list', ['unknown'])[0]}"
        evaluated_candidates = []
        
        logger.info(f"Evaluating {len(candidates)} candidates for {backbone_id}")
        
        for i, candidate_seq in enumerate(candidates):
            candidate_id = f"{backbone_id}_cand_{i}"
            
            # Evaluate sequence (real or mock based on config)
            metrics = self._evaluate_sequence(candidate_seq, backbone_data, backbone_id)
            
            candidate_metrics = CandidateMetrics(
                sequence=candidate_seq,
                backbone_id=backbone_id,
                candidate_id=candidate_id,
                tm_score=metrics['tm_score'],
                inf_all=metrics['inf_all'],
                ed_per_nt=metrics['ed_per_nt'],
                rmsd=metrics['rmsd'],
                plddt=metrics['plddt']
            )
            
            evaluated_candidates.append(candidate_metrics)
        
        logger.info(f"Completed evaluation for {len(evaluated_candidates)} candidates")
        return evaluated_candidates
    
    def _evaluate_sequence(self, sequence: str, backbone_data: Dict[str, Any], backbone_id: str) -> Dict[str, float]:
        """
        Evaluate sequence using real or mock metrics based on configuration.
        """
        # Check if we should use real metrics
        use_real = getattr(self.config.ribopo_v2, 'use_real_metrics', False)
        
        if use_real:
            # Use real metrics evaluation
            if not hasattr(self, '_metrics_evaluator'):
                from ribopo_v2.metrics_evaluation import RealMetricsEvaluator
                cache_dir = getattr(self.config.ribopo_v2, 'metrics_cache_dir', 'ribopo_v2/cache/metrics')
                self._metrics_evaluator = RealMetricsEvaluator(cache_dir=cache_dir)
            
            result = self._metrics_evaluator.evaluate_sequence(
                sequence,
                backbone_data,
                backbone_id,
                use_cache=getattr(self.config.ribopo_v2, 'use_metrics_cache', True)
            )
            
            return {
                'tm_score': result.tm_score,
                'inf_all': result.inf_all,
                'ed_per_nt': result.ed_per_nt,
                'rmsd': result.rmsd,
                'plddt': result.plddt
            }
        else:
            # Use mock evaluation for testing
            return self._mock_evaluate_sequence(sequence, backbone_data)
    
    def _mock_evaluate_sequence(self, sequence: str, backbone_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Mock evaluation function for testing.
        """
        import random
        seed_val = hash(sequence) % 2**32
        random.seed(seed_val)  # Deterministic based on sequence
        np.random.seed(seed_val)
        
        # Mock realistic RNA metrics
        tm_score = max(0.0, min(1.0, random.gauss(0.3, 0.15)))  # Centered around 0.3
        inf_all = max(-1.0, min(1.0, random.gauss(0.2, 0.3)))   # Can be negative
        ed_per_nt = max(0.0, np.random.exponential(0.4))        # Fixed: use np.random.exponential
        rmsd = max(0.0, np.random.lognormal(2.0, 0.8))          # Fixed: use np.random.lognormal
        plddt = max(0.0, min(1.0, random.gauss(0.75, 0.15)))   # Centered around 0.75
        
        return {
            'tm_score': tm_score,
            'inf_all': inf_all,
            'ed_per_nt': ed_per_nt,
            'rmsd': rmsd,
            'plddt': plddt
        }
    
    def process_backbone(self, backbone_idx: int) -> Tuple[List[CandidateMetrics], List[Dict[str, Any]]]:
        """
        Complete processing pipeline for a single backbone:
        1. Generate candidates
        2. Evaluate with comprehensive metrics
        3. Select winners using S_total
        4. Create preference pairs
        """
        # Get backbone data from evaluation dataset
        backbone_data = self.evaluator.eval_dataset.data_list[backbone_idx]
        backbone_id = f"backbone_{backbone_idx}_{backbone_data.get('id_list', ['unknown'])[0]}"
        
        logger.info(f"Processing backbone {backbone_idx}: {backbone_id}")
        
        # Step 1: Generate candidates
        candidate_sequences = self.generate_candidates_for_backbone(backbone_data, backbone_idx)
        
        # Skip if no valid candidates generated (e.g., due to empty base sequence)
        if not candidate_sequences:
            logger.warning(f"Skipping backbone {backbone_idx} - no valid candidates generated")
            return [], []
        
        # Step 2: Evaluate candidates
        evaluated_candidates = self.evaluate_candidates_comprehensive(
            candidate_sequences, backbone_data, backbone_idx
        )
        
        # Skip if no candidates passed evaluation
        if not evaluated_candidates:
            logger.warning(f"Skipping backbone {backbone_idx} - no candidates passed evaluation")
            return [], []
        
        # Step 3 & 4: Winner selection and pair creation
        processed_candidates, preference_pairs = self.winner_selector.process_backbone(
            evaluated_candidates, backbone_id
        )
        
        return processed_candidates, preference_pairs
    
    def run_candidate_evaluation(self, 
                                limit_backbones: Optional[int] = None,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete candidate evaluation pipeline across all test backbones.
        
        Args:
            limit_backbones: Limit to first N backbones for testing
            output_dir: Directory to save results
            
        Returns:
            Comprehensive results dictionary
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"ribopo_v2/results/candidate_eval_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting candidate evaluation pipeline")
        logger.info(f"Output directory: {output_path}")
        
        # Determine how many backbones to process
        n_backbones = len(self.evaluator.eval_dataset)
        if limit_backbones:
            n_backbones = min(limit_backbones, n_backbones)
        
        logger.info(f"Processing {n_backbones} backbones")
        
        # Process each backbone
        all_candidates = []
        all_pairs = []
        
        for backbone_idx in range(n_backbones):
            try:
                candidates, pairs = self.process_backbone(backbone_idx)
                all_candidates.extend(candidates)
                all_pairs.extend(pairs)
                
                logger.info(f"Completed backbone {backbone_idx}: "
                          f"{len([c for c in candidates if c.is_winner])} winners, "
                          f"{len(pairs)} pairs")
                
            except Exception as e:
                logger.error(f"Failed to process backbone {backbone_idx}: {e}")
                continue
        
        # Generate comprehensive summary
        summary = create_winner_summary(all_candidates, all_pairs)
        
        # Save results
        results = {
            'config': asdict(self.winner_selector.config),
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'n_backbones_processed': n_backbones,
            'total_candidates': len(all_candidates),
            'total_winners': len([c for c in all_candidates if c.is_winner]),
            'total_pairs': len(all_pairs)
        }
        
        # Save main results
        results_path = output_path / "candidate_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save preference pairs for training
        pairs_path = output_path / "preference_pairs.jsonl"
        with open(pairs_path, 'w') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + '\n')
        
        # Save winner details
        winners = [c for c in all_candidates if c.is_winner]
        winners_data = []
        for winner in winners:
            winner_dict = asdict(winner)
            winners_data.append(winner_dict)
        
        winners_path = output_path / "winners.json"
        with open(winners_path, 'w') as f:
            json.dump(winners_data, f, indent=2)
        
        logger.info(f"\n✅ Candidate evaluation complete!")
        logger.info(f"   Results saved to: {output_path}")
        logger.info(f"   Total winners: {len(winners)}")
        logger.info(f"   Total pairs: {len(all_pairs)}")
        logger.info(f"   Avg winners per backbone: {len(winners) / n_backbones:.1f}")
        
        return results

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiboPO v2 Candidate Evaluation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="ribopo_v2/config/experiments/01_candidate_generation.yaml",
        help="Path to RiboPO v2 configuration file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit to first N backbones for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RiboPOv2CandidateEvaluator(args.config)
    
    # Run evaluation
    results = evaluator.run_candidate_evaluation(
        limit_backbones=args.limit,
        output_dir=args.output_dir
    )
    
    print(f"\n🎯 RiboPO v2 Candidate Evaluation Complete!")
    print(f"   Configuration: {args.config}")
    print(f"   Winners selected: {results['total_winners']}")
    print(f"   Preference pairs: {results['total_pairs']}")

if __name__ == "__main__":
    main()