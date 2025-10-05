#!/usr/bin/env python3
"""
RiboPO v2 Winner Selection Module

Implements the winner-focused approach with S_total reward calculation:
S_total = 0.50*TM + 0.30*INF + 0.20*ED

Key Features:
- Per-backbone normalization via percentile ranking
- Best-vs-Random pairing strategy  
- Top-K winner selection per backbone
- Quality gates and validation
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WinnerConfig:
    """Configuration for winner selection"""
    # Reward formula weights (must sum to 1.0)
    tm_weight: float = 0.50      # TM-score weight (structure fidelity)
    inf_weight: float = 0.30     # INF_ALL weight (interaction fidelity)  
    ed_weight: float = 0.20      # ED/L weight (ensemble defect per nucleotide)
    
    # Winner selection parameters
    candidates_per_backbone: int = 200    # Total candidates to generate
    winners_per_backbone: int = 10        # Top winners to select
    negatives_per_winner: int = 3         # Random negatives per winner for pairing
    
    # Quality gates (loose filters)
    min_tm_threshold: float = 0.20        # Minimum TM-score filter
    max_rmsd_threshold: float = 12.0      # Maximum RMSD filter (Å)
    min_plddt_threshold: float = 0.60     # Minimum pLDDT for down-weighting
    
    # Normalization parameters
    normalization_method: str = "percentile"  # "percentile" or "zscore"
    winsorize_percentiles: Tuple[float, float] = (5.0, 95.0)  # Outlier handling
    
    # Pairing strategy
    pairing_method: str = "best_vs_random"    # "best_vs_random" or "semi_hard"
    margin_method: str = "mad"                # "mad" (median absolute deviation) or "std"
    margin_multiplier: float = 0.8           # Multiplier for margin calculation

    def __post_init__(self):
        """Validate configuration"""
        total_weight = self.tm_weight + self.inf_weight + self.ed_weight
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Reward weights must sum to 1.0, got {total_weight}")

@dataclass  
class CandidateMetrics:
    """Metrics for a single candidate sequence"""
    sequence: str
    backbone_id: str
    candidate_id: str
    
    # Raw metrics
    tm_score: float
    inf_all: float  
    ed_per_nt: float
    rmsd: float
    plddt: float
    
    # Normalized metrics (computed later)
    tm_norm: Optional[float] = None
    inf_norm: Optional[float] = None
    ed_norm: Optional[float] = None
    
    # Final score
    s_total: Optional[float] = None
    is_winner: bool = False
    is_valid: bool = True

class WinnerSelector:
    """
    RiboPO v2 Winner Selection System
    
    Implements the core winner-focused approach:
    1. Generate large candidate pools (200+ per backbone)
    2. Compute S_total with per-backbone normalization
    3. Select top-K winners per backbone
    4. Create best-vs-random preference pairs
    """
    
    def __init__(self, config: WinnerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def normalize_metrics_per_backbone(self, 
                                     candidates: List[CandidateMetrics],
                                     backbone_id: str) -> List[CandidateMetrics]:
        """
        Apply per-backbone normalization using percentile ranking.
        This handles variable-length RNAs and ensures fair comparison.
        """
        backbone_candidates = [c for c in candidates if c.backbone_id == backbone_id]
        if len(backbone_candidates) < 2:
            self.logger.warning(f"Insufficient candidates for backbone {backbone_id}: {len(backbone_candidates)}")
            return backbone_candidates
            
        # Extract raw metrics
        tm_scores = np.array([c.tm_score for c in backbone_candidates])
        inf_scores = np.array([c.inf_all for c in backbone_candidates]) 
        ed_scores = np.array([c.ed_per_nt for c in backbone_candidates])
        
        # Apply winsorization to handle outliers
        if self.config.winsorize_percentiles:
            p_low, p_high = self.config.winsorize_percentiles
            tm_scores = self._winsorize(tm_scores, p_low, p_high)
            inf_scores = self._winsorize(inf_scores, p_low, p_high)
            ed_scores = self._winsorize(ed_scores, p_low, p_high)
        
        # Normalize using percentile ranking (0-1 scale)
        if self.config.normalization_method == "percentile":
            tm_norm = self._percentile_rank(tm_scores)  # Higher is better
            inf_norm = self._percentile_rank(inf_scores)  # Higher is better  
            ed_norm = 1.0 - self._percentile_rank(ed_scores)  # Lower ED is better, so invert
        elif self.config.normalization_method == "zscore":
            tm_norm = self._zscore_normalize(tm_scores)
            inf_norm = self._zscore_normalize(inf_scores)
            ed_norm = -self._zscore_normalize(ed_scores)  # Invert for ED (lower is better)
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
        
        # Update candidates with normalized scores
        for i, candidate in enumerate(backbone_candidates):
            candidate.tm_norm = float(tm_norm[i])
            candidate.inf_norm = float(inf_norm[i])
            candidate.ed_norm = float(ed_norm[i])
            
        return backbone_candidates
    
    def compute_s_total(self, candidates: List[CandidateMetrics]) -> List[CandidateMetrics]:
        """
        Compute S_total = 0.50*TM + 0.30*INF + 0.20*ED for all candidates.
        Requires normalized metrics to be computed first.
        """
        for candidate in candidates:
            if None in [candidate.tm_norm, candidate.inf_norm, candidate.ed_norm]:
                raise ValueError(f"Normalized metrics missing for candidate {candidate.candidate_id}")
                
            # Compute S_total using weighted combination
            candidate.s_total = (
                self.config.tm_weight * candidate.tm_norm +
                self.config.inf_weight * candidate.inf_norm + 
                self.config.ed_weight * candidate.ed_norm
            )
            
            # Apply quality gates
            candidate.is_valid = self._apply_quality_gates(candidate)
            
        return candidates
    
    def select_winners_per_backbone(self, 
                                   candidates: List[CandidateMetrics],
                                   backbone_id: str) -> List[CandidateMetrics]:
        """
        Select top-K winners per backbone based on S_total score.
        Ensures every backbone contributes at least 1 winner.
        """
        backbone_candidates = [c for c in candidates 
                             if c.backbone_id == backbone_id and c.is_valid]
        
        if len(backbone_candidates) == 0:
            self.logger.warning(f"No valid candidates for backbone {backbone_id}")
            return []
            
        # Sort by S_total (descending)
        backbone_candidates.sort(key=lambda x: x.s_total, reverse=True)
        
        # Select top winners (at least 1, up to configured amount)
        n_winners = min(self.config.winners_per_backbone, len(backbone_candidates))
        n_winners = max(1, n_winners)  # Ensure at least 1 winner
        
        winners = backbone_candidates[:n_winners]
        for winner in winners:
            winner.is_winner = True
            
        self.logger.info(f"Selected {len(winners)} winners for backbone {backbone_id} "
                        f"(S_total range: {winners[-1].s_total:.3f} - {winners[0].s_total:.3f})")
        
        return winners
    
    def create_preference_pairs(self, 
                              candidates: List[CandidateMetrics],
                              backbone_id: str) -> List[Dict[str, Any]]:
        """
        Create best-vs-random preference pairs for DPO training.
        Uses simple strategy: each winner paired with M random non-winners.
        """
        backbone_candidates = [c for c in candidates if c.backbone_id == backbone_id]
        winners = [c for c in backbone_candidates if c.is_winner]
        non_winners = [c for c in backbone_candidates if not c.is_winner and c.is_valid]
        
        if len(winners) == 0 or len(non_winners) == 0:
            self.logger.warning(f"Insufficient candidates for pairing in backbone {backbone_id}: "
                              f"winners={len(winners)}, non_winners={len(non_winners)}")
            return []
        
        pairs = []
        
        # Compute margin for validation
        if len(backbone_candidates) >= 2:
            s_scores = [c.s_total for c in backbone_candidates if c.s_total is not None]
            if self.config.margin_method == "mad":
                margin = self.config.margin_multiplier * stats.median_abs_deviation(s_scores)
            else:  # std
                margin = self.config.margin_multiplier * np.std(s_scores)
        else:
            margin = 0.0
        
        # Create pairs: each winner vs M random negatives
        for winner in winners:
            # Sample negatives (with replacement if needed)
            n_negs = min(self.config.negatives_per_winner, len(non_winners))
            if n_negs == 0:
                continue
                
            negatives = np.random.choice(non_winners, size=n_negs, replace=(n_negs > len(non_winners)))
            
            for negative in negatives:
                delta_s = winner.s_total - negative.s_total
                
                # Optional: apply margin filter (currently disabled for simplicity)
                # if delta_s < margin:
                #     continue
                    
                pair = {
                    'backbone_id': backbone_id,
                    'winner_id': winner.candidate_id,
                    'winner_sequence': winner.sequence,
                    'winner_s_total': winner.s_total,
                    'loser_id': negative.candidate_id, 
                    'loser_sequence': negative.sequence,
                    'loser_s_total': negative.s_total,
                    'delta_s_total': delta_s,
                    'margin': margin,
                    'winner_metrics': {
                        'tm_score': winner.tm_score,
                        'inf_all': winner.inf_all,
                        'ed_per_nt': winner.ed_per_nt,
                        'rmsd': winner.rmsd,
                        'plddt': winner.plddt
                    },
                    'loser_metrics': {
                        'tm_score': negative.tm_score,
                        'inf_all': negative.inf_all, 
                        'ed_per_nt': negative.ed_per_nt,
                        'rmsd': negative.rmsd,
                        'plddt': negative.plddt
                    }
                }
                pairs.append(pair)
        
        self.logger.info(f"Created {len(pairs)} preference pairs for backbone {backbone_id}")
        return pairs
    
    def process_backbone(self, 
                        candidates: List[CandidateMetrics],
                        backbone_id: str) -> Tuple[List[CandidateMetrics], List[Dict[str, Any]]]:
        """
        Complete processing pipeline for a single backbone:
        1. Normalize metrics per backbone
        2. Compute S_total scores
        3. Select winners
        4. Create preference pairs
        """
        self.logger.info(f"Processing backbone {backbone_id} with {len(candidates)} candidates")
        
        # Step 1: Normalize metrics
        normalized_candidates = self.normalize_metrics_per_backbone(candidates, backbone_id)
        
        # Step 2: Compute S_total
        scored_candidates = self.compute_s_total(normalized_candidates)
        
        # Step 3: Select winners
        winners = self.select_winners_per_backbone(scored_candidates, backbone_id)
        
        # Step 4: Create preference pairs
        pairs = self.create_preference_pairs(scored_candidates, backbone_id)
        
        return scored_candidates, pairs
    
    def _apply_quality_gates(self, candidate: CandidateMetrics) -> bool:
        """Apply quality gates to filter out poor candidates"""
        # Basic quality gates (loose filters)
        if candidate.tm_score < self.config.min_tm_threshold:
            return False
        if candidate.rmsd > self.config.max_rmsd_threshold:
            return False
        # Note: pLDDT used for down-weighting, not hard filtering
        return True
    
    def _winsorize(self, data: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        """Apply winsorization to handle outliers"""
        low, high = np.percentile(data, [p_low, p_high])
        return np.clip(data, low, high)
    
    def _percentile_rank(self, data: np.ndarray) -> np.ndarray:
        """Convert to percentile ranks (0-1 scale)"""
        return stats.rankdata(data, method='average') / len(data)
    
    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

def create_winner_summary(all_candidates: List[CandidateMetrics], 
                         all_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for winner selection results"""
    
    winners = [c for c in all_candidates if c.is_winner]
    valid_candidates = [c for c in all_candidates if c.is_valid]
    
    # Backbone-level stats
    backbone_stats = {}
    backbones = set(c.backbone_id for c in all_candidates)
    
    for backbone_id in backbones:
        backbone_candidates = [c for c in all_candidates if c.backbone_id == backbone_id]
        backbone_winners = [c for c in backbone_candidates if c.is_winner]
        backbone_pairs = [p for p in all_pairs if p['backbone_id'] == backbone_id]
        
        backbone_stats[backbone_id] = {
            'total_candidates': len(backbone_candidates),
            'valid_candidates': len([c for c in backbone_candidates if c.is_valid]),
            'winners_selected': len(backbone_winners),
            'pairs_created': len(backbone_pairs),
            'winner_s_total_range': [
                min(w.s_total for w in backbone_winners) if backbone_winners else 0,
                max(w.s_total for w in backbone_winners) if backbone_winners else 0
            ]
        }
    
    summary = {
        'total_candidates': len(all_candidates),
        'valid_candidates': len(valid_candidates),
        'total_winners': len(winners),
        'total_pairs': len(all_pairs),
        'backbones_processed': len(backbones),
        'avg_candidates_per_backbone': len(all_candidates) / len(backbones) if backbones else 0,
        'avg_winners_per_backbone': len(winners) / len(backbones) if backbones else 0,
        'avg_pairs_per_backbone': len(all_pairs) / len(backbones) if backbones else 0,
        'winner_quality_stats': {
            'tm_score': {
                'mean': float(np.mean([w.tm_score for w in winners])) if winners else 0,
                'std': float(np.std([w.tm_score for w in winners])) if winners else 0,
                'min': float(np.min([w.tm_score for w in winners])) if winners else 0,
                'max': float(np.max([w.tm_score for w in winners])) if winners else 0
            },
            'inf_all': {
                'mean': float(np.mean([w.inf_all for w in winners])) if winners else 0,
                'std': float(np.std([w.inf_all for w in winners])) if winners else 0,
                'min': float(np.min([w.inf_all for w in winners])) if winners else 0,
                'max': float(np.max([w.inf_all for w in winners])) if winners else 0
            },
            'ed_per_nt': {
                'mean': float(np.mean([w.ed_per_nt for w in winners])) if winners else 0,
                'std': float(np.std([w.ed_per_nt for w in winners])) if winners else 0,
                'min': float(np.min([w.ed_per_nt for w in winners])) if winners else 0,
                'max': float(np.max([w.ed_per_nt for w in winners])) if winners else 0
            },
            's_total': {
                'mean': float(np.mean([w.s_total for w in winners])) if winners else 0,
                'std': float(np.std([w.s_total for w in winners])) if winners else 0,
                'min': float(np.min([w.s_total for w in winners])) if winners else 0,
                'max': float(np.max([w.s_total for w in winners])) if winners else 0
            }
        },
        'backbone_stats': backbone_stats
    }
    
    return summary

# Example usage and testing functions
def create_test_candidates(n_backbones: int = 3, n_candidates_per_backbone: int = 20) -> List[CandidateMetrics]:
    """Create synthetic test candidates for validation"""
    candidates = []
    
    for backbone_idx in range(n_backbones):
        backbone_id = f"test_backbone_{backbone_idx}"
        
        for cand_idx in range(n_candidates_per_backbone):
            candidate_id = f"{backbone_id}_cand_{cand_idx}"
            
            # Generate realistic RNA metrics with some variability
            tm_score = np.random.beta(2, 5) * 0.8 + 0.1  # Skewed toward lower values
            inf_all = np.random.beta(3, 2) * 0.9 + 0.1   # Skewed toward higher values
            ed_per_nt = np.random.exponential(0.3)        # Exponential distribution
            rmsd = np.random.lognormal(1.5, 0.5)          # Log-normal for RMSD
            plddt = np.random.beta(5, 2) * 0.4 + 0.5      # Concentrated around 0.7-0.9
            
            candidate = CandidateMetrics(
                sequence="A" * (20 + cand_idx),  # Dummy sequences
                backbone_id=backbone_id,
                candidate_id=candidate_id,
                tm_score=tm_score,
                inf_all=inf_all,
                ed_per_nt=ed_per_nt,
                rmsd=rmsd,
                plddt=plddt
            )
            candidates.append(candidate)
    
    return candidates

if __name__ == "__main__":
    # Test the winner selection system
    print("🧪 Testing RiboPO v2 Winner Selection System")
    
    # Create test data
    config = WinnerConfig()
    selector = WinnerSelector(config)
    test_candidates = create_test_candidates(n_backbones=3, n_candidates_per_backbone=50)
    
    print(f"📊 Created {len(test_candidates)} test candidates across 3 backbones")
    
    # Process each backbone
    all_processed_candidates = []
    all_pairs = []
    
    backbones = set(c.backbone_id for c in test_candidates)
    for backbone_id in backbones:
        backbone_candidates = [c for c in test_candidates if c.backbone_id == backbone_id]
        processed_candidates, pairs = selector.process_backbone(backbone_candidates, backbone_id)
        
        all_processed_candidates.extend(processed_candidates)
        all_pairs.extend(pairs)
    
    # Generate summary
    summary = create_winner_summary(all_processed_candidates, all_pairs)
    
    print(f"\n✅ Winner Selection Complete!")
    print(f"   Total winners: {summary['total_winners']}")
    print(f"   Total preference pairs: {summary['total_pairs']}")
    print(f"   Avg winners per backbone: {summary['avg_winners_per_backbone']:.1f}")
    print(f"   Winner S_total range: {summary['winner_quality_stats']['s_total']['min']:.3f} - {summary['winner_quality_stats']['s_total']['max']:.3f}")
    
    print("\n🎯 RiboPO v2 Winner Selection System Ready!")