# RiboPO v2 Detailed Implementation Workflow

**Generated**: 2025-01-04
**Framework**: Winner-focused DPO for RNA inverse folding
**Core Insight**: DPO learning signal dominated by winner quality, not negative complexity

---

## 🔬 Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     RiboPO v2 Pipeline                       │
├───────────────┬───────────────┬─────────────┬──────────────┤
│  Data Clean   │  Candidate    │   Metric    │  Training    │
│  & Filter     │  Generation   │  Evaluation │  & DPO       │
│  ✅ COMPLETE  │  🔄 Phase 1   │  🔄 Phase 2 │  🔄 Phase 6  │
└───────────────┴───────────────┴─────────────┴──────────────┘
                           ↓
              ┌──────────────────────────┐
              │   Winner Selection       │
              │   & Pair Construction    │
              │     🔄 Phases 3-5       │
              └──────────────────────────┘
```

---

## Phase 1: Candidate Generation Implementation

### 1.1 Set Up gRNAde Integration
```python
# ribopo_v2/candidate_generation/generate_candidates.py

import torch
from src.models.rna_model import RNADesignModel  # gRNAde model
from src.data.featurizer import RNAFeaturizer
import numpy as np
from typing import List, Dict, Tuple

class CandidateGenerator:
    def __init__(self, model_checkpoint: str, device: str = 'cuda'):
        self.model = RNADesignModel.from_pretrained(model_checkpoint)
        self.model.to(device)
        self.model.eval()
        self.featurizer = RNAFeaturizer()
        
    def generate_diverse_candidates(
        self, 
        backbone_pdb: str,
        n_candidates: int = 200,
        temperatures: List[float] = [0.1, 0.5, 1.0],
        strategies: List[str] = ['greedy', 'top_p', 'top_k']
    ) -> Dict[str, List[str]]:
        """
        Generate diverse candidate sequences for a backbone.
        
        Returns:
            Dict with keys: 'sequences', 'metadata', 'seeds'
        """
        candidates = []
        metadata = []
        
        # Load and featurize backbone
        features = self.featurizer.featurize_pdb(backbone_pdb)
        
        # Diversify across temperatures and strategies
        n_per_config = n_candidates // (len(temperatures) * len(strategies))
        
        for temp in temperatures:
            for strategy in strategies:
                batch = self._sample_batch(
                    features, 
                    n_samples=n_per_config,
                    temperature=temp,
                    strategy=strategy
                )
                candidates.extend(batch['sequences'])
                metadata.extend(batch['metadata'])
                
        return {
            'sequences': candidates,
            'metadata': metadata,
            'backbone': backbone_pdb
        }
```

### 1.2 Baseline Anchors Implementation
```python
# ribopo_v2/candidate_generation/baseline_anchors.py

import random
from Bio import SeqIO
from collections import Counter

def add_baseline_anchors(candidates: Dict, native_seq: str) -> Dict:
    """Add native sequence and GC-matched shuffle as baselines."""
    
    # Add native sequence
    candidates['sequences'].append(native_seq)
    candidates['metadata'].append({
        'type': 'native',
        'temperature': 0.0,
        'strategy': 'native'
    })
    
    # Generate GC-matched shuffle
    gc_shuffle = generate_gc_matched_shuffle(native_seq)
    candidates['sequences'].append(gc_shuffle)
    candidates['metadata'].append({
        'type': 'gc_shuffle',
        'temperature': 0.0,
        'strategy': 'shuffle'
    })
    
    return candidates

def generate_gc_matched_shuffle(seq: str) -> str:
    """Shuffle sequence while preserving exact nucleotide composition."""
    seq_list = list(seq)
    random.shuffle(seq_list)
    return ''.join(seq_list)
```

### 1.3 Diversity Metrics
```python
# ribopo_v2/candidate_generation/diversity_metrics.py

import itertools
from typing import List
import numpy as np

def calculate_trimer_diversity(sequences: List[str]) -> float:
    """Calculate 3-mer diversity across candidate pool."""
    
    all_trimers = [''.join(k) for k in itertools.product('ACGU', repeat=3)]
    trimer_counts = {t: 0 for t in all_trimers}
    
    for seq in sequences:
        for i in range(len(seq) - 2):
            trimer = seq[i:i+3]
            if trimer in trimer_counts:
                trimer_counts[trimer] += 1
    
    # Calculate Shannon entropy
    total = sum(trimer_counts.values())
    if total == 0:
        return 0.0
        
    probs = [count/total for count in trimer_counts.values() if count > 0]
    entropy = -sum(p * np.log(p) for p in probs)
    
    return entropy

def deduplicate_sequences(candidates: Dict) -> Dict:
    """Remove duplicate sequences while keeping metadata."""
    seen = set()
    unique_candidates = {
        'sequences': [],
        'metadata': [],
        'backbone': candidates['backbone']
    }
    
    for seq, meta in zip(candidates['sequences'], candidates['metadata']):
        if seq not in seen:
            seen.add(seq)
            unique_candidates['sequences'].append(seq)
            unique_candidates['metadata'].append(meta)
            
    return unique_candidates
```

---

## Phase 2: Metric Computation Pipeline

### 2.1 Batched RhoFold+ Evaluation
```python
# ribopo_v2/metric_computation/structure_prediction.py

import subprocess
import os
from pathlib import Path
import json

class RhoFoldBatchRunner:
    def __init__(self, rhofold_path: str, cache_dir: str):
        self.rhofold_path = rhofold_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def predict_batch(self, sequences: List[str], backbone_id: str) -> Dict:
        """Run RhoFold+ predictions for a batch of sequences."""
        
        results = {}
        cache_file = self.cache_dir / f"{backbone_id}_rhofold.json"
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                results.update(cached)
        
        # Process uncached sequences
        for i, seq in enumerate(sequences):
            seq_hash = hash(seq)
            if str(seq_hash) not in results:
                pdb_path, plddt = self._run_rhofold(seq)
                results[str(seq_hash)] = {
                    'sequence': seq,
                    'pdb_path': pdb_path,
                    'plddt': plddt.tolist()
                }
        
        # Save updated cache
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
```

### 2.2 US-align TM/RMSD/lDDT Computation
```python
# ribopo_v2/metric_computation/structural_metrics.py

from tools.usalign_utils import run_rna_usalign
import numpy as np

class StructuralMetricsCalculator:
    def __init__(self, native_pdb_paths: List[str]):
        self.native_pdbs = native_pdb_paths
        
    def calculate_metrics(self, predicted_pdb: str) -> Dict:
        """Calculate TM, RMSD, lDDT against all native structures."""
        
        tm_scores = []
        rmsd_values = []
        lddt_scores = []
        
        for native_pdb in self.native_pdbs:
            # Run US-align
            result = run_rna_usalign(predicted_pdb, native_pdb)
            
            # Aggregate TM scores (average of two normalized values)
            tm = 0.5 * (result.tmscore_chain1 + result.tmscore_chain2)
            tm_scores.append(tm)
            rmsd_values.append(result.rmsd)
            
            # Calculate lDDT separately
            lddt = calculate_lddt(predicted_pdb, native_pdb)
            lddt_scores.append(lddt)
        
        return {
            'tm_score': np.mean(tm_scores),
            'rmsd': np.mean(rmsd_values),
            'lddt': np.mean(lddt_scores),
            'tm_all': tm_scores,
            'rmsd_all': rmsd_values,
            'lddt_all': lddt_scores
        }
```

### 2.3 DSSR INF Analysis
```python
# ribopo_v2/metric_computation/interaction_metrics.py

import subprocess
import json
from pathlib import Path

class InteractionAnalyzer:
    def __init__(self, dssr_path: str):
        self.dssr_path = dssr_path
        
    def calculate_inf(self, predicted_pdb: str, native_pdbs: List[str]) -> Dict:
        """Calculate Interaction Network Fidelity metrics."""
        
        # Get interactions for predicted structure
        pred_interactions = self._run_dssr(predicted_pdb)
        
        inf_scores = {
            'all': [],
            'wc': [],
            'nwc': [],
            'stack': []
        }
        
        for native_pdb in native_pdbs:
            native_interactions = self._run_dssr(native_pdb)
            
            # Compare interaction networks
            scores = self._compare_interactions(pred_interactions, native_interactions)
            for key in inf_scores:
                inf_scores[key].append(scores[key])
        
        # Return mean INF scores
        return {
            f'inf_{key}': np.mean(values) if values else 0.0
            for key, values in inf_scores.items()
        }
    
    def _run_dssr(self, pdb_path: str) -> Dict:
        """Run DSSR and parse JSON output."""
        cmd = [self.dssr_path, '--json', pdb_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
```

### 2.4 ViennaRNA Ensemble Metrics
```python
# ribopo_v2/metric_computation/ensemble_metrics.py

import RNA  # ViennaRNA Python bindings

class EnsembleMetricsCalculator:
    def __init__(self, temperature: float = 37.0):
        self.temperature = temperature
        
    def calculate_ensemble_metrics(self, sequence: str, target_structure: str = None) -> Dict:
        """Calculate MFE, ensemble defect, Shannon entropy, P(target)."""
        
        # Create fold compound
        md = RNA.md()
        md.temperature = self.temperature
        fc = RNA.fold_compound(sequence, md)
        
        # MFE structure
        mfe_struct, mfe = fc.mfe()
        
        # Partition function for ensemble metrics
        fc.pf()
        
        # Use target structure if provided, else use MFE
        if target_structure is None:
            target_structure = mfe_struct
            
        # Ensemble defect
        ed = fc.ensemble_defect(target_structure)
        ed_per_nt = ed / len(sequence)
        
        # Probability of target structure
        p_target = fc.pr_structure(target_structure)
        
        # Positional Shannon entropy
        entropy = fc.positional_entropy()
        mean_entropy = np.mean(entropy[1:])  # Skip index 0
        
        # Mean base-pair distance (diversity)
        diversity = fc.mean_bp_distance()
        
        return {
            'mfe': mfe,
            'mfe_structure': mfe_struct,
            'ensemble_defect': ed,
            'ed_per_nt': ed_per_nt,
            'p_target': p_target,
            'shannon_entropy': mean_entropy,
            'bp_diversity': diversity
        }
```

---

## Phase 3: Normalization Implementation

### 3.1 Per-Backbone Normalization
```python
# ribopo_v2/normalization/normalize_metrics.py

import pandas as pd
import numpy as np
from scipy import stats

class MetricNormalizer:
    def __init__(self):
        self.normalization_params = {}
        
    def fit_normalization(self, metrics_df: pd.DataFrame) -> None:
        """Fit normalization parameters on training data."""
        
        # Group by backbone
        for backbone_id, group in metrics_df.groupby('backbone_id'):
            
            # Calculate percentile ranks for each metric
            params = {
                'tm': {
                    'values': group['tm_score'].values,
                    'percentiles': stats.rankdata(group['tm_score']) / len(group)
                },
                'inf': {
                    'values': group['inf_all'].values,
                    'percentiles': stats.rankdata(group['inf_all']) / len(group)
                },
                'ed': {
                    'values': 1 - group['ed_per_nt'].values,  # Convert to higher-is-better
                    'percentiles': stats.rankdata(1 - group['ed_per_nt']) / len(group)
                }
            }
            
            self.normalization_params[backbone_id] = params
            
    def normalize(self, metrics: Dict, backbone_id: str) -> Dict:
        """Apply normalization to new metrics."""
        
        if backbone_id not in self.normalization_params:
            raise ValueError(f"No normalization parameters for backbone {backbone_id}")
            
        params = self.normalization_params[backbone_id]
        
        # Percentile rank normalization
        tm_norm = self._get_percentile_rank(metrics['tm_score'], params['tm'])
        inf_norm = self._get_percentile_rank(metrics['inf_all'], params['inf'])
        ed_norm = self._get_percentile_rank(1 - metrics['ed_per_nt'], params['ed'])
        
        return {
            'tm_norm': tm_norm,
            'inf_norm': inf_norm,
            'ed_norm': ed_norm
        }
    
    def _get_percentile_rank(self, value: float, params: Dict) -> float:
        """Calculate percentile rank for a value."""
        return (params['values'] < value).sum() / len(params['values'])
```

### 3.2 Reward Calculation
```python
# ribopo_v2/normalization/reward_calculator.py

class RewardCalculator:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'tm': 0.50,
            'inf': 0.30,
            'ed': 0.20
        }
        
    def calculate_reward(self, normalized_metrics: Dict) -> float:
        """Calculate S_total reward score."""
        
        s_total = (
            self.weights['tm'] * normalized_metrics['tm_norm'] +
            self.weights['inf'] * normalized_metrics['inf_norm'] +
            self.weights['ed'] * normalized_metrics['ed_norm']
        )
        
        return s_total
    
    def apply_gates(self, metrics: Dict, reward: float) -> Tuple[float, bool]:
        """Apply loose gates and pLDDT down-weighting."""
        
        # TM gate (or RMSD alternative)
        if metrics['tm_score'] < 0.20 and metrics['rmsd'] > 12.0:
            return 0.0, False  # Failed gate
            
        # pLDDT down-weighting
        if metrics.get('plddt', 1.0) < 0.6:
            weight = metrics['plddt'] / 0.6  # Linear down-weight
            reward *= weight
            
        return reward, True  # Passed gates
```

---

## Phase 4: Winner Selection

### 4.1 Top-K Winner Selection
```python
# ribopo_v2/winner_selection/select_winners.py

import pandas as pd
from typing import List

class WinnerSelector:
    def __init__(self, k_winners: int = 10):
        self.k_winners = k_winners
        
    def select_winners(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Select top-K winners per backbone."""
        
        winners = []
        
        for backbone_id, group in metrics_df.groupby('backbone_id'):
            # Sort by S_total reward
            group_sorted = group.sort_values('s_total', ascending=False)
            
            # Select top-K
            backbone_winners = group_sorted.head(self.k_winners)
            
            # Ensure at least 1 winner per backbone
            if len(backbone_winners) == 0:
                # Take best available even if below threshold
                backbone_winners = group_sorted.head(1)
                print(f"Warning: Backbone {backbone_id} has weak winners")
                
            winners.append(backbone_winners)
        
        winners_df = pd.concat(winners, ignore_index=True)
        
        # Save winners with all metrics
        winners_df.to_csv('winners.csv', index=False)
        
        return winners_df
```

---

## Phase 5: Pair Construction

### 5.1 Best-vs-Random Pairing
```python
# ribopo_v2/pair_construction/build_pairs.py

import json
import random
import numpy as np
from scipy.stats import median_abs_deviation

class PairBuilder:
    def __init__(self, negatives_per_winner: int = 3):
        self.negatives_per_winner = negatives_per_winner
        
    def build_pairs(self, winners_df: pd.DataFrame, all_candidates_df: pd.DataFrame) -> List[Dict]:
        """Build preference pairs using best-vs-random strategy."""
        
        pairs = []
        
        for backbone_id in winners_df['backbone_id'].unique():
            backbone_winners = winners_df[winners_df['backbone_id'] == backbone_id]
            backbone_all = all_candidates_df[all_candidates_df['backbone_id'] == backbone_id]
            
            # Get non-winners for this backbone
            winner_ids = set(backbone_winners['sequence_id'])
            non_winners = backbone_all[~backbone_all['sequence_id'].isin(winner_ids)]
            
            # Calculate margin (0.8 * MAD)
            s_total_values = backbone_all['s_total'].values
            margin = 0.8 * median_abs_deviation(s_total_values)
            
            # Build pairs for each winner
            for _, winner in backbone_winners.iterrows():
                # Sample random negatives
                if len(non_winners) >= self.negatives_per_winner:
                    negatives = non_winners.sample(n=self.negatives_per_winner)
                else:
                    negatives = non_winners  # Use all available
                    
                for _, loser in negatives.iterrows():
                    # Check margin
                    delta_s = winner['s_total'] - loser['s_total']
                    if delta_s > margin:
                        # Apply light veto
                        if not self._veto_pair(winner, loser):
                            pairs.append({
                                'backbone_id': backbone_id,
                                'winner_seq': winner['sequence'],
                                'loser_seq': loser['sequence'],
                                'winner_metrics': winner.to_dict(),
                                'loser_metrics': loser.to_dict(),
                                'delta_s': delta_s,
                                'margin': margin
                            })
        
        # Save pairs
        with open('pairs_offline.jsonl', 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
                
        return pairs
    
    def _veto_pair(self, winner: pd.Series, loser: pd.Series) -> bool:
        """Light veto: drop if winner is >0.75σ worse on TM or INF."""
        
        # Calculate standard deviations (would need from full dataset)
        # This is simplified - in practice, calculate from all candidates
        
        tm_diff = winner['tm_score'] - loser['tm_score']
        inf_diff = winner['inf_all'] - loser['inf_all']
        
        # Veto if winner is significantly worse on key metrics
        # (simplified - would need actual std values)
        if tm_diff < -0.75 or inf_diff < -0.75:
            return True
            
        return False
```

---

## Integration Scripts

### Main Pipeline Runner
```python
# ribopo_v2/run_pipeline.py

#!/usr/bin/env python3
"""
Main pipeline runner for RiboPO v2 preference pair construction.
"""

import argparse
from pathlib import Path
import logging
import json

from candidate_generation import CandidateGenerator
from metric_computation import MetricRunner
from normalization import MetricNormalizer, RewardCalculator
from winner_selection import WinnerSelector
from pair_construction import PairBuilder

def main(args):
    logging.info("Starting RiboPO v2 pipeline")
    
    # Phase 1: Generate candidates
    if args.phase in ['all', 'generate']:
        generator = CandidateGenerator(args.model_checkpoint)
        candidates = generator.run_for_all_backbones(
            args.backbone_dir,
            n_candidates=args.n_candidates
        )
        
    # Phase 2: Compute metrics
    if args.phase in ['all', 'metrics']:
        runner = MetricRunner()
        metrics = runner.compute_all_metrics(candidates)
        
    # Phase 3: Normalize
    if args.phase in ['all', 'normalize']:
        normalizer = MetricNormalizer()
        normalizer.fit_normalization(metrics)
        normalized = normalizer.apply_normalization(metrics)
        
    # Phase 4: Calculate rewards and select winners
    if args.phase in ['all', 'winners']:
        calculator = RewardCalculator()
        rewards = calculator.calculate_all_rewards(normalized)
        
        selector = WinnerSelector(k_winners=args.k_winners)
        winners = selector.select_winners(rewards)
        
    # Phase 5: Build pairs
    if args.phase in ['all', 'pairs']:
        builder = PairBuilder(negatives_per_winner=args.negatives_per_winner)
        pairs = builder.build_pairs(winners, all_candidates)
        
    logging.info(f"Pipeline complete. Generated {len(pairs)} preference pairs")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'generate', 'metrics', 'normalize', 'winners', 'pairs'])
    parser.add_argument('--backbone_dir', type=str, default='data_clean_filtered/train/')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/grnade_base.pt')
    parser.add_argument('--n_candidates', type=int, default=200)
    parser.add_argument('--k_winners', type=int, default=10)
    parser.add_argument('--negatives_per_winner', type=int, default=3)
    
    args = parser.parse_args()
    main(args)
```

---

## Next Immediate Actions

1. **Set up gRNAde model loading**:
   ```bash
   cd ribopo_v2
   mkdir -p candidate_generation metric_computation normalization
   cp ../src/models/rna_model.py candidate_generation/
   ```

2. **Create test script for Phase 1**:
   ```bash
   python -m ribopo_v2.test_candidate_generation \
       --backbone data_clean_filtered/train/1CSL_1_B.pdb \
       --n_candidates 10
   ```

3. **Verify metric computation tools**:
   - RhoFold+ installation and paths
   - US-align binary availability
   - ViennaRNA Python bindings
   - DSSR installation

4. **Start with small subset**:
   - Use 5 backbones from training set
   - Generate 50 candidates each
   - Full metric evaluation
   - Test normalization and reward calculation

---

## Success Metrics

| Phase | Success Criteria |
|-------|------------------|
| 1 | Generate 200+ unique sequences per backbone with >0.5 3-mer diversity |
| 2 | Complete metrics in <30s per sequence with caching |
| 3 | Normalized metrics with std ∈ [0.25, 0.35] |
| 4 | ≥5 winners per backbone with S_total > 0.5 |
| 5 | ≥500 valid preference pairs with proper margins |
| 6 | DPO training convergence with improved validation metrics |

---

*This workflow will be updated as implementation progresses.*