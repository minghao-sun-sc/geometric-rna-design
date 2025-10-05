"""
RiboPO v2 Real Metrics Evaluation Module
Integrates actual structure prediction and metric calculations for RNA sequences.
"""

import os
import sys
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import tempfile
import subprocess
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions from existing evaluator
from src.evaluator import (
    vienna_ensemble_metrics,
    get_inf,
    get_tmscore,
    get_gddt,
    self_consistency_score_rhofold_extended,
    get_usalign_tmscore
)
from src.constants import DATA_PATH, PROJECT_PATH

# Import data utilities
from src.data.data_utils import pdb_to_tensor, get_c4p_coords

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Container for comprehensive RNA metrics"""
    sequence: str
    backbone_id: str
    
    # Structure metrics
    tm_score: float = np.nan
    rmsd: float = np.nan
    gdt: float = np.nan
    lddt: float = np.nan
    plddt: float = np.nan
    
    # Interaction metrics
    inf_all: float = np.nan
    inf_wc: float = np.nan
    inf_nwc: float = np.nan
    inf_stack: float = np.nan
    
    # Thermodynamic metrics
    mfe: float = np.nan
    ensemble_defect: float = np.nan
    ed_per_nt: float = np.nan
    p_target: float = np.nan
    shannon_entropy: float = np.nan
    
    # Additional metadata
    predicted_pdb_path: Optional[str] = None
    cache_key: Optional[str] = None


class RealMetricsEvaluator:
    """
    Evaluates RNA sequences using actual structure prediction and metrics.
    Integrates RhoFold+, ViennaRNA, and RNA_normalizer for comprehensive evaluation.
    """
    
    def __init__(self, cache_dir: str = "ribopo_v2/cache/metrics"):
        """
        Initialize metrics evaluator with caching support.
        
        Args:
            cache_dir: Directory for caching expensive metric calculations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RhoFold model (lazy loading)
        self.rhofold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized RealMetricsEvaluator with cache at {cache_dir}")
        logger.info(f"Using device: {self.device}")
        
    def _get_cache_key(self, sequence: str, backbone_id: str) -> str:
        """Generate unique cache key for sequence+backbone pair"""
        content = f"{sequence}_{backbone_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[MetricsResult]:
        """Load metrics from cache if available"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, result: MetricsResult):
        """Save metrics to cache"""
        if result.cache_key:
            cache_file = self.cache_dir / f"{result.cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {result.cache_key}: {e}")
    
    def _initialize_rhofold(self):
        """Lazy initialization of RhoFold model"""
        if self.rhofold is None:
            try:
                from tools.rhofold.rf import RhoFold
                from tools.rhofold.config import rhofold_config
                
                self.rhofold = RhoFold(rhofold_config, self.device)
                rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
                logger.info(f"Loading RhoFold from {rhofold_path}")
                self.rhofold.load_state_dict(
                    torch.load(rhofold_path, map_location=self.device)['model']
                )
                self.rhofold = self.rhofold.to(self.device)
                self.rhofold.eval()
                logger.info("RhoFold initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RhoFold: {e}")
                raise
    
    def predict_structure(self, sequence: str, output_dir: str) -> Tuple[str, np.ndarray]:
        """
        Predict 3D structure using RhoFold.
        
        Args:
            sequence: RNA sequence to fold
            output_dir: Directory to save predicted PDB
            
        Returns:
            Tuple of (pdb_path, plddt_scores)
        """
        self._initialize_rhofold()
        
        # Create temporary FASTA file
        os.makedirs(output_dir, exist_ok=True)
        fasta_path = os.path.join(output_dir, "sequence.fasta")
        pdb_path = os.path.join(output_dir, "predicted.pdb")
        
        with open(fasta_path, 'w') as f:
            f.write(f">seq\n{sequence}\n")
        
        # Run RhoFold prediction
        try:
            _, plddt = self.rhofold.predict(fasta_path, pdb_path, use_relax=False)
            logger.debug(f"Predicted structure saved to {pdb_path}")
            return pdb_path, plddt
        except Exception as e:
            logger.error(f"RhoFold prediction failed: {e}")
            raise
        finally:
            # Clean up temporary FASTA
            if os.path.exists(fasta_path):
                os.remove(fasta_path)
    
    def calculate_structure_metrics(self, 
                                  predicted_pdb: str,
                                  native_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate structural similarity metrics vs native structure(s).
        
        Args:
            predicted_pdb: Path to predicted PDB structure
            native_data: Native structure data with coords_list
            
        Returns:
            Dictionary with TM-score, RMSD, GDT, lDDT
        """
        metrics = {}
        
        try:
            # Load predicted coordinates
            _, pred_coords, _, _ = pdb_to_tensor(
                predicted_pdb,
                return_sec_struct=False,
                return_sasa=False,
                keep_insertions=False
            )
            pred_c4p = get_c4p_coords(pred_coords)
            pred_c4p = pred_c4p - pred_c4p.mean(dim=0)  # Center
            
            # Calculate metrics vs each native structure
            tm_scores = []
            rmsds = []
            gdts = []
            
            for native_coords in native_data.get('coords_list', []):
                native_c4p = get_c4p_coords(native_coords)
                native_c4p = native_c4p - native_c4p.mean(dim=0)
                
                # Align structures
                from MDAnalysis.analysis.align import rotation_matrix
                from MDAnalysis.analysis.rms import rmsd as get_rmsd
                
                R_hat = rotation_matrix(native_c4p, pred_c4p)[0]
                native_aligned = native_c4p @ R_hat.T
                
                # Calculate metrics
                tm_scores.append(get_tmscore(pred_c4p, native_aligned).item())
                rmsds.append(get_rmsd(pred_c4p, native_aligned, superposition=True, center=True))
                gdts.append(get_gddt(pred_c4p, native_aligned).item())
            
            metrics['tm_score'] = np.mean(tm_scores) if tm_scores else np.nan
            metrics['rmsd'] = np.mean(rmsds) if rmsds else np.nan
            metrics['gdt'] = np.mean(gdts) if gdts else np.nan
            
            # Calculate lDDT if native PDB available
            if 'id_list' in native_data and native_data['id_list']:
                native_pdb = os.path.join(DATA_PATH, "raw", f"{native_data['id_list'][0]}.pdb")
                if os.path.exists(native_pdb):
                    try:
                        from src.evaluator import get_lddt_openstructure_v2
                        metrics['lddt'] = get_lddt_openstructure_v2(predicted_pdb, native_pdb)
                    except Exception as e:
                        logger.warning(f"lDDT calculation failed: {e}")
                        metrics['lddt'] = np.nan
                        
        except Exception as e:
            logger.error(f"Structure metrics calculation failed: {e}")
            metrics = {'tm_score': np.nan, 'rmsd': np.nan, 'gdt': np.nan, 'lddt': np.nan}
            
        return metrics
    
    def calculate_inf_metrics(self,
                            predicted_pdb: str,
                            native_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Interaction Network Fidelity metrics.
        
        Args:
            predicted_pdb: Path to predicted PDB structure
            native_data: Native structure data with id_list
            
        Returns:
            Dictionary with INF_ALL, INF_WC, INF_NWC, INF_STACK
        """
        try:
            inf_results = get_inf(predicted_pdb, native_data, DATA_PATH)
            return {
                'inf_all': inf_results['all'],
                'inf_wc': inf_results['wc'],
                'inf_nwc': inf_results['nwc'],
                'inf_stack': inf_results['stack']
            }
        except Exception as e:
            logger.warning(f"INF calculation failed: {e}")
            return {
                'inf_all': np.nan,
                'inf_wc': np.nan,
                'inf_nwc': np.nan,
                'inf_stack': np.nan
            }
    
    def calculate_thermodynamic_metrics(self,
                                      sequence: str,
                                      target_structure: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate thermodynamic ensemble metrics using ViennaRNA.
        
        Args:
            sequence: RNA sequence
            target_structure: Target secondary structure in dot-bracket notation
            
        Returns:
            Dictionary with MFE, ED, ED/nt, P(target), Shannon entropy
        """
        try:
            results = vienna_ensemble_metrics(
                sequence,
                target_db=target_structure,
                T=37.0,
                return_positional_entropy=False
            )
            
            return {
                'mfe': results['mfe'],
                'ensemble_defect': results['ED'],
                'ed_per_nt': results['ED_per_nt'],
                'p_target': results['pS0'],
                'shannon_entropy': results['entropy_mean']
            }
        except Exception as e:
            logger.warning(f"Vienna metrics calculation failed: {e}")
            return {
                'mfe': np.nan,
                'ensemble_defect': np.nan,
                'ed_per_nt': np.nan,
                'p_target': np.nan,
                'shannon_entropy': np.nan
            }
    
    def evaluate_sequence(self,
                         sequence: str,
                         backbone_data: Dict[str, Any],
                         backbone_id: str,
                         use_cache: bool = True,
                         cleanup: bool = True) -> MetricsResult:
        """
        Evaluate a single RNA sequence with comprehensive metrics.
        
        Args:
            sequence: RNA sequence to evaluate
            backbone_data: Native backbone structure data
            backbone_id: Unique identifier for the backbone
            use_cache: Whether to use cached results
            cleanup: Whether to cleanup temporary files
            
        Returns:
            MetricsResult with all calculated metrics
        """
        # Check cache first
        cache_key = self._get_cache_key(sequence, backbone_id)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Using cached metrics for {cache_key}")
                return cached
        
        # Initialize result
        result = MetricsResult(
            sequence=sequence,
            backbone_id=backbone_id,
            cache_key=cache_key
        )
        
        # Create temporary directory for predictions
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. Predict 3D structure with RhoFold
                logger.debug(f"Predicting structure for sequence of length {len(sequence)}")
                pdb_path, plddt = self.predict_structure(sequence, temp_dir)
                result.predicted_pdb_path = pdb_path
                result.plddt = np.mean(plddt) if len(plddt) > 0 else np.nan
                
                # 2. Calculate structure metrics (TM, RMSD, GDT, lDDT)
                struct_metrics = self.calculate_structure_metrics(pdb_path, backbone_data)
                for key, val in struct_metrics.items():
                    setattr(result, key, val)
                
                # 3. Calculate INF metrics
                inf_metrics = self.calculate_inf_metrics(pdb_path, backbone_data)
                for key, val in inf_metrics.items():
                    setattr(result, key, val)
                
                # 4. Calculate thermodynamic metrics
                # Use first secondary structure as target if available
                target_ss = None
                if 'sec_struct_list' in backbone_data and backbone_data['sec_struct_list']:
                    target_ss = backbone_data['sec_struct_list'][0]
                
                thermo_metrics = self.calculate_thermodynamic_metrics(sequence, target_ss)
                for key, val in thermo_metrics.items():
                    setattr(result, key, val)
                
                logger.debug(f"Completed evaluation for {backbone_id}: TM={result.tm_score:.3f}, "
                           f"INF={result.inf_all:.3f}, ED/nt={result.ed_per_nt:.3f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {backbone_id}: {e}")
                # Result will have NaN values for failed metrics
        
        # Save to cache
        if use_cache:
            self._save_to_cache(result)
        
        return result
    
    def evaluate_batch(self,
                      sequences: List[str],
                      backbone_data: Dict[str, Any],
                      backbone_id: str,
                      use_cache: bool = True,
                      parallel: bool = False) -> List[MetricsResult]:
        """
        Evaluate multiple sequences for the same backbone.
        
        Args:
            sequences: List of RNA sequences
            backbone_data: Native backbone structure data
            backbone_id: Unique identifier for the backbone
            use_cache: Whether to use cached results
            parallel: Whether to evaluate in parallel (future enhancement)
            
        Returns:
            List of MetricsResult objects
        """
        results = []
        
        for i, seq in enumerate(sequences):
            logger.info(f"Evaluating sequence {i+1}/{len(sequences)} for {backbone_id}")
            result = self.evaluate_sequence(seq, backbone_data, backbone_id, use_cache)
            results.append(result)
        
        return results


def create_test_metrics_evaluator():
    """Factory function to create metrics evaluator for testing"""
    return RealMetricsEvaluator(cache_dir="ribopo_v2/cache/metrics_test")


if __name__ == "__main__":
    # Simple test of the metrics evaluator
    evaluator = create_test_metrics_evaluator()
    
    # Test sequence
    test_seq = "GGAAAGCUGAAGCUGGCCCUGAUGGAGCUGAGAACUGGGGCUCC"
    test_backbone_data = {
        'id_list': ['1Y26_1_X'],
        'coords_list': [torch.zeros((len(test_seq), 27, 3))],  # Mock coords
        'sec_struct_list': ['.' * len(test_seq)]
    }
    
    logger.info("Testing RealMetricsEvaluator...")
    result = evaluator.evaluate_sequence(
        test_seq,
        test_backbone_data,
        "test_backbone",
        use_cache=False
    )
    
    print(f"\nTest Results:")
    print(f"  Sequence: {result.sequence[:20]}...")
    print(f"  TM-score: {result.tm_score:.3f}")
    print(f"  RMSD: {result.rmsd:.3f}")
    print(f"  INF_ALL: {result.inf_all:.3f}")
    print(f"  MFE: {result.mfe:.3f}")
    print(f"  ED/nt: {result.ed_per_nt:.3f}")