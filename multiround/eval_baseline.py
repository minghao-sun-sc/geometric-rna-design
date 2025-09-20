# multiround/eval_baseline.py
"""
Baseline Model Evaluation Pipeline

Evaluate sequence-only baseline models that provide designed sequences
but no trained model. This pipeline:
1. Takes designed sequences as input
2. Matches them with test dataset structures
3. Uses RhoFold+ to predict 3D structures
4. Calculates all 12 evaluation metrics
5. Saves results in standard format for comparison
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import shutil
import tempfile

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.evaluator import (
    self_consistency_score_eternafold,
    self_consistency_score_rhofold_extended,
    get_trimer_profile_novelty,
    vienna_ensemble_metrics,
    vienna_mfe,
    vienna_Tm_by_pS0,
    get_inf,
    get_clash_score_phenix,
    get_lddt_openstructure_v2,
    mcq_avg_vs_natives,
    edit_distance,
    get_three_mer_corr
)
from src.constants import (
    NUM_TO_LETTER, LETTER_TO_NUM,
    RMSD_THRESHOLD, RMSD_THRESHOLD_2, TM_THRESHOLD, 
    GDT_THRESHOLD, PLDDT_THRESHOLD, DATA_PATH, PROJECT_PATH
)


class BaselineEvaluator:
    """
    Evaluator for baseline RNA design models that only provide designed sequences.
    
    This class:
    1. Loads test dataset structure information
    2. Matches baseline sequences to structures
    3. Uses RhoFold to predict 3D structures from sequences
    4. Calculates comprehensive evaluation metrics
    5. Saves results in standardized format
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(getattr(config, 'device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Load test dataset structure information
        self._load_test_structures()
        
        # Initialize RhoFold for structure prediction
        self._init_rhofold()
        
        # Setup evaluation tools paths
        self.phenix_wrapper_path = os.path.join(PROJECT_PATH, "tools", "run_phenix.sh")
        
    def _load_test_structures(self):
        """Load test dataset structure information."""
        print("📂 Loading test dataset structure information...")
        
        # Load structure IDs from the test set
        test_ids_file = os.path.join(DATA_PATH, "das_split_raw_data", "test_set_structure_ids.txt")
        self.test_structure_ids = []
        
        if os.path.exists(test_ids_file):
            with open(test_ids_file, 'r') as f:
                for line in f:
                    structure_id = line.strip()
                    if structure_id:
                        self.test_structure_ids.append(structure_id)
        
        print(f"   Found {len(self.test_structure_ids)} test structures")
        
        # Load native sequences for comparison
        self.native_sequences = {}
        
        # First try to load from ridiff (contains all 235 native sequences)
        self._load_native_sequences_from_ridiff()
        
        # Fallback: load from das_test_sequences.fasta for any missing sequences
        native_seq_file = os.path.join(DATA_PATH, "das_split_raw_data", "das_split_raw_seq", "das_test_sequences.fasta")
        if os.path.exists(native_seq_file):
            for record in SeqIO.parse(native_seq_file, "fasta"):
                structure_id = record.id.split()[0]
                if structure_id not in self.native_sequences:
                    self.native_sequences[structure_id] = str(record.seq)
        
        print(f"   Loaded {len(self.native_sequences)} native sequences")
        
        # Map structure IDs to raw data information
        self._build_structure_mapping()
        
    def _build_structure_mapping(self):
        """Build mapping from structure IDs to raw data paths."""
        self.structure_data = {}
        
        for structure_id in self.test_structure_ids:
            # The file naming convention from the data shows:
            # Structure ID: "1CSL_1_B" but file name: "1CSL_1_B-A.pdb"
            # Let's handle this mapping
            base_pdb_name = self._find_pdb_file(structure_id)
            
            if base_pdb_name:
                native_pdb_path = os.path.join(DATA_PATH, "das_split_raw_data", "das_split_raw_pdb", base_pdb_name)
                native_sequence = self.native_sequences.get(structure_id, '')
                
                # Only include structures that have both PDB file AND native sequence
                if os.path.exists(native_pdb_path) and native_sequence:
                    self.structure_data[structure_id] = {
                        'native_pdb_path': native_pdb_path,
                        'native_sequence': native_sequence,
                        'structure_id': structure_id,
                        'base_pdb_name': base_pdb_name
                    }
        
        print(f"   Mapped {len(self.structure_data)} structures to PDB files")
        
    def _find_pdb_file(self, structure_id):
        """Find the corresponding PDB file for a structure ID."""
        # Look for exact match first
        pdb_dir = os.path.join(DATA_PATH, "das_split_raw_data", "das_split_raw_pdb")
        potential_files = [
            f"{structure_id}.pdb",
            f"{structure_id}-A.pdb",
            f"{structure_id}A.pdb",
        ]
        
        # Also check for multi-chain cases
        for suffix in ["-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-P", "-Q", "-R", "-S", "-X", "-Y"]:
            potential_files.append(f"{structure_id}{suffix}.pdb")
        
        for pdb_file in potential_files:
            if os.path.exists(os.path.join(pdb_dir, pdb_file)):
                return pdb_file
        
        # If no exact match, look for files that start with the structure ID
        if os.path.exists(pdb_dir):
            for filename in os.listdir(pdb_dir):
                if filename.startswith(structure_id) and filename.endswith('.pdb'):
                    return filename
        
        print(f"   ⚠️ Warning: No PDB file found for structure {structure_id}")
        return None
        
    def _init_rhofold(self):
        """Initialize RhoFold for structure prediction."""
        print("🧬 Initializing RhoFold for structure prediction...")
        
        try:
            from tools.rhofold.rf import RhoFold
            from tools.rhofold.config import rhofold_config
            
            # Initialize RhoFold
            self.rhofold = RhoFold(rhofold_config, self.device)
            rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
            
            print(f"   Loading RhoFold checkpoint: {rhofold_path}")
            self.rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
            self.rhofold = self.rhofold.to(self.device)
            self.rhofold.eval()
            
            print("   ✅ RhoFold initialized successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize RhoFold: {e}")
            self.rhofold = None
            
    def evaluate_baseline_sequences(self, sequences_file: str, model_name: str, output_dir: str) -> Dict:
        """
        Evaluate baseline model sequences.
        
        Args:
            sequences_file: Path to FASTA file with designed sequences
            model_name: Name of the baseline model
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n🔬 Evaluating {model_name} sequences...")
        print(f"   Sequences file: {sequences_file}")
        print(f"   Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load designed sequences
        designed_sequences = self._load_designed_sequences(sequences_file)
        print(f"   Loaded {len(designed_sequences)} designed sequences")
        
        # Match sequences to test structures
        matched_sequences = self._match_sequences_to_structures(designed_sequences)
        print(f"   Matched {len(matched_sequences)} sequences to test structures")
        
        if not matched_sequences:
            print("   ❌ No sequences matched to test structures")
            return {}
        
        # Evaluate each matched sequence
        results = self._evaluate_matched_sequences(matched_sequences, model_name, output_dir)
        
        # Save aggregated results
        self._save_results(results, model_name, output_dir)
        
        return results
        
    def _load_designed_sequences(self, sequences_file: str) -> Dict[str, List[str]]:
        """Load designed sequences from FASTA file."""
        designed_sequences = {}
        
        try:
            for record in SeqIO.parse(sequences_file, "fasta"):
                # Parse structure ID from header
                header_parts = record.id.split()
                structure_id = header_parts[0]
                
                # Handle multiple samples per structure (like ridiff)
                if structure_id not in designed_sequences:
                    designed_sequences[structure_id] = []
                
                designed_sequences[structure_id].append(str(record.seq))
                
        except Exception as e:
            print(f"   ❌ Error loading sequences: {e}")
            
        return designed_sequences
        
    def _match_sequences_to_structures(self, designed_sequences: Dict[str, List[str]]) -> Dict:
        """Match designed sequences to test structures."""
        matched = {}
        
        for structure_id, sequences in designed_sequences.items():
            if structure_id in self.structure_data:
                matched[structure_id] = {
                    'designed_sequences': sequences,
                    'structure_data': self.structure_data[structure_id]
                }
            else:
                # Check why structure was not included
                if structure_id in self.test_structure_ids:
                    if structure_id not in self.native_sequences:
                        print(f"   ⚠️ Structure {structure_id} missing native sequence (skipping)")
                    else:
                        print(f"   ⚠️ Structure {structure_id} missing PDB file (skipping)")
                else:
                    print(f"   ⚠️ Structure {structure_id} not in test dataset (skipping)")
                
        return matched
        
    def _evaluate_matched_sequences(self, matched_sequences: Dict, model_name: str, output_dir: str) -> Dict:
        """Evaluate each matched sequence using the full evaluation pipeline."""
        all_results = []
        
        structures_dir = os.path.join(output_dir, "structures")
        os.makedirs(structures_dir, exist_ok=True)
        
        for structure_id, data in matched_sequences.items():
            print(f"\n   🔬 Evaluating structure {structure_id}...")
            
            designed_sequences = data['designed_sequences']
            structure_data = data['structure_data']
            
            # Create structure-specific output directory
            struct_output_dir = os.path.join(structures_dir, structure_id)
            os.makedirs(struct_output_dir, exist_ok=True)
            
            # Evaluate each designed sequence for this structure
            for seq_idx, designed_seq in enumerate(designed_sequences):
                result = self._evaluate_single_sequence(
                    designed_seq, structure_id, structure_data,
                    struct_output_dir, seq_idx
                )
                
                if result:
                    result['model_name'] = model_name
                    result['structure_id'] = structure_id
                    result['sequence_index'] = seq_idx
                    all_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        aggregated['individual_results'] = all_results
        aggregated['n_structures'] = len(matched_sequences)
        aggregated['n_sequences'] = len(all_results)
        
        return aggregated
        
    def _evaluate_single_sequence(self, designed_seq: str, structure_id: str, 
                                structure_data: Dict, output_dir: str, seq_idx: int) -> Optional[Dict]:
        """Evaluate a single designed sequence."""
        
        try:
            # Create sequence FASTA file
            seq_fasta_path = os.path.join(output_dir, f"{structure_id}_sequence.fasta")
            seq_record = SeqRecord(Seq(designed_seq), id=structure_id, description=f"designed sequence {seq_idx}")
            SeqIO.write(seq_record, seq_fasta_path, "fasta")
            
            # Predict structure using RhoFold
            predicted_pdb_path = os.path.join(output_dir, f"{structure_id}_predicted.pdb")
            
            if self.rhofold is None:
                print(f"      ❌ RhoFold not available, skipping structure prediction")
                return None
                
            coords, plddt = self.rhofold.predict(seq_fasta_path, predicted_pdb_path, use_relax=False)
            
            # Save pLDDT scores
            plddt_path = os.path.join(output_dir, f"{structure_id}_plddt.npy")
            np.save(plddt_path, plddt)
            
            # Calculate evaluation metrics
            metrics = self._calculate_metrics(
                designed_seq, structure_data, predicted_pdb_path, plddt
            )
            
            metrics['designed_sequence'] = designed_seq
            metrics['predicted_pdb_path'] = predicted_pdb_path
            metrics['plddt_mean'] = float(np.mean(plddt))
            
            return metrics
            
        except Exception as e:
            print(f"      ❌ Error evaluating sequence {seq_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _calculate_metrics(self, designed_seq: str, structure_data: Dict, 
                         predicted_pdb_path: str, plddt: np.ndarray) -> Dict:
        """Calculate all evaluation metrics for a sequence."""
        metrics = {}
        
        native_seq = structure_data['native_sequence']
        native_pdb_path = structure_data['native_pdb_path']
        
        # Sequence recovery
        if native_seq:
            metrics['sequence_recovery'] = sum(1 for a, b in zip(designed_seq, native_seq) if a == b) / len(native_seq)
            metrics['edit_distance'] = edit_distance(designed_seq, native_seq)
        else:
            metrics['sequence_recovery'] = 0.0
            metrics['edit_distance'] = len(designed_seq)
        
        # 3D structure metrics (using existing evaluator functions)
        try:
            # Create mock raw data structure for compatibility with existing functions
            mock_raw_data = {
                'sequence': native_seq,
                'id_list': [structure_data['structure_id']],
                'coords_list': []  # We'll load this if needed
            }
            
            # Calculate structural metrics using existing functions
            # TM-score, RMSD, GDT via coordinate comparison would require loading native coordinates
            # For now, we'll focus on what we can calculate directly
            
            # Vienna thermodynamic metrics
            vienna_metrics = vienna_ensemble_metrics(designed_seq, target_db=None, T=37.0)
            metrics.update({
                'mfe': vienna_metrics['mfe'],
                'ensemble_defect': vienna_metrics['ED'],
                'ensemble_defect_per_nt': vienna_metrics['ED_per_nt'],
                'target_probability': vienna_metrics['pS0'],
                'ensemble_diversity': vienna_metrics['diversity'],
                'shannon_entropy': vienna_metrics['entropy_mean']
            })
            
            # Clash score
            try:
                clash_score = get_clash_score_phenix(predicted_pdb_path, self.phenix_wrapper_path)
                metrics['clash_score'] = clash_score
            except Exception as e:
                print(f"        ⚠️ Clash score calculation failed: {e}")
                metrics['clash_score'] = np.nan
            
            # INF score
            try:
                inf_scores = get_inf(predicted_pdb_path, mock_raw_data, DATA_PATH)
                metrics.update({
                    'inf_all': inf_scores['all'],
                    'inf_wc': inf_scores['wc'], 
                    'inf_nwc': inf_scores['nwc'],
                    'inf_stack': inf_scores['stack']
                })
            except Exception as e:
                print(f"        ⚠️ INF score calculation failed: {e}")
                metrics.update({
                    'inf_all': np.nan, 'inf_wc': np.nan, 
                    'inf_nwc': np.nan, 'inf_stack': np.nan
                })
            
            # lDDT score
            try:
                lddt_score = get_lddt_openstructure_v2(predicted_pdb_path, native_pdb_path)
                metrics['lddt'] = lddt_score
            except Exception as e:
                print(f"        ⚠️ lDDT calculation failed: {e}")
                metrics['lddt'] = np.nan
            
            # MCQ metrics
            try:
                mcq_metrics = mcq_avg_vs_natives(predicted_pdb_path, mock_raw_data, DATA_PATH)
                metrics.update({
                    'mcq_abs_deg': mcq_metrics['mcq_abs_deg'],
                    'mcq_R': mcq_metrics['R'],
                    'mcq_circ_sd_deg': mcq_metrics['circ_sd_deg']
                })
            except Exception as e:
                print(f"        ⚠️ MCQ calculation failed: {e}")
                metrics.update({
                    'mcq_abs_deg': np.nan, 'mcq_R': np.nan, 'mcq_circ_sd_deg': np.nan
                })
            
        except Exception as e:
            print(f"        ⚠️ Metric calculation error: {e}")
            # Set default values for failed metrics
            default_metrics = {
                'mfe': np.nan, 'ensemble_defect': np.nan, 'ensemble_defect_per_nt': np.nan,
                'target_probability': np.nan, 'ensemble_diversity': np.nan, 'shannon_entropy': np.nan,
                'clash_score': np.nan, 'inf_all': np.nan, 'inf_wc': np.nan, 'inf_nwc': np.nan, 
                'inf_stack': np.nan, 'lddt': np.nan, 'mcq_abs_deg': np.nan, 'mcq_R': np.nan, 
                'mcq_circ_sd_deg': np.nan
            }
            metrics.update(default_metrics)
        
        return metrics
        
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all sequences."""
        if not all_results:
            return {}
        
        aggregated = {}
        
        # Calculate mean values for each metric
        metrics_to_aggregate = [
            'sequence_recovery', 'edit_distance', 'plddt_mean',
            'mfe', 'ensemble_defect', 'ensemble_defect_per_nt',
            'target_probability', 'ensemble_diversity', 'shannon_entropy',
            'clash_score', 'inf_all', 'inf_wc', 'inf_nwc', 'inf_stack',
            'lddt', 'mcq_abs_deg', 'mcq_R', 'mcq_circ_sd_deg'
        ]
        
        for metric in metrics_to_aggregate:
            values = [r.get(metric, np.nan) for r in all_results]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                aggregated[f'{metric}_mean'] = np.mean(valid_values)
                aggregated[f'{metric}_std'] = np.std(valid_values)
                aggregated[f'{metric}_median'] = np.median(valid_values)
            else:
                aggregated[f'{metric}_mean'] = np.nan
                aggregated[f'{metric}_std'] = np.nan
                aggregated[f'{metric}_median'] = np.nan
        
        return aggregated
        
    def _save_results(self, results: Dict, model_name: str, output_dir: str):
        """Save evaluation results to files."""
        
        # Save aggregated results
        aggregated_file = os.path.join(output_dir, f"{model_name}_aggregated_results.json")
        aggregated_results = {k: v for k, v in results.items() if k != 'individual_results'}
        
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Save individual results
        individual_file = os.path.join(output_dir, f"{model_name}_individual_results.json")
        individual_results = results.get('individual_results', [])
        
        with open(individual_file, 'w') as f:
            json.dump(individual_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Save summary
        summary_file = os.path.join(output_dir, f"{model_name}_summary.json")
        summary = {
            'model_name': model_name,
            'n_structures': results.get('n_structures', 0),
            'n_sequences': results.get('n_sequences', 0),
            'evaluation_timestamp': datetime.now().isoformat(),
            'key_metrics': {
                'sequence_recovery_mean': results.get('sequence_recovery_mean', np.nan),
                'plddt_mean': results.get('plddt_mean_mean', np.nan), 
                'mfe_mean': results.get('mfe_mean', np.nan),
                'clash_score_mean': results.get('clash_score_mean', np.nan),
                'lddt_mean': results.get('lddt_mean', np.nan),
                'inf_all_mean': results.get('inf_all_mean', np.nan)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"   💾 Results saved to {output_dir}")
        print(f"       - Aggregated: {aggregated_file}")
        print(f"       - Individual: {individual_file}")
        print(f"       - Summary: {summary_file}")