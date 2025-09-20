# multiround/evaluator.py
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.evaluator import evaluate
from src.constants import NUM_TO_LETTER
from dpo.data import DPOPairDataset


class MultiRoundEvaluator:
    """
    Multi-round evaluation coordinator that handles:
    - Per-round evaluation with appropriate sample sizes
    - Integration with 12-metric evaluation pipeline
    - Pass@k analysis for specific rounds (3, 5)
    - Result aggregation and visualization
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Evaluation settings from config (handle both naming schemes)
        # Prefer eval_samples/final_eval_samples, fallback to n_samples_eval/n_samples_final_eval
        self.eval_samples = (getattr(cfg.multiround, 'eval_samples', None) or 
                           getattr(cfg.multiround, 'n_samples_eval', 8))
        self.final_eval_samples = (getattr(cfg.multiround, 'final_eval_samples', None) or 
                                 getattr(cfg.multiround, 'n_samples_final_eval', 64))
        self.eval_temperature = getattr(cfg.multiround, 'eval_temperature', 0.5)
        self.final_eval_temperature = getattr(cfg.multiround, 'final_eval_temperature', 0.1)
        
        # Load evaluation dataset
        self._load_eval_dataset()
        
        # Pass@k configuration
        self.passk_config = getattr(cfg.evaluation, 'pass_k', None)
        self.passk_rounds = [3, 5]  # Rounds where pass@k analysis is performed
        
    def _load_eval_dataset(self):
        """Load the test dataset for evaluation."""
        try:
            # Use the DPO dataset infrastructure to load test data
            test_dataset = DPOPairDataset(
                pairs_path=self.cfg.paths.pairs.test,
                processed_pt_path=self.cfg.paths.processed_pt,
                featurizer_cfg=vars(self.cfg.featurizer),  # Convert SimpleNamespace to dict
                split_name="test",
                device="cpu"
            )
            
            # Extract the underlying dataset (without pairs)
            self.eval_dataset = test_dataset.dataset
            print(f"✅ Loaded evaluation dataset: {len(self.eval_dataset)} structures")
            
        except Exception as e:
            print(f"⚠️ Failed to load evaluation dataset: {e}")
            self.eval_dataset = None
    
    def evaluate_round(self, model, round_num: int, output_dir: str, is_final_round: bool = False) -> Dict:
        """
        Evaluate model performance for a specific round.
        
        Args:
            model: Policy model to evaluate
            round_num: Current round number
            output_dir: Directory to save evaluation results
            is_final_round: Whether this is the final round (affects sample size and pass@k)
            
        Returns:
            dict: Evaluation results and metrics
        """
        if self.eval_dataset is None:
            return {'error': 'Evaluation dataset not available'}
        
        print(f"🔍 Evaluating round {round_num} (final: {is_final_round})")
        
        # Create organized output directories
        eval_output_dir = os.path.join(output_dir, "evaluation")
        designs_output_dir = os.path.join(output_dir, "designs")
        os.makedirs(eval_output_dir, exist_ok=True)
        os.makedirs(designs_output_dir, exist_ok=True)
        
        # Determine evaluation parameters
        n_samples = self.final_eval_samples if is_final_round else self.eval_samples
        temperature = self.final_eval_temperature if is_final_round else self.eval_temperature
        compute_passk = round_num in self.passk_rounds or is_final_round
        
        print(f"   Samples per structure: {n_samples}")
        print(f"   Temperature: {temperature}")
        print(f"   Pass@k analysis: {'Yes' if compute_passk else 'No'}")
        print(f"   Designs output: {designs_output_dir}")
        
        # Determine which metrics to compute
        metrics = self._get_metrics_for_round(round_num, is_final_round, compute_passk)
        
        try:
            # Run evaluation using the existing 12-metric pipeline
            eval_results = evaluate(
                model=model,
                dataset=self.eval_dataset,
                n_samples=n_samples,
                temperature=temperature,
                device=self.device,
                model_name=f"round_{round_num}",
                metrics=metrics,
                save_designs=True  # Save designs for analysis
            )
            
            # Save individual metrics before aggregation
            self._save_individual_metrics(eval_results, eval_output_dir, round_num, n_samples)
            
            # Aggregate results
            aggregated_results = self._aggregate_evaluation_results(eval_results)
            
            # Add round-specific metadata
            aggregated_results.update({
                'round': round_num,
                'n_samples': n_samples,
                'temperature': temperature,
                'n_structures': len(self.eval_dataset),
                'timestamp': datetime.now().isoformat(),
                'is_final_round': is_final_round,
                'output_dirs': {
                    'evaluation': eval_output_dir,
                    'designs': designs_output_dir
                }
            })
            
            # Compute pass@k analysis if requested
            if compute_passk:
                if self.passk_config:
                    print(f"🎯 Computing pass@k analysis for round {round_num}...")
                    passk_results = self._compute_passk_analysis(eval_results, n_samples)
                    aggregated_results.update(passk_results)
                else:
                    print(f"⚠️ Pass@k analysis requested for round {round_num} but no pass@k config found")
                    print(f"   Expected cfg.evaluation.pass_k but got: {self.passk_config}")
                    # Use default pass@k analysis
                    print(f"🎯 Using default pass@k configuration...")
                    passk_results = self._compute_default_passk_analysis(eval_results, n_samples)
                    aggregated_results.update(passk_results)
            
            # Save results to output directory
            self._save_evaluation_results(aggregated_results, eval_output_dir, round_num)
            
            # Create distribution plots if enough data
            if len(eval_results.get('sc_score_tm', [])) > 5:
                self._create_distribution_plots(eval_results, eval_output_dir, round_num)
            
            print(f"✅ Round {round_num} evaluation completed")
            return aggregated_results
            
        except Exception as e:
            print(f"❌ Evaluation failed for round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'round': round_num}
    
    def _get_metrics_for_round(self, round_num: int, is_final_round: bool, compute_passk: bool) -> List[str]:
        """Determine which metrics to compute for this round."""
        # Base metrics for all rounds
        metrics = ['recovery', 'perplexity']
        
        # Add 3D metrics (RhoFold-based) for comprehensive evaluation
        metrics.append('sc_score_rhofold')
        
        # Add 2D metrics
        metrics.append('sc_score_eternafold')
        
        # Add Vienna ensemble metrics for thermostability
        metrics.append('sc_score_vienna')
        
        # For final round or pass@k rounds, include all metrics
        if is_final_round or compute_passk:
            metrics.extend(['sc_score_ribonanzanet', 'sc_score_assessment'])
        
        return metrics
    
    def _aggregate_evaluation_results(self, eval_results: Dict) -> Dict:
        """Aggregate evaluation results into summary statistics."""
        aggregated = {}
        
        # Basic metrics
        if 'recovery_list' in eval_results:
            aggregated['recovery'] = float(np.mean(eval_results['recovery_list']))
            
        if 'perplexity_list' in eval_results:
            aggregated['perplexity'] = float(np.mean(eval_results['perplexity_list']))
        
        # 3D structural metrics
        if 'sc_score_rmsd' in eval_results:
            aggregated['rmsd_mean'] = float(np.mean(eval_results['sc_score_rmsd']))
            aggregated['rmsd_std'] = float(np.std(eval_results['sc_score_rmsd']))
            
        if 'sc_score_tm' in eval_results:
            aggregated['tm_mean'] = float(np.mean(eval_results['sc_score_tm']))
            aggregated['tm_std'] = float(np.std(eval_results['sc_score_tm']))
            
        if 'sc_score_gddt' in eval_results:
            aggregated['gdt_mean'] = float(np.mean(eval_results['sc_score_gddt']))
            
        # Success rates (percentage within thresholds)
        if 'rmsd_within_thresh' in eval_results:
            aggregated['rmsd_success_rate'] = float(np.mean(eval_results['rmsd_within_thresh']))
            
        if 'tm_within_thresh' in eval_results:
            aggregated['tm_success_rate'] = float(np.mean(eval_results['tm_within_thresh']))
            
        if 'gddt_within_thresh' in eval_results:
            aggregated['gdt_success_rate'] = float(np.mean(eval_results['gddt_within_thresh']))
        
        # pLDDT and other structural quality metrics
        if 'plddt_within_thresh' in eval_results:
            aggregated['plddt_success_rate'] = float(np.mean(eval_results['plddt_within_thresh']))
        
        # INF metrics (interaction network fidelity)
        if 'inf_all' in eval_results:
            aggregated['inf_all_mean'] = float(np.nanmean(eval_results['inf_all']))
        if 'inf_wc' in eval_results:
            aggregated['inf_wc_mean'] = float(np.nanmean(eval_results['inf_wc']))
        if 'inf_nwc' in eval_results:
            aggregated['inf_nwc_mean'] = float(np.nanmean(eval_results['inf_nwc']))
        if 'inf_stack' in eval_results:
            aggregated['inf_stack_mean'] = float(np.nanmean(eval_results['inf_stack']))
        
        # Clash score
        if 'clashscore' in eval_results:
            aggregated['clashscore_mean'] = float(np.nanmean(eval_results['clashscore']))
        
        # Vienna ensemble metrics (thermostability)
        if 'vienna_mfe' in eval_results:
            aggregated['mfe_mean'] = float(np.mean(eval_results['vienna_mfe']))
            aggregated['mfe_std'] = float(np.std(eval_results['vienna_mfe']))
            
        if 'vienna_ED_per_nt' in eval_results:
            aggregated['ed_per_nt_mean'] = float(np.mean(eval_results['vienna_ED_per_nt']))
            
        if 'vienna_pS0' in eval_results:
            aggregated['pS0_mean'] = float(np.nanmean(eval_results['vienna_pS0']))
            
        if 'vienna_entropy' in eval_results:
            aggregated['entropy_mean'] = float(np.mean(eval_results['vienna_entropy']))
            
        if 'vienna_diversity' in eval_results:
            aggregated['diversity_mean'] = float(np.mean(eval_results['vienna_diversity']))
            
        if 'vienna_Tm' in eval_results:
            aggregated['tm_thermo_mean'] = float(np.mean(eval_results['vienna_Tm']))
        
        # 2D structural metrics
        if 'sc_score_eternafold' in eval_results:
            aggregated['sc_eternafold_mean'] = float(np.mean(eval_results['sc_score_eternafold']))
        
        # Diversity metrics
        if 'samples_list' in eval_results:
            aggregated['diversity_3mer_mean'] = self._compute_diversity_metrics(eval_results['samples_list'])
        
        return aggregated
    
    def _compute_diversity_metrics(self, samples_list: List) -> float:
        """Compute sequence diversity metrics using 3-mer correlation."""
        try:
            from src.evaluator import get_three_mer_corr
            
            # For each structure, compute 3-mer diversity
            diversities = []
            for i, samples in enumerate(samples_list):
                if len(samples) > 1:
                    # Convert samples to strings
                    if isinstance(samples[0], np.ndarray):
                        sample_seqs = ["".join([NUM_TO_LETTER[int(n)] for n in seq]) for seq in samples]
                    else:
                        sample_seqs = samples
                    
                    # Compute pairwise diversity (1 - correlation)
                    total_corr = 0
                    count = 0
                    for j in range(len(sample_seqs)):
                        for k in range(j+1, len(sample_seqs)):
                            # Simple 3-mer overlap computation
                            corr = self._compute_3mer_overlap(sample_seqs[j], sample_seqs[k])
                            total_corr += corr
                            count += 1
                    
                    if count > 0:
                        avg_corr = total_corr / count
                        diversity = 1.0 - avg_corr  # diversity = 1 - correlation
                        diversities.append(diversity)
            
            return float(np.mean(diversities)) if diversities else 0.0
            
        except Exception as e:
            print(f"⚠️ Failed to compute diversity metrics: {e}")
            return 0.0
    
    def _compute_3mer_overlap(self, seq1: str, seq2: str) -> float:
        """Compute 3-mer overlap between two sequences."""
        if len(seq1) < 3 or len(seq2) < 3:
            return 0.0
        
        kmers1 = set(seq1[i:i+3] for i in range(len(seq1)-2))
        kmers2 = set(seq2[i:i+3] for i in range(len(seq2)-2))
        
        if len(kmers1) == 0 and len(kmers2) == 0:
            return 1.0
        
        intersection = len(kmers1.intersection(kmers2))
        union = len(kmers1.union(kmers2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_default_passk_analysis(self, eval_results: Dict, n_samples: int) -> Dict:
        """Compute pass@k analysis using default configuration when config is missing."""
        try:
            from multiround.passk import Rule, pass_at_k_from_metrics
            
            # Default configuration when config is missing
            k_values = [1, 2, 4, 8, 16, 32, 64]
            
            # Build metric dict for pass@k analysis
            metric_dict_list = []
            n_structures = len(eval_results.get('sc_score_tm', []))
            
            for i in range(n_structures):
                item_metrics = {}
                
                # TM score metrics (higher is better)
                if 'sc_score_tm' in eval_results:
                    item_metrics['tm'] = np.array(eval_results['sc_score_tm'][i])
                
                # RMSD metrics (lower is better) 
                if 'sc_score_rmsd' in eval_results:
                    item_metrics['rmsd'] = np.array(eval_results['sc_score_rmsd'][i])
                
                # MFE metrics (more negative is better)
                if 'vienna_mfe' in eval_results:
                    item_metrics['mfe'] = np.array(eval_results['vienna_mfe'][i])
                
                metric_dict_list.append(item_metrics)
            
            if not metric_dict_list:
                return {'passk_error': 'No metrics available for pass@k analysis'}
            
            # Define default success rules
            passk_results = {}
            
            # Default TM score rule: TM ≥ 0.45
            tm_rule = [Rule("tm", ">=", 0.45)]
            for k in k_values:
                if k <= n_samples:
                    passk = pass_at_k_from_metrics(
                        metric_dict_list, tm_rule, k, 
                        combine_mode="all", selection="unbiased"
                    )
                    passk_results[f"passk_tm_0.45_k{k}"] = float(passk)
            
            # Default RMSD rule: RMSD ≤ 8.0
            if 'rmsd' in metric_dict_list[0]:
                rmsd_rule = [Rule("rmsd", "<=", 8.0)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, rmsd_rule, k,
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_rmsd_8.0_k{k}"] = float(passk)
            
            # Default combined rule: TM ≥ 0.45 AND RMSD ≤ 8.0
            if 'rmsd' in metric_dict_list[0]:
                combined_rules = [Rule("tm", ">=", 0.45), Rule("rmsd", "<=", 8.0)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, combined_rules, k,
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_combined_tm0.45_rmsd8.0_k{k}"] = float(passk)
            
            print(f"✅ Default pass@k analysis completed: {len(passk_results)} metrics computed")
            return passk_results
            
        except Exception as e:
            print(f"❌ Default pass@k analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'passk_error': str(e)}
    
    def _compute_passk_analysis(self, eval_results: Dict, n_samples: int) -> Dict:
        """Compute pass@k analysis using the dedicated pass@k module."""
        try:
            # Import pass@k functionality from multiround module
            from multiround.passk import Rule, pass_at_k_from_metrics
            
            if not self.passk_config:
                return {'passk_error': 'Pass@k configuration not found'}
            
            # Extract k values and thresholds from config
            k_values = getattr(self.passk_config, 'k_values', [1, 2, 4, 8, 16, 32, 64])
            thresholds = getattr(self.passk_config, 'thresholds', None)
            
            # Build metric dict for pass@k analysis
            metric_dict_list = []
            n_structures = len(eval_results.get('sc_score_tm', []))
            
            for i in range(n_structures):
                item_metrics = {}
                
                # TM score metrics (higher is better)
                if 'sc_score_tm' in eval_results:
                    item_metrics['tm'] = np.array(eval_results['sc_score_tm'][i])
                
                # RMSD metrics (lower is better) 
                if 'sc_score_rmsd' in eval_results:
                    item_metrics['rmsd'] = np.array(eval_results['sc_score_rmsd'][i])
                
                # MFE metrics (more negative is better)
                if 'vienna_mfe' in eval_results:
                    item_metrics['mfe'] = np.array(eval_results['vienna_mfe'][i])
                
                # Ensemble defect per nucleotide (lower is better)
                if 'vienna_ED_per_nt' in eval_results:
                    item_metrics['ed_per_nt'] = np.array(eval_results['vienna_ED_per_nt'][i])
                
                metric_dict_list.append(item_metrics)
            
            if not metric_dict_list:
                return {'passk_error': 'No metrics available for pass@k analysis'}
            
            # Define success rules from thresholds
            passk_results = {}
            
            # TM score rules
            tm_thresholds = getattr(thresholds, 'tm_score', [0.45]) if thresholds else [0.45]
            for tm_thr in tm_thresholds:
                rule_set = [Rule("tm", ">=", tm_thr)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, rule_set, k, 
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_tm_{tm_thr}_k{k}"] = float(passk)
            
            # RMSD rules  
            rmsd_thresholds = getattr(thresholds, 'rmsd', [8.0]) if thresholds else [8.0]
            for rmsd_thr in rmsd_thresholds:
                rule_set = [Rule("rmsd", "<=", rmsd_thr)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, rule_set, k,
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_rmsd_{rmsd_thr}_k{k}"] = float(passk)
            
            # Combined TM + RMSD rules
            tm_thr = tm_thresholds[0] if tm_thresholds else 0.45
            rmsd_thr = rmsd_thresholds[0] if rmsd_thresholds else 8.0
            combined_rules = [Rule("tm", ">=", tm_thr), Rule("rmsd", "<=", rmsd_thr)]
            for k in k_values:
                if k <= n_samples:
                    passk = pass_at_k_from_metrics(
                        metric_dict_list, combined_rules, k,
                        combine_mode="all", selection="unbiased"
                    )
                    passk_results[f"passk_combined_tm{tm_thr}_rmsd{rmsd_thr}_k{k}"] = float(passk)
            
            # MFE rules (if available)
            if 'mfe' in metric_dict_list[0]:
                mfe_thresholds = getattr(thresholds, 'mfe', [-15.0]) if thresholds else [-15.0]
                for mfe_thr in mfe_thresholds:
                    rule_set = [Rule("mfe", "<=", mfe_thr)]
                    for k in k_values:
                        if k <= n_samples:
                            passk = pass_at_k_from_metrics(
                                metric_dict_list, rule_set, k,
                                combine_mode="all", selection="unbiased"
                            )
                            passk_results[f"passk_mfe_{mfe_thr}_k{k}"] = float(passk)
            
            print(f"✅ Pass@k analysis completed: {len(passk_results)} metrics computed")
            return passk_results
            
        except Exception as e:
            print(f"❌ Pass@k analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'passk_error': str(e)}
    
    def _save_evaluation_results(self, results: Dict, output_dir: str, round_num: int):
        """Save evaluation results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save aggregated results
        results_path = os.path.join(output_dir, f"eval_results_round_{round_num}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Evaluation results saved: {results_path}")
        
        # Create summary for quick inspection
        summary = {
            'round': round_num,
            'timestamp': results.get('timestamp'),
            'n_structures': results.get('n_structures'),
            'n_samples': results.get('n_samples'),
            'key_metrics': {
                'tm_mean': results.get('tm_mean', 'N/A'),
                'rmsd_mean': results.get('rmsd_mean', 'N/A'), 
                'mfe_mean': results.get('mfe_mean', 'N/A'),
                'recovery': results.get('recovery', 'N/A')
            }
        }
        
        # Add pass@k summary if available
        passk_keys = [k for k in results.keys() if k.startswith('passk_')]
        if passk_keys:
            summary['passk_summary'] = {k: results[k] for k in passk_keys[:5]}  # Top 5 for summary
        
        summary_path = os.path.join(output_dir, f"eval_summary_round_{round_num}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_individual_metrics(self, eval_results: Dict, output_dir: str, round_num: int, n_samples: int):
        """Save individual structure metrics for detailed analysis."""
        try:
            individual_metrics = []
            n_structures = len(eval_results.get('sc_score_tm', []))
            
            for i in range(n_structures):
                structure_metrics = {
                    'structure_idx': i,
                    'round': round_num,
                    'n_samples': n_samples
                }
                
                # Basic metrics
                if 'recovery_list' in eval_results:
                    structure_metrics['recovery'] = float(eval_results['recovery_list'][i])
                if 'perplexity_list' in eval_results:
                    structure_metrics['perplexity'] = float(eval_results['perplexity_list'][i])
                    
                # 3D structural metrics
                if 'sc_score_tm' in eval_results:
                    structure_metrics['tm_score'] = float(eval_results['sc_score_tm'][i])
                if 'sc_score_rmsd' in eval_results:
                    structure_metrics['rmsd'] = float(eval_results['sc_score_rmsd'][i])
                if 'sc_score_gddt' in eval_results:
                    structure_metrics['gdt'] = float(eval_results['sc_score_gddt'][i])
                if 'sc_score_plddt' in eval_results:
                    structure_metrics['plddt'] = float(eval_results['sc_score_plddt'][i])
                
                # INF metrics
                if 'inf_all' in eval_results:
                    structure_metrics['inf_all'] = float(eval_results['inf_all'][i])
                if 'inf_wc' in eval_results:
                    structure_metrics['inf_wc'] = float(eval_results['inf_wc'][i])
                if 'inf_nwc' in eval_results:
                    structure_metrics['inf_nwc'] = float(eval_results['inf_nwc'][i])
                if 'inf_stack' in eval_results:
                    structure_metrics['inf_stack'] = float(eval_results['inf_stack'][i])
                
                # Clash scores
                if 'clashscore_pre_relax' in eval_results:
                    structure_metrics['clash_pre'] = float(eval_results['clashscore_pre_relax'][i])
                if 'clashscore_post_relax' in eval_results:
                    structure_metrics['clash_post'] = float(eval_results['clashscore_post_relax'][i])
                
                # Vienna thermodynamics
                if 'vienna_mfe' in eval_results:
                    structure_metrics['mfe'] = float(eval_results['vienna_mfe'][i])
                if 'vienna_ED_per_nt' in eval_results:
                    structure_metrics['ed_per_nt'] = float(eval_results['vienna_ED_per_nt'][i])
                if 'vienna_pS0' in eval_results:
                    structure_metrics['pS0'] = float(eval_results['vienna_pS0'][i])
                if 'vienna_entropy' in eval_results:
                    structure_metrics['entropy'] = float(eval_results['vienna_entropy'][i])
                if 'vienna_diversity' in eval_results:
                    structure_metrics['diversity'] = float(eval_results['vienna_diversity'][i])
                if 'vienna_Tm' in eval_results:
                    structure_metrics['tm_thermo'] = float(eval_results['vienna_Tm'][i])
                
                # 2D structural metrics
                if 'sc_score_eternafold' in eval_results:
                    structure_metrics['sc_eternafold'] = float(eval_results['sc_score_eternafold'][i])
                
                individual_metrics.append(structure_metrics)
            
            # Save individual metrics
            individual_path = os.path.join(output_dir, f"individual_metrics_round_{round_num}.json")
            with open(individual_path, 'w') as f:
                json.dump(individual_metrics, f, indent=2)
            
            print(f"💾 Individual metrics saved: {individual_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save individual metrics: {e}")
    
    def _create_distribution_plots(self, eval_results: Dict, output_dir: str, round_num: int):
        """Create distribution plots for key metrics."""
        try:
            import matplotlib.pyplot as plt
            
            # Key metrics to plot
            metrics_to_plot = {
                'pLDDT': eval_results.get('sc_score_plddt', []),
                'RMSD (Å)': eval_results.get('sc_score_rmsd', []),
                'MFE (kcal/mol)': eval_results.get('vienna_mfe', []),
                'TM Score': eval_results.get('sc_score_tm', [])
            }
            
            # Filter out empty metrics
            metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if v}
            
            if not metrics_to_plot:
                print("⚠️ No metrics available for distribution plots")
                return
            
            # Create subplots
            n_metrics = len(metrics_to_plot)
            n_cols = min(2, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
                ax = axes[i]
                
                # Convert to numpy array and filter out NaNs
                values = np.array(values)
                values = values[~np.isnan(values)]
                
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{metric_name} Distribution (Round {round_num})')
                    ax.set_xlabel(metric_name)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                              label=f'Mean: {mean_val:.3f}')
                    ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\\nStd: {std_val:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data available', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{metric_name} Distribution (Round {round_num})')
            
            # Hide extra subplots
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"distribution_plots_round_{round_num}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Distribution plots saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create distribution plots: {e}")
            import traceback
            traceback.print_exc()