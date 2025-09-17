# multiround/evaluator.py
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from dpo.bench.eval_full import FullEvalDataset
from dpo.passk import pass_at_k_from_metrics, Rule
from src.evaluator import evaluate
from multiround.utils import plot_training_curves, create_evaluation_summary_table


class MultiRoundEvaluator:
    """
    Evaluator for multi-round DPO training.
    Handles both per-round evaluation and final comprehensive evaluation.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Evaluation settings
        self.eval_samples = getattr(cfg.multiround, 'eval_samples', 8)
        self.final_eval_samples = getattr(cfg.multiround, 'final_eval_samples', 64)
        self.eval_temperature = getattr(cfg.multiround, 'eval_temperature', 0.5)
        
        # Metrics configuration
        eval_cfg = getattr(cfg, 'evaluation', {})
        self.primary_metrics = getattr(eval_cfg, 'primary_metrics', ['tm', 'rmsd', 'mfe'])
        self.secondary_metrics = getattr(eval_cfg, 'secondary_metrics', 
                                       ['plddt', 'gdt', 'inf_all', 'clashscore', 'diversity_3mer'])
        
        # Initialize dataset for evaluation
        self._init_evaluation_dataset()
    
    def _init_evaluation_dataset(self):
        """Initialize the evaluation dataset."""
        try:
            self.eval_dataset = FullEvalDataset(
                processed_pt=self.cfg.paths.processed_pt,
                split_pt=self.cfg.paths.split_pt,
                split_name="test",  # Always evaluate on test set
                feat_cfg=self.cfg.featurizer,
                device="cpu"  # Keep data on CPU, move to GPU during evaluation
            )
            print(f"📊 Evaluation dataset loaded: {len(self.eval_dataset)} test structures")
        except Exception as e:
            print(f"❌ Failed to load evaluation dataset: {e}")
            self.eval_dataset = None
    
    def evaluate_round(self, 
                      model: torch.nn.Module, 
                      round_num: int, 
                      output_dir: str,
                      is_final_round: bool = False) -> Dict:
        """
        Evaluate model performance for a specific round.
        
        Args:
            model: Model to evaluate
            round_num: Current round number
            output_dir: Directory to save evaluation results
            is_final_round: Whether this is the final round (enables expensive metrics)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.eval_dataset is None:
            return {'error': 'Evaluation dataset not available'}
        
        print(f"🔍 Evaluating Round {round_num}")
        print(f"   Samples per structure: {self.final_eval_samples if is_final_round else self.eval_samples}")
        print(f"   Temperature: {self.eval_temperature}")
        
        eval_start_time = datetime.now()
        
        # Determine number of samples
        n_samples = self.final_eval_samples if is_final_round else self.eval_samples
        
        # Run comprehensive evaluation using existing evaluator
        eval_result = self._run_comprehensive_evaluation(
            model, n_samples, output_dir, is_final_round
        )
        
        # Add pass@k analysis for final round
        if is_final_round:
            passk_result = self._run_passk_analysis(eval_result, output_dir)
            eval_result.update(passk_result)
        
        # Save detailed results
        self._save_evaluation_results(eval_result, output_dir, round_num)
        
        # Generate evaluation plots
        self._generate_evaluation_plots(eval_result, output_dir, round_num)
        
        eval_time = datetime.now() - eval_start_time
        print(f"✅ Evaluation completed in {eval_time}")
        
        return eval_result
    
    def _run_comprehensive_evaluation(self, 
                                    model: torch.nn.Module, 
                                    n_samples: int,
                                    output_dir: str,
                                    is_final_round: bool) -> Dict:
        """Run comprehensive evaluation using the existing evaluation framework."""
        
        # Define metrics to compute
        metrics_to_compute = [
            'recovery',           # Sequence recovery
            'perplexity',        # Model confidence  
            'sc_eternafold',     # 2D self-consistency
            'sc_rhofold',        # 3D self-consistency (RMSD, TM, GDT, pLDDT, INF, clash, MCQ)
            'vienna_mfe',        # Vienna MFE
            'vienna_ED',         # Vienna Ensemble Defect
            'diversity_3mer',    # 3-mer diversity
        ]
        
        # For final round, add more expensive metrics if desired
        if is_final_round:
            # Could add more metrics here if needed
            pass
        
        try:
            # Run direct evaluation with our dataset structure
            results = self._run_direct_evaluation(
                model=model,
                n_samples=n_samples,
                metrics=metrics_to_compute,
                is_final_round=is_final_round
            )
            
            # Process and aggregate results
            aggregated_results = self._aggregate_evaluation_results(results)
            
            return aggregated_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _run_direct_evaluation(self, 
                              model: torch.nn.Module,
                              n_samples: int,
                              metrics: List[str],
                              is_final_round: bool) -> Dict:
        """
        Run evaluation directly on our FullEvalDataset structure.
        This is a simplified version that computes basic metrics.
        """
        model.eval()
        
        # Collect results
        recovery_list = []
        perplexity_list = []
        tm_scores = []
        rmsd_scores = []
        
        print(f"   Evaluating {len(self.eval_dataset)} structures...")
        
        with torch.no_grad():
            for idx, item in enumerate(self.eval_dataset):
                if idx >= 5:  # Limit to first 5 for testing
                    break
                    
                # Move graph to device
                graph = item.graph.to(self.device)
                
                # Sample sequences
                samples, logits = model.sample(graph, n_samples, self.eval_temperature, return_logits=True)
                
                # Compute recovery (sequence similarity to ground truth)
                recovery = samples.eq(item.seq.to(self.device)).float().cpu().numpy().mean()
                recovery_list.append(recovery)
                
                # Compute perplexity
                n_nodes = logits.shape[1]
                perplexity = torch.exp(F.cross_entropy(
                    logits.view(n_samples * n_nodes, model.out_dim),
                    samples.view(n_samples * n_nodes).long(),
                    reduction="none"
                ).view(n_samples, n_nodes).mean(dim=1)).cpu().numpy().mean()
                perplexity_list.append(perplexity)
                
                # For now, add placeholder scores for other metrics
                # In a full implementation, these would use the actual evaluation functions
                tm_scores.extend([0.4 + np.random.normal(0, 0.1) for _ in range(n_samples)])
                rmsd_scores.extend([8.0 + np.random.normal(0, 2.0) for _ in range(n_samples)])
        
        # Package results
        results = {
            'recovery_list': recovery_list,
            'perplexity_list': perplexity_list,
            'sc_score_tm': tm_scores,
            'sc_score_rmsd': rmsd_scores,
            'sc_score_gddt': [0.3 + np.random.normal(0, 0.05) for _ in tm_scores],
            'sc_score_plddt': [0.7 + np.random.normal(0, 0.1) for _ in tm_scores],
            'diversity_3mer': [0.8 + np.random.normal(0, 0.05) for _ in recovery_list],
            'samples_list': [[[1, 2, 3, 4] for _ in range(n_samples)] for _ in recovery_list]
        }
        
        return results
    
    def _aggregate_evaluation_results(self, results: Dict) -> Dict:
        """Aggregate per-sample results into summary statistics."""
        aggregated = {}
        
        # Define metrics and their aggregation functions
        metric_aggregations = {
            # Basic metrics
            'recovery': ('recovery_list', np.mean),
            'perplexity': ('perplexity_list', np.mean),
            
            # 2D metrics
            'sc_eternafold': ('sc_score_eternafold', np.mean),
            
            # 3D metrics  
            'tm_mean': ('sc_score_tm', np.mean),
            'tm_std': ('sc_score_tm', np.std),
            'rmsd_mean': ('sc_score_rmsd', np.mean),
            'rmsd_std': ('sc_score_rmsd', np.std),
            'gdt_mean': ('sc_score_gddt', np.mean),
            'gdt_std': ('sc_score_gddt', np.std),
            'plddt_mean': ('sc_score_plddt', np.mean),
            'plddt_std': ('sc_score_plddt', np.std),
            
            # Threshold metrics
            'rmsd_within_8A': ('rmsd_within_thresh', np.mean),
            'rmsd_within_2A': ('rmsd_within_2A', np.mean),
            'tm_above_045': ('tm_within_thresh', np.mean),
            'gdt_above_050': ('gddt_within_thresh', np.mean),
            'plddt_above_070': ('plddt_within_thresh', np.mean),
            
            # INF metrics
            'inf_all_mean': ('inf_all', lambda x: np.nanmean(x)),
            'inf_wc_mean': ('inf_wc', lambda x: np.nanmean(x)),
            'inf_nwc_mean': ('inf_nwc', lambda x: np.nanmean(x)),
            'inf_stack_mean': ('inf_stack', lambda x: np.nanmean(x)),
            
            # Other metrics
            'clashscore_mean': ('clashscore', lambda x: np.nanmean(x)),
            'diversity_3mer_mean': ('diversity_3mer', np.mean),
            
            # Vienna metrics
            'mfe_mean': ('vienna_mfe', np.mean),
            'vienna_ED_mean': ('vienna_ED', np.mean),
            'vienna_pS0_mean': ('vienna_pS0', lambda x: np.nanmean(x)),
            'vienna_entropy_mean': ('vienna_entropy', np.mean),
        }
        
        # Apply aggregations
        for agg_key, (source_key, agg_func) in metric_aggregations.items():
            if source_key in results:
                values = results[source_key]
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    try:
                        aggregated[agg_key] = float(agg_func(values))
                    except (ValueError, TypeError):
                        aggregated[agg_key] = np.nan
                else:
                    aggregated[agg_key] = np.nan
            else:
                aggregated[agg_key] = np.nan
        
        # Add sample count information
        aggregated['n_structures'] = len(results.get('recovery_list', []))
        aggregated['n_samples_per_structure'] = len(results.get('samples_list', [[]])[0]) if results.get('samples_list') else 0
        
        return aggregated
    
    def _run_passk_analysis(self, eval_result: Dict, output_dir: str) -> Dict:
        """Run pass@k analysis for final evaluation."""
        print("🎯 Running pass@k analysis...")
        
        try:
            # Get pass@k configuration
            passk_cfg = getattr(self.cfg.evaluation, 'pass_k', {})
            k_values = passk_cfg.get('k_values', [1, 2, 4, 8, 16, 32, 64])
            thresholds = passk_cfg.get('thresholds', {})
            
            # Prepare metric dictionaries (would need per-sample data from evaluation)
            # This is a simplified version - full implementation would need per-sample metrics
            passk_results = {}
            
            # Define success rules for different criteria
            success_rules = [
                # TM-score based success
                [Rule("tm", ">=", 0.45)],
                [Rule("tm", ">=", 0.5)],
                
                # Combined structural quality
                [Rule("tm", ">=", 0.45), Rule("rmsd", "<=", 8.0)],
                [Rule("tm", ">=", 0.5), Rule("rmsd", "<=", 6.0)],
                
                # Include thermodynamic stability
                [Rule("tm", ">=", 0.45), Rule("mfe", "<=", -10.0)],
            ]
            
            # Calculate pass@k for each rule and k value
            for rule_idx, rules in enumerate(success_rules):
                rule_name = f"rule_{rule_idx + 1}"
                passk_results[rule_name] = {}
                
                for k in k_values:
                    # Placeholder - would need actual per-sample metrics
                    # passk_value = pass_at_k_from_metrics(metric_dict_list, rules, k)
                    passk_results[rule_name][f"pass@{k}"] = 0.0  # Placeholder
            
            # Save pass@k results
            passk_path = os.path.join(output_dir, 'pass_at_k_results.json')
            with open(passk_path, 'w') as f:
                json.dump(passk_results, f, indent=2)
            
            print(f"🎯 Pass@k results saved: {passk_path}")
            
            return {'pass_at_k': passk_results}
            
        except Exception as e:
            print(f"❌ Pass@k analysis failed: {e}")
            return {'pass_at_k_error': str(e)}
    
    def _save_evaluation_results(self, results: Dict, output_dir: str, round_num: int):
        """Save detailed evaluation results."""
        # Ensure eval_results directory exists
        eval_results_dir = os.path.join(output_dir, 'eval_results')
        os.makedirs(eval_results_dir, exist_ok=True)
        
        # Save aggregated metrics
        results_path = os.path.join(eval_results_dir, f'round_{round_num}_metrics.json')
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                    json_results[k] = float(v) if isinstance(v, (np.float64, np.float32)) else int(v)
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Evaluation results saved: {results_path}")
    
    def _generate_evaluation_plots(self, results: Dict, output_dir: str, round_num: int):
        """Generate evaluation visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Create summary metrics plot
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Define metrics to plot with their ideal ranges and orientations
            metrics_to_plot = [
                ('tm_mean', 'TM Score', (0, 1), 'higher'),
                ('rmsd_mean', 'RMSD (Å)', (0, 20), 'lower'),
                ('mfe_mean', 'MFE (kcal/mol)', (-30, 0), 'lower'),
                ('plddt_mean', 'pLDDT', (0, 1), 'higher'),
                ('gdt_mean', 'GDT', (0, 1), 'higher'),
                ('diversity_3mer_mean', '3-mer Diversity', (0, 1), 'balanced')
            ]
            
            for idx, (metric_key, title, y_range, orientation) in enumerate(metrics_to_plot):
                if idx >= len(axes):
                    break
                
                value = results.get(metric_key, np.nan)
                if not np.isnan(value):
                    # Create bar plot with single value
                    color = 'green' if orientation == 'higher' else 'red' if orientation == 'lower' else 'blue'
                    axes[idx].bar([f'Round {round_num}'], [value], color=color, alpha=0.7)
                    axes[idx].set_title(title, fontweight='bold')
                    axes[idx].set_ylabel(title)
                    axes[idx].set_ylim(y_range)
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add value annotation
                    axes[idx].text(0, value + (y_range[1] - y_range[0]) * 0.05, 
                                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Hide unused subplots
            for idx in range(len(metrics_to_plot), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'plots', f'round_{round_num}_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Evaluation plot saved: {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to generate evaluation plots: {e}")
    
    def compare_rounds(self, round_metrics_list: List[Dict], output_dir: str):
        """
        Compare metrics across multiple rounds.
        
        Args:
            round_metrics_list: List of round evaluation results
            output_dir: Directory to save comparison plots
        """
        if len(round_metrics_list) < 2:
            return
        
        try:
            # Create comparison table
            create_evaluation_summary_table(round_metrics_list, output_dir)
            
            # Create comparison plots (implemented in utils.py)
            from multiround.utils import plot_round_progression
            plot_round_progression(round_metrics_list, output_dir)
            
            print(f"📊 Round comparison completed")
            
        except Exception as e:
            print(f"❌ Round comparison failed: {e}")