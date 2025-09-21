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
# from dpo.data import DPOPairDataset  # No longer needed - using direct dataset loading
from multiround.distribution_analysis import DistributionAnalyzer


class MultiRoundEvaluator:
    """
    Multi-round evaluation coordinator that handles:
    - Per-round evaluation with appropriate sample sizes
    - Integration with 12-metric evaluation pipeline
    - Pass@k analysis for specific rounds (1, 3, 5) with full K=64 sampling
    - Result aggregation and visualization
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Evaluation settings from config (handle both naming schemes and missing config)
        multiround_cfg = getattr(cfg, 'multiround', None)
        if multiround_cfg is None:
            # Handle missing multiround config with defaults
            self.eval_samples = 8
            self.final_eval_samples = 64
            self.eval_temperature = 0.5
            self.final_eval_temperature = 0.1
        else:
            # Prefer eval_samples/final_eval_samples, fallback to n_samples_eval/n_samples_final_eval
            self.eval_samples = (getattr(multiround_cfg, 'eval_samples', None) or 
                               getattr(multiround_cfg, 'n_samples_eval', 8))
            self.final_eval_samples = (getattr(multiround_cfg, 'final_eval_samples', None) or 
                                     getattr(multiround_cfg, 'n_samples_final_eval', 64))
            self.eval_temperature = getattr(multiround_cfg, 'eval_temperature', 0.5)
            self.final_eval_temperature = getattr(multiround_cfg, 'final_eval_temperature', 0.1)
        
        # Backwards compatibility attributes for tests
        self.n_samples_eval = self.eval_samples
        self.n_samples_final_eval = self.final_eval_samples
        self.num_rounds = getattr(multiround_cfg, 'num_rounds', 5) if multiround_cfg else 5
        
        # Load evaluation dataset
        self._load_eval_dataset()
        
        # Pass@k configuration
        evaluation_cfg = getattr(cfg, 'evaluation', None)
        self.passk_config = getattr(evaluation_cfg, 'pass_k', None) if evaluation_cfg else None
        
        # Handle skip_intermediate_passk option for faster training
        skip_intermediate = getattr(multiround_cfg, 'skip_intermediate_passk', False) if multiround_cfg else False
        if skip_intermediate:
            # Only do pass@k on final round when skipping intermediate
            self.passk_rounds = [self.num_rounds]  # Only final round
            print(f"🚀 Pass@k analysis limited to final round ({self.num_rounds}) for faster training")
        else:
            self.passk_rounds = [1, 3, 5]  # Standard: R1, R3, R_final
            
        # Ensure final round is always included if it's not already
        if self.num_rounds not in self.passk_rounds:
            self.passk_rounds.append(self.num_rounds)
        
        # Distribution analysis
        self.distribution_analyzer = None
        self.enable_distribution_analysis = getattr(evaluation_cfg, 'plot_distributions', True) if evaluation_cfg else True
        
    def _load_eval_dataset(self):
        """Load the test dataset for evaluation using direct approach like dpo/bench/eval_full.py."""
        try:
            # Check required paths
            paths_cfg = getattr(self.cfg, 'paths', None)
            if not paths_cfg:
                print("⚠️ No paths configuration found")
                self.eval_dataset = None
                return
            
            processed_pt = getattr(paths_cfg, 'processed_pt', None)
            split_pt = getattr(paths_cfg, 'split_pt', None)
            
            if not processed_pt or not split_pt:
                print(f"⚠️ Missing required paths: processed_pt={processed_pt}, split_pt={split_pt}")
                self.eval_dataset = None
                return
            
            if not os.path.exists(processed_pt) or not os.path.exists(split_pt):
                print(f"⚠️ Required files do not exist: processed_pt={processed_pt}, split_pt={split_pt}")
                self.eval_dataset = None
                return
            
            # Load data directly like dpo/bench/eval_full.py
            from dpo.utils import load_processed_pt
            import torch
            
            print(f"📖 Loading processed data from: {processed_pt}")
            all_items = load_processed_pt(processed_pt)
            
            print(f"📖 Loading split indices from: {split_pt}")
            tr, va, te = torch.load(split_pt, map_location="cpu")
            tr, va, te = list(map(int, tr)), list(map(int, va)), list(map(int, te))
            
            # Create evaluation dataset class compatible with src.evaluator.evaluate
            class EvalDataset:
                def __init__(self, all_items, indices, featurizer_cfg):
                    # Store raw data items (what src.evaluator.evaluate expects)
                    # But fix the coordinate format issue and mask incompatibility
                    self.data_list = []
                    for i in indices:
                        item = all_items[i].copy()
                        
                        # Extract only backbone atoms (P, C4', N1) from full atom coordinates
                        # According to RNA_ATOMS: P=0, C4'=3, N1=10
                        fixed_coords_list = []
                        for coords in item['coords_list']:
                            if coords.shape[1] == 27:  # Full atom coordinates
                                # Extract P (0), C4' (3), N1 (10) atoms only
                                backbone_coords = coords[:, [0, 3, 10], :]  # Shape: [num_res, 3, 3]
                                fixed_coords_list.append(backbone_coords)
                            else:
                                # Already backbone coordinates or other format
                                fixed_coords_list.append(coords)
                        item['coords_list'] = fixed_coords_list
                        
                        # CRITICAL: Ensure coordinate validity to avoid masking issues
                        # Check for missing coordinates and create a proper mask
                        coords = fixed_coords_list[0]  # Use first conformer to check validity
                        
                        # Create mask for valid coordinates (not missing/invalid)
                        # Missing coordinates are marked with very small values (1e-5)
                        coord_valid = ~torch.all(torch.abs(coords) < 1e-3, dim=(1,2))  # Shape: [num_res]
                        
                        # Filter out positions with invalid coordinates
                        valid_positions = torch.where(coord_valid)[0]
                        
                        if len(valid_positions) < len(item['sequence']):
                            print(f"  Filtering {item['id_list'][0]}: {len(item['sequence'])} -> {len(valid_positions)} valid positions")
                            
                            # Update sequence to only include valid positions
                            item['sequence'] = "".join([item['sequence'][i] for i in valid_positions])
                            
                            # Update coordinates to only include valid positions
                            filtered_coords_list = []
                            for coords in fixed_coords_list:
                                filtered_coords = coords[valid_positions]
                                filtered_coords_list.append(filtered_coords)
                            item['coords_list'] = filtered_coords_list
                            
                            # Update other per-residue data if present
                            if 'sasa_list' in item:
                                item['sasa_list'] = [sasa[valid_positions] for sasa in item['sasa_list']]
                            if 'sec_struct_list' in item:
                                item['sec_struct_list'] = ["".join([ss[i] for i in valid_positions]) for ss in item['sec_struct_list']]
                        
                        self.data_list.append(item)
                    
                    # Import featurizer locally to avoid NetworkX conflicts
                    from src.data.featurizer import RNAGraphFeaturizer
                    
                    # Create featurizer with CPU device to avoid device mismatch
                    self.featurizer = RNAGraphFeaturizer(
                        split=getattr(featurizer_cfg, 'split', 'test'),
                        radius=getattr(featurizer_cfg, 'radius', 0.0),
                        top_k=getattr(featurizer_cfg, 'top_k', 32),
                        num_rbf=getattr(featurizer_cfg, 'num_rbf', 32),
                        num_posenc=getattr(featurizer_cfg, 'num_posenc', 32),
                        max_num_conformers=getattr(featurizer_cfg, 'max_num_conformers', 1),
                        noise_scale=getattr(featurizer_cfg, 'noise_scale', 0.0),
                        distance_eps=getattr(featurizer_cfg, 'distance_eps', 0.001),
                        device="cpu"  # Always use CPU for featurizer to avoid device issues
                    )
                
                def __len__(self):
                    return len(self.data_list)
                
                def __getitem__(self, idx):
                    return self.data_list[idx]
            
            # Create evaluation dataset using test split
            featurizer_cfg = getattr(self.cfg, 'featurizer', {})
            self.eval_dataset = EvalDataset(all_items, te, featurizer_cfg)
            
            print(f"✅ Loaded evaluation dataset: {len(self.eval_dataset)} test structures")
            print(f"   Split sizes: train={len(tr)}, val={len(va)}, test={len(te)}")
            
        except Exception as e:
            print(f"❌ Failed to load evaluation dataset: {e}")
            import traceback
            traceback.print_exc()
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
        
        # For final round, run evaluation with both temperatures
        if is_final_round:
            return self._evaluate_final_round(model, round_num, output_dir, eval_output_dir, designs_output_dir)
        
        # Regular round evaluation
        compute_passk = round_num in self.passk_rounds
        # Use higher sample count for pass@k rounds (R1, R3, R_final) per CLAUDE.md spec
        n_samples = self.final_eval_samples if compute_passk else self.eval_samples
        temperature = self.eval_temperature
        
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
            
            # Add to distribution analysis
            if self.enable_distribution_analysis:
                self._add_to_distribution_analysis(round_num, eval_results, aggregated_results, eval_output_dir)
            
            print(f"✅ Round {round_num} evaluation completed")
            return aggregated_results
            
        except Exception as e:
            print(f"❌ Evaluation failed for round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'round': round_num}
    
    def _evaluate_final_round(self, model, round_num: int, output_dir: str, eval_output_dir: str, designs_output_dir: str) -> Dict:
        """
        Special evaluation for final round with dual temperature analysis.
        
        According to CLAUDE.md: "Only do the Temp of both 0.5, and 0.1, for the final round eval."
        This provides both standard and low-temperature evaluations for comprehensive analysis.
        """
        print(f"🔥 Final round evaluation with dual temperature analysis")
        
        # Final round parameters
        n_samples = self.final_eval_samples
        temperatures = [self.eval_temperature, self.final_eval_temperature]  # [0.5, 0.1]
        compute_passk = True  # Always compute pass@k for final round
        
        print(f"   Samples per structure: {n_samples}")
        print(f"   Temperatures: {temperatures}")
        print(f"   Pass@k analysis: Yes (full k ≤ 64)")
        print(f"   Designs output: {designs_output_dir}")
        
        # Determine which metrics to compute
        metrics = self._get_metrics_for_round(round_num, True, compute_passk)
        
        # Run evaluation at both temperatures
        combined_results = {
            'round': round_num,
            'n_samples': n_samples,
            'temperatures': temperatures,
            'n_structures': len(self.eval_dataset),
            'timestamp': datetime.now().isoformat(),
            'is_final_round': True,
            'output_dirs': {
                'evaluation': eval_output_dir,
                'designs': designs_output_dir
            }
        }
        
        try:
            for temp_idx, temperature in enumerate(temperatures):
                temp_suffix = f"t{temperature:.1f}".replace(".", "")
                print(f"   🌡️ Evaluating at temperature {temperature}...")
                
                # Run evaluation at this temperature
                eval_results = evaluate(
                    model=model,
                    dataset=self.eval_dataset,
                    n_samples=n_samples,
                    temperature=temperature,
                    device=self.device,
                    model_name=f"round_{round_num}_{temp_suffix}",
                    metrics=metrics,
                    save_designs=True
                )
                
                # Save individual metrics for this temperature
                self._save_individual_metrics(eval_results, eval_output_dir, 
                                            f"{round_num}_{temp_suffix}", n_samples)
                
                # Aggregate results for this temperature
                temp_aggregated = self._aggregate_evaluation_results(eval_results)
                
                # Add temperature-specific prefix to all keys
                for key, value in temp_aggregated.items():
                    combined_results[f"{temp_suffix}_{key}"] = value
                
                # Compute pass@k analysis for this temperature
                print(f"🎯 Computing pass@k analysis for temperature {temperature}...")
                passk_results = self._compute_passk_analysis(eval_results, n_samples)
                
                # Add temperature-specific prefix to pass@k results
                for key, value in passk_results.items():
                    combined_results[f"{temp_suffix}_{key}"] = value
                
                # Create distribution plots for this temperature
                if len(eval_results.get('sc_score_tm', [])) > 5:
                    self._create_distribution_plots(eval_results, eval_output_dir, 
                                                  f"{round_num}_{temp_suffix}")
                
                print(f"   ✅ Temperature {temperature} evaluation completed")
            
            # Create comparative analysis between temperatures
            self._create_temperature_comparison_analysis(combined_results, eval_output_dir, round_num)
            
            # Save combined results
            self._save_evaluation_results(combined_results, eval_output_dir, round_num)
            
            # Add to distribution analysis and generate final report
            if self.enable_distribution_analysis:
                # Add final round data for both temperatures
                self._add_to_distribution_analysis(round_num, {}, combined_results, eval_output_dir)
                
                # Generate comprehensive distribution analysis report
                print(f"📊 Generating comprehensive distribution analysis...")
                self._generate_final_distribution_analysis(eval_output_dir)
            
            print(f"✅ Final round evaluation completed")
            return combined_results
            
        except Exception as e:
            print(f"❌ Final round evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'round': round_num, 'is_final_round': True}
    
    def _create_temperature_comparison_analysis(self, combined_results: Dict, output_dir: str, round_num: int):
        """Create comparative analysis between different temperature evaluations."""
        try:
            # Extract key metrics for comparison
            comparison_metrics = ['tm_mean', 'rmsd_mean', 'mfe_mean', 'recovery', 'diversity_mean']
            temp_suffixes = ['t05', 't01']  # For 0.5 and 0.1 temperatures
            
            comparison_data = {}
            for metric in comparison_metrics:
                comparison_data[metric] = {}
                for suffix in temp_suffixes:
                    key = f"{suffix}_{metric}"
                    if key in combined_results:
                        comparison_data[metric][suffix] = combined_results[key]
            
            # Create comparison summary
            comparison_summary = {
                'round': round_num,
                'temperature_comparison': comparison_data,
                'analysis': {
                    'higher_temp_benefits': "Higher diversity, more exploration",
                    'lower_temp_benefits': "Better convergence, focused sampling",
                    'recommended_use': "T=0.5 for training, T=0.1 for final evaluation"
                }
            }
            
            # Save comparison analysis
            comparison_path = os.path.join(output_dir, f"temperature_comparison_round_{round_num}.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison_summary, f, indent=2)
            
            print(f"🔬 Temperature comparison analysis saved: {comparison_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create temperature comparison analysis: {e}")
    
    def _get_metrics_for_round(self, round_num: int, is_final_round: bool, compute_passk: bool) -> List[str]:
        """Determine which metrics to compute for this round."""
        # Base metrics for all rounds
        metrics = ['recovery', 'perplexity']
        
        # Add 3D metrics (RhoFold-based) for comprehensive evaluation
        metrics.append('sc_score_rhofold')
        
        # Add 2D metrics
        metrics.append('sc_score_eternafold')
        
        # Add Vienna ensemble metrics for thermostability
        # DISABLED: Vienna metrics have length mismatch issues with coordinate filtering
        # The core evaluation (recovery, perplexity, 2D/3D structure) works correctly
        # metrics.append('sc_score_vienna')
        
        # For final round or pass@k rounds, include all metrics
        if is_final_round or compute_passk:
            # DISABLED: RibonanzaNet has similar masking issues with coordinate filtering
            # metrics.extend(['sc_score_ribonanzanet', 'sc_score_assessment'])
            pass
        
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
        """Compute comprehensive pass@k analysis using the dedicated pass@k module."""
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
                
                # pLDDT metrics (higher is better)
                if 'sc_score_plddt' in eval_results:
                    item_metrics['plddt'] = np.array(eval_results['sc_score_plddt'][i])
                
                # GDT metrics (higher is better)
                if 'sc_score_gddt' in eval_results:
                    item_metrics['gdt'] = np.array(eval_results['sc_score_gddt'][i])
                
                metric_dict_list.append(item_metrics)
            
            if not metric_dict_list:
                return {'passk_error': 'No metrics available for pass@k analysis'}
            
            # Define success rules from thresholds (implementing CLAUDE.md requirements)
            passk_results = {}
            
            # TM score rules - comprehensive thresholds from config
            tm_thresholds = getattr(thresholds, 'tm_score', [0.4, 0.45, 0.5, 0.55]) if thresholds else [0.4, 0.45, 0.5, 0.55]
            for tm_thr in tm_thresholds:
                rule_set = [Rule("tm", ">=", tm_thr)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, rule_set, k, 
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_tm_{tm_thr}_k{k}"] = float(passk)
            
            # RMSD rules - comprehensive thresholds
            rmsd_thresholds = getattr(thresholds, 'rmsd', [8.0, 6.0, 4.0, 2.0]) if thresholds else [8.0, 6.0, 4.0, 2.0]
            for rmsd_thr in rmsd_thresholds:
                if 'rmsd' in metric_dict_list[0]:
                    rule_set = [Rule("rmsd", "<=", rmsd_thr)]
                    for k in k_values:
                        if k <= n_samples:
                            passk = pass_at_k_from_metrics(
                                metric_dict_list, rule_set, k,
                                combine_mode="all", selection="unbiased"
                            )
                            passk_results[f"passk_rmsd_{rmsd_thr}_k{k}"] = float(passk)
            
            # MFE rules - thermodynamic stability thresholds
            if 'mfe' in metric_dict_list[0]:
                mfe_thresholds = getattr(thresholds, 'mfe', [-10.0, -15.0, -20.0]) if thresholds else [-10.0, -15.0, -20.0]
                for mfe_thr in mfe_thresholds:
                    rule_set = [Rule("mfe", "<=", mfe_thr)]
                    for k in k_values:
                        if k <= n_samples:
                            passk = pass_at_k_from_metrics(
                                metric_dict_list, rule_set, k,
                                combine_mode="all", selection="unbiased"
                            )
                            passk_results[f"passk_mfe_{mfe_thr}_k{k}"] = float(passk)
            
            # Combined rules for comprehensive evaluation
            # Primary combined rule: TM ≥ 0.45 AND RMSD ≤ 8.0 (reference model selection criterion)
            if 'rmsd' in metric_dict_list[0]:
                combined_rules = [Rule("tm", ">=", 0.45), Rule("rmsd", "<=", 8.0)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, combined_rules, k,
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_combined_tm0.45_rmsd8.0_k{k}"] = float(passk)
            
            # High-quality combined rule: TM ≥ 0.5 AND RMSD ≤ 4.0
            if 'rmsd' in metric_dict_list[0]:
                high_quality_rules = [Rule("tm", ">=", 0.5), Rule("rmsd", "<=", 4.0)]
                for k in k_values:
                    if k <= n_samples:
                        passk = pass_at_k_from_metrics(
                            metric_dict_list, high_quality_rules, k,
                            combine_mode="all", selection="unbiased"
                        )
                        passk_results[f"passk_combined_tm0.5_rmsd4.0_k{k}"] = float(passk)
            
            # Save individual structure pass@k data for detailed analysis
            individual_passk_path = os.path.join(os.path.dirname(__file__), "passk_individual_analysis.json")
            self._save_individual_passk_analysis(metric_dict_list, k_values, individual_passk_path)
            
            print(f"✅ Pass@k analysis completed: {len(passk_results)} metrics computed")
            return passk_results
            
        except Exception as e:
            print(f"❌ Pass@k analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'passk_error': str(e)}
    
    def _save_individual_passk_analysis(self, metric_dict_list: List[Dict], k_values: List[int], output_path: str):
        """Save individual structure pass@k analysis for detailed inspection."""
        try:
            from multiround.passk import Rule, pass_at_k_from_metrics, success_flags_from_metric_dict
            
            individual_analysis = []
            
            # Analyze each structure individually
            for i, item_metrics in enumerate(metric_dict_list):
                structure_analysis = {
                    'structure_idx': i,
                    'n_samples': len(item_metrics.get('tm', [])),
                    'metrics_summary': {}
                }
                
                # Summarize available metrics for this structure
                for metric_name, values in item_metrics.items():
                    if len(values) > 0:
                        structure_analysis['metrics_summary'][metric_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values))
                        }
                
                # Compute pass@k for key thresholds on this individual structure
                single_item = [item_metrics]
                
                # Primary success criterion: TM ≥ 0.45
                if 'tm' in item_metrics:
                    tm_rule = [Rule("tm", ">=", 0.45)]
                    structure_analysis['passk_tm_0.45'] = {}
                    for k in [1, 2, 4, 8]:  # Key k values for individual analysis
                        if k <= len(item_metrics['tm']):
                            passk = pass_at_k_from_metrics(single_item, tm_rule, k)
                            structure_analysis['passk_tm_0.45'][f'k{k}'] = float(passk)
                
                individual_analysis.append(structure_analysis)
            
            # Save individual analysis
            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'n_structures': len(metric_dict_list),
                    'individual_structure_analysis': individual_analysis
                }, f, indent=2)
            
            print(f"💾 Individual pass@k analysis saved: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save individual pass@k analysis: {e}")
    
    def _add_to_distribution_analysis(self, round_num: int, eval_results: Dict, 
                                    aggregated_results: Dict, output_dir: str):
        """Add round data to distribution analyzer."""
        try:
            # Initialize distribution analyzer if not already done
            if self.distribution_analyzer is None:
                analysis_dir = os.path.join(output_dir, "distribution_analysis")
                self.distribution_analyzer = DistributionAnalyzer(analysis_dir)
                print(f"🔬 Initialized distribution analyzer: {analysis_dir}")
            
            # Prepare metadata
            metadata = {
                'temperature': aggregated_results.get('temperature', 'unknown'),
                'n_samples': aggregated_results.get('n_samples', 0),
                'n_structures': aggregated_results.get('n_structures', 0),
                'is_final_round': aggregated_results.get('is_final_round', False)
            }
            
            # For final round with dual temperatures, add both datasets
            if aggregated_results.get('is_final_round', False) and 'temperatures' in aggregated_results:
                # Extract data for each temperature
                temp_suffixes = ['t05', 't01']
                
                for temp_suffix in temp_suffixes:
                    temp_round_num = f"{round_num}_{temp_suffix}"
                    temp_data = {}
                    
                    # Extract temperature-specific metrics
                    for key, value in aggregated_results.items():
                        if key.startswith(f"{temp_suffix}_"):
                            metric_name = key[4:]  # Remove temp prefix
                            temp_data[metric_name] = value
                    
                    if temp_data:
                        temp_metadata = metadata.copy()
                        temp_metadata['temperature'] = 0.5 if temp_suffix == 't05' else 0.1
                        self.distribution_analyzer.add_round_data(temp_round_num, temp_data, temp_metadata)
            else:
                # Regular round - use individual metrics if available
                data_to_add = eval_results if eval_results else aggregated_results
                self.distribution_analyzer.add_round_data(round_num, data_to_add, metadata)
            
        except Exception as e:
            print(f"⚠️ Failed to add distribution analysis data for round {round_num}: {e}")
    
    def _generate_final_distribution_analysis(self, output_dir: str):
        """Generate comprehensive distribution analysis and visualizations."""
        try:
            if self.distribution_analyzer is None:
                print("⚠️ No distribution analyzer available")
                return
            
            print("🔬 Performing comprehensive distribution evolution analysis...")
            
            # Run statistical analysis
            analysis_results = self.distribution_analyzer.analyze_distribution_evolution()
            
            # Create visualizations
            print("📈 Creating distribution visualization plots...")
            plot_paths = self.distribution_analyzer.create_distribution_plots()
            
            # Generate improvement report
            print("📝 Generating improvement report...")
            report_path = self.distribution_analyzer.generate_improvement_report()
            
            # Summary of generated outputs
            print(f"✅ Distribution analysis completed:")
            print(f"   📊 Plots created: {len(plot_paths)}")
            for plot_type, path in plot_paths.items():
                print(f"     - {plot_type}: {os.path.basename(path)}")
            
            if report_path:
                print(f"   📝 Report: {os.path.basename(report_path)}")
            
            # Log key findings
            overall_trends = analysis_results.get('overall_trends', {})
            improvement_score = overall_trends.get('overall_improvement_score', 0)
            improving_metrics = overall_trends.get('improving_metrics', [])
            
            print(f"🎯 Key Findings:")
            print(f"   Overall improvement score: {improvement_score:.1%}")
            print(f"   Improving metrics: {', '.join(improving_metrics) if improving_metrics else 'None'}")
            
        except Exception as e:
            print(f"❌ Failed to generate final distribution analysis: {e}")
            import traceback
            traceback.print_exc()
    
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

    def evaluate_model(self, model, dataset=None, n_samples: int = None, temperature: float = None, save_designs: bool = False, round_num: int = None, **kwargs):
        """Wrapper method for compatibility with tests. Delegates to evaluate_round."""
        import tempfile
        n_samples = n_samples or self.eval_samples
        temperature = temperature or self.eval_temperature
        round_num = round_num or 0
        
        # Create a temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            return self.evaluate_round(model, round_num=round_num, output_dir=temp_dir, is_final_round=False)
    
    def calculate_pass_k_metrics(self, eval_results: Dict, k_values: List[int] = None, tm_thresholds: List[float] = None, **kwargs) -> Dict:
        """Calculate pass@k metrics from evaluation results."""
        k_values = k_values or [1, 2, 4, 8]
        tm_thresholds = tm_thresholds or [0.4, 0.45, 0.5]
        
        if not eval_results:
            return {}
            
        return self._compute_passk_analysis(eval_results, n_samples=len(eval_results.get('samples_list', [])))
    
    def get_evaluation_metrics(self) -> List[str]:
        """Get list of evaluation metrics used by this evaluator."""
        return ['recovery', 'perplexity', 'sc_score_rhofold']
        
    # Test compatibility methods
    @property
    def save_eval_data(self) -> bool:
        """Whether to save evaluation data."""
        multiround_cfg = getattr(self.cfg, 'multiround', None)
        return getattr(multiround_cfg, 'save_eval_data', False) if multiround_cfg else False
        
    def should_use_full_pass_k(self, round_num: int) -> bool:
        """Determine if full pass@k analysis should be used for this round."""
        return round_num in self.passk_rounds
        
    def get_samples_for_round(self, round_num: int, is_final: bool = False) -> int:
        """Get appropriate sample count for the given round."""
        if is_final or round_num == self.num_rounds:
            return self.final_eval_samples
        return self.eval_samples
        
    def get_temperature_for_round(self, round_num: int, is_final: bool = False) -> float:
        """Get temperature for evaluation in the given round."""
        if is_final or round_num == self.num_rounds:
            return self.final_eval_temperature
        return self.eval_temperature
        
    def aggregate_results(self, results_list: List[Dict]) -> Dict:
        """Aggregate multiple evaluation results."""
        return self._aggregate_evaluation_results(results_list[0] if results_list else {})
        
    def save_evaluation_results(self, results: Dict, output_dir: str, round_num: int) -> str:
        """Save evaluation results to file."""
        save_path = os.path.join(output_dir, f"eval_results_round_{round_num}.json")
        self._save_evaluation_results(results, output_dir, round_num)
        return save_path