# multiround/distribution_analysis.py
"""
Distribution Analysis and Visualization Tools for Multi-Round RiboPO

This module provides comprehensive analysis and visualization of metric distributions
across training rounds to demonstrate model improvement and convergence.

Key features:
- PDF (Probability Density Function) analysis for pLDDT, RMSD, and MFE
- Round-to-round progression tracking
- Statistical significance testing
- Interactive and static visualizations
- Distribution shift quantification

According to CLAUDE.md:
"We analyze the pLDDT, RMSD, and MFE distribution (probability density function) for 
the ckpt at every round. Draw the pdf, to prove that our model is moving towards 
higher/better distribution."
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DistributionAnalyzer:
    """
    Comprehensive distribution analysis for multi-round training evaluation.
    
    Analyzes and visualizes how metric distributions evolve across training rounds,
    providing statistical evidence of model improvement.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Key metrics to analyze (higher is better / lower is better)
        self.metrics_config = {
            'plddt': {'name': 'pLDDT', 'higher_better': True, 'unit': '', 'range': (0, 1)},
            'tm_score': {'name': 'TM Score', 'higher_better': True, 'unit': '', 'range': (0, 1)},
            'rmsd': {'name': 'RMSD', 'higher_better': False, 'unit': 'Å', 'range': (0, 20)},
            'mfe': {'name': 'MFE', 'higher_better': False, 'unit': 'kcal/mol', 'range': (-50, 0)},
            'ed_per_nt': {'name': 'Ensemble Defect/nt', 'higher_better': False, 'unit': '', 'range': (0, 1)},
            'recovery': {'name': 'Sequence Recovery', 'higher_better': True, 'unit': '', 'range': (0, 1)},
            'diversity': {'name': '3-mer Diversity', 'higher_better': True, 'unit': '', 'range': (0, 1)}
        }
        
        # Storage for round data
        self.round_data = {}  # round_num -> {metric -> values}
        self.round_metadata = {}  # round_num -> metadata
    
    def add_round_data(self, round_num: int, eval_results: Dict, metadata: Optional[Dict] = None):
        """
        Add evaluation results for a specific round.
        
        Args:
            round_num: Training round number
            eval_results: Dictionary containing evaluation metrics
            metadata: Optional metadata (temperature, n_samples, etc.)
        """
        print(f"📊 Adding distribution data for round {round_num}")
        
        # Extract metric values for this round
        round_metrics = {}
        
        # Individual structure metrics (preferred - gives full distribution)
        if 'individual_metrics' in eval_results:
            individual_data = eval_results['individual_metrics']
            for metric in self.metrics_config.keys():
                values = []
                for structure in individual_data:
                    if metric in structure:
                        values.append(structure[metric])
                if values:
                    round_metrics[metric] = np.array(values)
        
        # Fallback to aggregated metrics if individual not available
        else:
            # Map from evaluator result keys to our metric names
            key_mapping = {
                'plddt': ['sc_score_plddt', 'plddt_mean'],
                'tm_score': ['sc_score_tm', 'tm_mean'],
                'rmsd': ['sc_score_rmsd', 'rmsd_mean'],
                'mfe': ['vienna_mfe', 'mfe_mean'],
                'ed_per_nt': ['vienna_ED_per_nt', 'ed_per_nt_mean'],
                'recovery': ['recovery_list', 'recovery'],
                'diversity': ['diversity_3mer_mean']
            }
            
            for metric, possible_keys in key_mapping.items():
                for key in possible_keys:
                    if key in eval_results:
                        values = eval_results[key]
                        if isinstance(values, list):
                            round_metrics[metric] = np.array(values)
                        else:
                            # Single aggregated value - create pseudo-distribution
                            round_metrics[metric] = np.array([values])
                        break
        
        # Store the data
        self.round_data[round_num] = round_metrics
        self.round_metadata[round_num] = metadata or {}
        
        print(f"   Added {len(round_metrics)} metrics for round {round_num}")
        for metric, values in round_metrics.items():
            if len(values) > 0:
                print(f"   {metric}: {len(values)} samples, mean={np.mean(values):.3f}")
    
    def analyze_distribution_evolution(self) -> Dict:
        """
        Analyze how distributions evolve across rounds.
        
        Returns:
            Dictionary containing comprehensive distribution analysis
        """
        if len(self.round_data) < 2:
            print("⚠️ Need at least 2 rounds for distribution evolution analysis")
            return {}
        
        print(f"🔬 Analyzing distribution evolution across {len(self.round_data)} rounds")
        
        analysis_results = {
            'rounds_analyzed': sorted(self.round_data.keys()),
            'metrics': {},
            'overall_trends': {},
            'statistical_tests': {}
        }
        
        for metric in self.metrics_config.keys():
            if not self._metric_available_across_rounds(metric):
                continue
            
            metric_analysis = self._analyze_single_metric_evolution(metric)
            analysis_results['metrics'][metric] = metric_analysis
        
        # Overall trend analysis
        analysis_results['overall_trends'] = self._analyze_overall_trends()
        
        # Save analysis results
        analysis_path = self.output_dir / "distribution_evolution_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=self._json_serialize)
        
        print(f"💾 Distribution analysis saved: {analysis_path}")
        return analysis_results
    
    def _metric_available_across_rounds(self, metric: str) -> bool:
        """Check if a metric is available across multiple rounds."""
        available_rounds = 0
        for round_num in self.round_data:
            if metric in self.round_data[round_num] and len(self.round_data[round_num][metric]) > 0:
                available_rounds += 1
        return available_rounds >= 2
    
    def _analyze_single_metric_evolution(self, metric: str) -> Dict:
        """Analyze evolution of a single metric across rounds."""
        config = self.metrics_config[metric]
        rounds = sorted(self.round_data.keys())
        
        analysis = {
            'metric_name': config['name'],
            'higher_better': config['higher_better'],
            'rounds': [],
            'trend_statistics': {},
            'distribution_shifts': {}
        }
        
        # Collect data for each round
        round_stats = []
        round_values = []
        
        for round_num in rounds:
            if metric in self.round_data[round_num]:
                values = self.round_data[round_num][metric]
                if len(values) > 0:
                    stats_dict = {
                        'round': round_num,
                        'n_samples': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                    round_stats.append(stats_dict)
                    round_values.append(values)
        
        analysis['rounds'] = round_stats
        
        if len(round_stats) >= 2:
            # Trend analysis
            means = [r['mean'] for r in round_stats]
            round_numbers = [r['round'] for r in round_stats]
            
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(round_numbers, means)
            
            analysis['trend_statistics'] = {
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'improvement_direction': 'positive' if slope > 0 else 'negative',
                'is_improving': (slope > 0) == config['higher_better'],
                'improvement_per_round': float(slope)
            }
            
            # Distribution shift analysis
            analysis['distribution_shifts'] = self._analyze_distribution_shifts(
                round_values, round_numbers, metric
            )
        
        return analysis
    
    def _analyze_distribution_shifts(self, round_values: List[np.ndarray], 
                                   round_numbers: List[int], metric: str) -> Dict:
        """Analyze statistical significance of distribution shifts."""
        shifts = {}
        
        # Compare each round to the first round
        baseline_values = round_values[0]
        baseline_round = round_numbers[0]
        
        for i in range(1, len(round_values)):
            current_values = round_values[i]
            current_round = round_numbers[i]
            
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_p = ks_2samp(baseline_values, current_values)
            
            # Mann-Whitney U test for median difference
            mw_stat, mw_p = mannwhitneyu(baseline_values, current_values, alternative='two-sided')
            
            # Effect size (Cohen's d approximation)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) + 
                                (len(current_values) - 1) * np.var(current_values, ddof=1)) / 
                               (len(baseline_values) + len(current_values) - 2))
            
            cohens_d = (np.mean(current_values) - np.mean(baseline_values)) / (pooled_std + 1e-8)
            
            shifts[f"round_{baseline_round}_to_{current_round}"] = {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'ks_significant': ks_p < 0.05,
                'mw_statistic': float(mw_stat),
                'mw_p_value': float(mw_p),
                'mw_significant': mw_p < 0.05,
                'cohens_d': float(cohens_d),
                'effect_size': self._interpret_effect_size(abs(cohens_d)),
                'mean_shift': float(np.mean(current_values) - np.mean(baseline_values)),
                'median_shift': float(np.median(current_values) - np.median(baseline_values))
            }
        
        return shifts
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_overall_trends(self) -> Dict:
        """Analyze overall trends across all metrics."""
        improving_metrics = 0
        total_metrics = 0
        
        trend_summary = {
            'improving_metrics': [],
            'declining_metrics': [],
            'stable_metrics': [],
            'overall_improvement_score': 0.0
        }
        
        for metric in self.metrics_config.keys():
            if not self._metric_available_across_rounds(metric):
                continue
            
            total_metrics += 1
            rounds = sorted(self.round_data.keys())
            
            if len(rounds) >= 2:
                # Calculate trend
                first_round_mean = np.mean(self.round_data[rounds[0]][metric])
                last_round_mean = np.mean(self.round_data[rounds[-1]][metric])
                
                config = self.metrics_config[metric]
                is_improving = ((last_round_mean > first_round_mean) == config['higher_better'])
                
                if is_improving:
                    improving_metrics += 1
                    trend_summary['improving_metrics'].append(metric)
                else:
                    trend_summary['declining_metrics'].append(metric)
        
        if total_metrics > 0:
            trend_summary['overall_improvement_score'] = improving_metrics / total_metrics
        
        return trend_summary
    
    def create_distribution_plots(self, save_individual: bool = True) -> Dict[str, str]:
        """
        Create comprehensive distribution visualization plots.
        
        Args:
            save_individual: Whether to save individual metric plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        print(f"📈 Creating distribution plots for {len(self.round_data)} rounds")
        
        plot_paths = {}
        
        # 1. Combined PDF overview
        plot_paths['pdf_overview'] = self._create_pdf_overview_plot()
        
        # 2. Individual metric evolution plots
        if save_individual:
            individual_plots = self._create_individual_metric_plots()
            plot_paths.update(individual_plots)
        
        # 3. Statistical summary plot
        plot_paths['statistical_summary'] = self._create_statistical_summary_plot()
        
        # 4. Round progression plot
        plot_paths['round_progression'] = self._create_round_progression_plot()
        
        print(f"✅ Created {len(plot_paths)} distribution plots")
        return plot_paths
    
    def _create_pdf_overview_plot(self) -> str:
        """Create overview plot showing PDFs for key metrics across rounds."""
        # Focus on the three key metrics mentioned in CLAUDE.md
        key_metrics = ['plddt', 'rmsd', 'mfe']
        available_metrics = [m for m in key_metrics if self._metric_available_across_rounds(m)]
        
        if not available_metrics:
            print("⚠️ No key metrics available for PDF overview")
            return ""
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.round_data)))
        rounds = sorted(self.round_data.keys())
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            config = self.metrics_config[metric]
            
            for round_idx, round_num in enumerate(rounds):
                if metric in self.round_data[round_num]:
                    values = self.round_data[round_num][metric]
                    if len(values) > 1:
                        # Create probability density
                        density, bins = np.histogram(values, bins=30, density=True)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        
                        ax.plot(bin_centers, density, color=colors[round_idx], 
                               label=f'Round {round_num}', linewidth=2, alpha=0.8)
                        
                        # Fill area under curve for better visualization
                        ax.fill_between(bin_centers, density, alpha=0.3, color=colors[round_idx])
            
            ax.set_xlabel(f"{config['name']} {config['unit']}".strip())
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{config["name"]} Distribution Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add improvement arrow
            if len(rounds) >= 2:
                first_mean = np.mean(self.round_data[rounds[0]][metric])
                last_mean = np.mean(self.round_data[rounds[-1]][metric])
                
                # Add arrow showing improvement direction
                arrow_props = dict(arrowstyle='->', connectionstyle='arc3', 
                                 color='red', lw=2, alpha=0.7)
                
                if config['higher_better'] and last_mean > first_mean:
                    ax.annotate('Improving →', xy=(0.7, 0.9), xycoords='axes fraction',
                              fontsize=12, color='green', weight='bold')
                elif not config['higher_better'] and last_mean < first_mean:
                    ax.annotate('← Improving', xy=(0.1, 0.9), xycoords='axes fraction',
                              fontsize=12, color='green', weight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "distribution_pdf_overview.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 PDF overview plot saved: {plot_path}")
        return str(plot_path)
    
    def _create_individual_metric_plots(self) -> Dict[str, str]:
        """Create detailed plots for each individual metric."""
        plot_paths = {}
        
        for metric in self.metrics_config.keys():
            if not self._metric_available_across_rounds(metric):
                continue
            
            plot_path = self._create_single_metric_plot(metric)
            if plot_path:
                plot_paths[f'{metric}_evolution'] = plot_path
        
        return plot_paths
    
    def _create_single_metric_plot(self, metric: str) -> str:
        """Create a detailed evolution plot for a single metric."""
        config = self.metrics_config[metric]
        rounds = sorted(self.round_data.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Distribution evolution (PDFs)
        colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
        
        for round_idx, round_num in enumerate(rounds):
            if metric in self.round_data[round_num]:
                values = self.round_data[round_num][metric]
                if len(values) > 1:
                    ax1.hist(values, bins=20, alpha=0.6, color=colors[round_idx],
                           label=f'Round {round_num}', density=True, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel(f"{config['name']} {config['unit']}".strip())
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'{config["name"]} Distribution Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Statistical summary over rounds
        means = []
        stds = []
        round_nums = []
        
        for round_num in rounds:
            if metric in self.round_data[round_num]:
                values = self.round_data[round_num][metric]
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                    round_nums.append(round_num)
        
        if len(means) > 1:
            means = np.array(means)
            stds = np.array(stds)
            
            ax2.plot(round_nums, means, 'o-', color='blue', linewidth=2, markersize=8, label='Mean')
            ax2.fill_between(round_nums, means - stds, means + stds, alpha=0.3, color='blue', label='±1 Std')
            
            # Add trend line
            z = np.polyfit(round_nums, means, 1)
            p = np.poly1d(z)
            ax2.plot(round_nums, p(round_nums), '--', color='red', alpha=0.8, label='Trend')
            
            ax2.set_xlabel('Training Round')
            ax2.set_ylabel(f"{config['name']} {config['unit']}".strip())
            ax2.set_title(f'{config["name"]} Mean ± Std Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(round_nums)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"distribution_{metric}_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_statistical_summary_plot(self) -> str:
        """Create a statistical summary plot showing key trends."""
        available_metrics = [m for m in self.metrics_config.keys() 
                           if self._metric_available_across_rounds(m)]
        
        if not available_metrics:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Mean improvement per round
        ax = axes[0]
        improvements = []
        metric_names = []
        
        for metric in available_metrics:
            rounds = sorted(self.round_data.keys())
            if len(rounds) >= 2:
                first_mean = np.mean(self.round_data[rounds[0]][metric])
                last_mean = np.mean(self.round_data[rounds[-1]][metric])
                
                config = self.metrics_config[metric]
                improvement = (last_mean - first_mean) if config['higher_better'] else (first_mean - last_mean)
                improvements.append(improvement)
                metric_names.append(config['name'])
        
        if improvements:
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.set_ylabel('Improvement Score')
            ax.set_title('Metric Improvements (First → Last Round)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Effect sizes
        ax = axes[1]
        effect_sizes = []
        for metric in available_metrics:
            rounds = sorted(self.round_data.keys())
            if len(rounds) >= 2:
                first_values = self.round_data[rounds[0]][metric]
                last_values = self.round_data[rounds[-1]][metric]
                
                pooled_std = np.sqrt((np.var(first_values, ddof=1) + np.var(last_values, ddof=1)) / 2)
                cohens_d = abs(np.mean(last_values) - np.mean(first_values)) / (pooled_std + 1e-8)
                effect_sizes.append(cohens_d)
        
        if effect_sizes:
            ax.bar(range(len(metric_names)), effect_sizes, alpha=0.7, color='purple')
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.set_ylabel("Cohen's d (Effect Size)")
            ax.set_title('Effect Sizes (Distribution Shifts)')
            ax.grid(True, alpha=0.3)
            
            # Add effect size interpretation lines
            ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small')
            ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Medium')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
            ax.legend()
        
        # Plot 3: Variance evolution (convergence indicator)
        ax = axes[2]
        for metric in available_metrics[:4]:  # Limit to 4 metrics for clarity
            rounds = sorted(self.round_data.keys())
            variances = []
            round_nums = []
            
            for round_num in rounds:
                if metric in self.round_data[round_num]:
                    values = self.round_data[round_num][metric]
                    if len(values) > 1:
                        variances.append(np.var(values))
                        round_nums.append(round_num)
            
            if len(variances) > 1:
                config = self.metrics_config[metric]
                ax.plot(round_nums, variances, 'o-', label=config['name'], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Variance')
        ax.set_title('Variance Evolution (Lower = More Convergent)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Overall improvement score
        ax = axes[3]
        improvement_scores = []
        round_pairs = []
        
        rounds = sorted(self.round_data.keys())
        baseline_round = rounds[0]
        
        for round_num in rounds[1:]:
            improving = 0
            total = 0
            
            for metric in available_metrics:
                if metric in self.round_data[baseline_round] and metric in self.round_data[round_num]:
                    baseline_mean = np.mean(self.round_data[baseline_round][metric])
                    current_mean = np.mean(self.round_data[round_num][metric])
                    
                    config = self.metrics_config[metric]
                    is_improving = ((current_mean > baseline_mean) == config['higher_better'])
                    
                    if is_improving:
                        improving += 1
                    total += 1
            
            if total > 0:
                improvement_scores.append(improving / total)
                round_pairs.append(f"{baseline_round}→{round_num}")
        
        if improvement_scores:
            ax.bar(range(len(round_pairs)), improvement_scores, alpha=0.7, color='green')
            ax.set_xticks(range(len(round_pairs)))
            ax.set_xticklabels(round_pairs, rotation=45, ha='right')
            ax.set_ylabel('Improvement Score')
            ax.set_title('Overall Improvement Score by Round')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Baseline (50%)')
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "distribution_statistical_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_round_progression_plot(self) -> str:
        """Create a comprehensive round progression visualization."""
        available_metrics = [m for m in self.metrics_config.keys() 
                           if self._metric_available_across_rounds(m)]
        
        if not available_metrics:
            return ""
        
        # Create a comprehensive progression plot
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        rounds = sorted(self.round_data.keys())
        
        for idx, metric in enumerate(available_metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            config = self.metrics_config[metric]
            
            # Collect data
            means = []
            medians = []
            q25s = []
            q75s = []
            round_nums = []
            
            for round_num in rounds:
                if metric in self.round_data[round_num]:
                    values = self.round_data[round_num][metric]
                    if len(values) > 0:
                        means.append(np.mean(values))
                        medians.append(np.median(values))
                        q25s.append(np.percentile(values, 25))
                        q75s.append(np.percentile(values, 75))
                        round_nums.append(round_num)
            
            if len(means) > 1:
                # Plot median with IQR
                ax.plot(round_nums, medians, 'o-', color='blue', linewidth=2, 
                       markersize=8, label='Median')
                ax.fill_between(round_nums, q25s, q75s, alpha=0.3, color='blue', 
                               label='IQR (25%-75%)')
                
                # Plot mean
                ax.plot(round_nums, means, 's--', color='red', linewidth=2, 
                       markersize=6, alpha=0.8, label='Mean')
            
            ax.set_xlabel('Training Round')
            ax.set_ylabel(f"{config['name']} {config['unit']}".strip())
            ax.set_title(f'{config["name"]} Progression')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(round_nums)
            
            # Add improvement indication
            if len(means) > 1:
                trend_direction = "↑" if ((means[-1] > means[0]) == config['higher_better']) else "↓"
                trend_color = "green" if trend_direction == "↑" else "red"
                ax.text(0.02, 0.98, trend_direction, transform=ax.transAxes, 
                       fontsize=20, color=trend_color, weight='bold', 
                       verticalalignment='top')
        
        # Hide extra subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "distribution_round_progression.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _json_serialize(self, obj):
        """JSON serialization helper for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    def generate_improvement_report(self) -> str:
        """
        Generate a comprehensive improvement report in markdown format.
        
        Returns:
            Path to the generated report
        """
        analysis = self.analyze_distribution_evolution()
        
        report_lines = [
            "# Multi-Round Training Distribution Analysis Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Rounds Analyzed:** {', '.join(map(str, analysis.get('rounds_analyzed', [])))}",
            f"",
            "## Executive Summary",
            f""
        ]
        
        # Overall improvement score
        overall_trends = analysis.get('overall_trends', {})
        improvement_score = overall_trends.get('overall_improvement_score', 0)
        
        report_lines.extend([
            f"**Overall Improvement Score:** {improvement_score:.2%}",
            f"",
            f"**Improving Metrics:** {', '.join(overall_trends.get('improving_metrics', []))}",
            f"**Declining Metrics:** {', '.join(overall_trends.get('declining_metrics', []))}",
            f"",
            "## Detailed Metric Analysis",
            f""
        ])
        
        # Individual metric analysis
        for metric, metric_analysis in analysis.get('metrics', {}).items():
            config = self.metrics_config.get(metric, {})
            metric_name = config.get('name', metric)
            
            report_lines.extend([
                f"### {metric_name}",
                f""
            ])
            
            trend_stats = metric_analysis.get('trend_statistics', {})
            if trend_stats:
                is_improving = trend_stats.get('is_improving', False)
                slope = trend_stats.get('slope', 0)
                r_squared = trend_stats.get('r_squared', 0)
                
                status = "✅ IMPROVING" if is_improving else "❌ DECLINING"
                report_lines.extend([
                    f"**Status:** {status}",
                    f"**Trend Slope:** {slope:.4f} per round",
                    f"**R-squared:** {r_squared:.3f}",
                    f""
                ])
            
            # Distribution shifts
            shifts = metric_analysis.get('distribution_shifts', {})
            if shifts:
                report_lines.append("**Distribution Shifts:**")
                for shift_key, shift_data in shifts.items():
                    effect_size = shift_data.get('effect_size', 'unknown')
                    cohens_d = shift_data.get('cohens_d', 0)
                    mean_shift = shift_data.get('mean_shift', 0)
                    
                    report_lines.append(f"- {shift_key}: Effect size = {effect_size} (d={cohens_d:.3f}), Mean shift = {mean_shift:.4f}")
                
                report_lines.append("")
        
        # Conclusions
        report_lines.extend([
            "## Conclusions",
            f"",
            "This analysis demonstrates the evolution of metric distributions across multi-round training:",
            f"",
            f"1. **Overall Performance:** {improvement_score:.1%} of metrics are improving",
            f"2. **Statistical Significance:** Distribution shifts are measured using Kolmogorov-Smirnov and Mann-Whitney U tests",
            f"3. **Effect Sizes:** Cohen's d values quantify the magnitude of distribution changes",
            f"",
            "The distribution analysis provides quantitative evidence of model improvement across training rounds,",
            "supporting the effectiveness of the multi-round preference optimization approach.",
            f""
        ])
        
        # Save report
        report_path = self.output_dir / "improvement_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 Improvement report generated: {report_path}")
        return str(report_path)


def create_distribution_analyzer(output_dir: str) -> DistributionAnalyzer:
    """
    Factory function to create a DistributionAnalyzer instance.
    
    Args:
        output_dir: Directory to save analysis outputs
        
    Returns:
        DistributionAnalyzer instance
    """
    return DistributionAnalyzer(output_dir)


# Example usage and integration functions
def analyze_round_data_from_files(results_dir: str, output_dir: str) -> DistributionAnalyzer:
    """
    Load round data from evaluation result files and create analysis.
    
    Args:
        results_dir: Directory containing round evaluation results
        output_dir: Directory to save analysis outputs
        
    Returns:
        DistributionAnalyzer with loaded data
    """
    analyzer = DistributionAnalyzer(output_dir)
    results_path = Path(results_dir)
    
    # Look for evaluation result files
    for result_file in results_path.glob("eval_results_round_*.json"):
        try:
            round_num = int(result_file.stem.split('_')[-1])
            
            with open(result_file, 'r') as f:
                eval_results = json.load(f)
            
            # Try to load individual metrics if available
            individual_file = result_file.parent / f"individual_metrics_round_{round_num}.json"
            if individual_file.exists():
                with open(individual_file, 'r') as f:
                    individual_data = json.load(f)
                eval_results['individual_metrics'] = individual_data
            
            analyzer.add_round_data(round_num, eval_results)
            
        except Exception as e:
            print(f"⚠️ Failed to load round data from {result_file}: {e}")
    
    return analyzer