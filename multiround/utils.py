# multiround/utils.py
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime


def create_round_output_dir(output_root: str, round_num: int) -> str:
    """Create output directory for a specific round."""
    round_dir = os.path.join(output_root, f"round_{round_num:02d}")
    os.makedirs(round_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'eval_results', 'plots', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(round_dir, subdir), exist_ok=True)
    
    return round_dir


def update_reference_model(policy_model: torch.nn.Module, reference_model: torch.nn.Module):
    """Update reference model by copying weights from policy model."""
    reference_model.load_state_dict(policy_model.state_dict())


def save_round_metrics(metrics: Dict, round_dir: str):
    """Save round metrics to JSON file."""
    metrics_path = os.path.join(round_dir, 'round_metrics.json')
    
    # Convert any numpy types to native Python types for JSON serialization
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        elif isinstance(v, (np.int64, np.int32)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v
    
    with open(metrics_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2)


def plot_round_progression(round_metrics: List[Dict], output_dir: str):
    """
    Plot progression of key metrics across rounds.
    
    Args:
        round_metrics: List of metrics dictionaries from each round
        output_dir: Directory to save plots
    """
    if len(round_metrics) < 2:
        return
    
    # Extract round numbers and metrics
    rounds = [r['round'] for r in round_metrics]
    
    # Primary metrics to plot
    primary_metrics = {
        'TM Score': 'tm_mean',
        'RMSD (Å)': 'rmsd_mean', 
        'MFE (kcal/mol)': 'mfe_mean'
    }
    
    # Secondary metrics to plot
    secondary_metrics = {
        'pLDDT': 'plddt_mean',
        'GDT': 'gdt_mean',
        'Diversity (3-mer)': 'diversity_3mer_mean',
        'INF (All)': 'inf_all_mean'
    }
    
    # Create progression plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot primary metrics
    for display_name, metric_key in primary_metrics.items():
        if plot_idx >= len(axes):
            break
            
        values = [r.get(metric_key, np.nan) for r in round_metrics]
        valid_data = [(r, v) for r, v in zip(rounds, values) if not np.isnan(v)]
        
        if valid_data:
            valid_rounds, valid_values = zip(*valid_data)
            axes[plot_idx].plot(valid_rounds, valid_values, 'o-', linewidth=2, markersize=8)
            axes[plot_idx].set_title(f'{display_name} Progression', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Round')
            axes[plot_idx].set_ylabel(display_name)
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Add value annotations
            for r, v in zip(valid_rounds, valid_values):
                axes[plot_idx].annotate(f'{v:.3f}', (r, v), 
                                      textcoords="offset points", xytext=(0,10), ha='center')
        
        plot_idx += 1
    
    # Plot secondary metrics
    for display_name, metric_key in secondary_metrics.items():
        if plot_idx >= len(axes):
            break
            
        values = [r.get(metric_key, np.nan) for r in round_metrics]
        valid_data = [(r, v) for r, v in zip(rounds, values) if not np.isnan(v)]
        
        if valid_data:
            valid_rounds, valid_values = zip(*valid_data)
            axes[plot_idx].plot(valid_rounds, valid_values, 's-', linewidth=2, markersize=6)
            axes[plot_idx].set_title(f'{display_name} Progression', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Round')
            axes[plot_idx].set_ylabel(display_name)
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Add value annotations
            for r, v in zip(valid_rounds, valid_values):
                axes[plot_idx].annotate(f'{v:.3f}', (r, v), 
                                      textcoords="offset points", xytext=(0,10), ha='center')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'plots', 'round_progression.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Round progression plot saved: {plot_path}")


def plot_training_curves(training_logs: List[Dict], output_dir: str):
    """
    Plot training curves (loss, learning rate, etc.) for a round.
    
    Args:
        training_logs: List of training step logs
        output_dir: Directory to save plots
    """
    if not training_logs:
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(training_logs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    if 'loss' in df.columns:
        axes[0, 0].plot(df['step'], df['loss'], label='Total Loss', alpha=0.7)
        if 'loss_dpo' in df.columns:
            axes[0, 0].plot(df['step'], df['loss_dpo'], label='DPO Loss', alpha=0.7)
        if 'loss_sft' in df.columns:
            axes[0, 0].plot(df['step'], df['loss_sft'], label='SFT Loss', alpha=0.7)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate
    if 'lr' in df.columns:
        axes[0, 1].plot(df['step'], df['lr'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Preference Accuracy
    if 'pref_acc' in df.columns:
        axes[1, 0].plot(df['step'], df['pref_acc'])
        axes[1, 0].set_title('Preference Accuracy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: KL Divergence (if available)
    if 'kl_div' in df.columns:
        axes[1, 1].plot(df['step'], df['kl_div'])
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('KL Divergence')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'plots', 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training curves saved: {plot_path}")


def create_evaluation_summary_table(round_metrics: List[Dict], output_dir: str):
    """
    Create a comprehensive evaluation summary table across all rounds.
    
    Args:
        round_metrics: List of metrics from all rounds
        output_dir: Directory to save the table
    """
    if not round_metrics:
        return
    
    # Define metrics to include in summary
    metric_columns = [
        'round', 'tm_mean', 'rmsd_mean', 'mfe_mean', 'plddt_mean', 
        'gdt_mean', 'inf_all_mean', 'clashscore_mean', 'diversity_3mer_mean'
    ]
    
    # Extract data for table
    table_data = []
    for metrics in round_metrics:
        row = {}
        for col in metric_columns:
            value = metrics.get(col, np.nan)
            if isinstance(value, (int, float)) and not np.isnan(value):
                row[col] = f"{value:.4f}"
            else:
                row[col] = "N/A"
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as formatted markdown table
    md_path = os.path.join(output_dir, 'evaluation_summary.md')
    with open(md_path, 'w') as f:
        f.write("# Multi-Round DPO Evaluation Summary\n\n")
        f.write("## Primary Metrics Across Rounds\n\n")
        f.write(df.to_markdown(index=False, tablefmt='github'))
        f.write("\n\n")
        
        # Add metric descriptions
        f.write("## Metric Descriptions\n\n")
        descriptions = {
            'tm_mean': 'Template Modeling score (0-1, higher better)',
            'rmsd_mean': 'Root Mean Square Deviation in Å (lower better)',
            'mfe_mean': 'Minimum Free Energy in kcal/mol (more negative better)',
            'plddt_mean': 'Predicted Local Distance Difference Test (0-1, higher better)',
            'gdt_mean': 'Global Distance Test (0-1, higher better)',
            'inf_all_mean': 'Interaction Network Fidelity - All contacts (0-1, higher better)',
            'clashscore_mean': 'MolProbity clash score (lower better)',
            'diversity_3mer_mean': '3-mer sequence diversity (0-1, balanced preferred)'
        }
        
        for metric, desc in descriptions.items():
            f.write(f"- **{metric}**: {desc}\n")
    
    print(f"📊 Evaluation summary saved: {csv_path}")
    print(f"📊 Markdown summary saved: {md_path}")


def load_config_with_inheritance(config_path: str) -> Dict:
    """
    Load YAML config with inheritance support.
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict: Merged configuration
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit_from' in config:
        base_path = config['inherit_from']
        # Resolve relative paths from project root (current working directory)
        if not os.path.isabs(base_path):
            # If path doesn't exist relative to config file, try from current working directory
            config_relative_path = os.path.join(os.path.dirname(config_path), base_path)
            if os.path.exists(config_relative_path):
                base_path = config_relative_path
            else:
                # Try from current working directory (project root)
                base_path = base_path
        
        # Load base config recursively
        base_config = load_config_with_inheritance(base_path)
        
        # Merge configs (current overrides base)
        merged_config = deep_merge_dicts(base_config, config)
        
        # Remove inheritance key from final config
        merged_config.pop('inherit_from', None)
        
        return merged_config
    
    return config


def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        dict: Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def format_time_delta(seconds: float) -> str:
    """Format time delta in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_best_checkpoint_by_metric(checkpoints: List[Dict], metric_name: str = 'tm_mean') -> Optional[Dict]:
    """
    Find the best checkpoint by a given metric.
    
    Args:
        checkpoints: List of checkpoint info dicts
        metric_name: Metric to optimize for
        
    Returns:
        dict or None: Best checkpoint info
    """
    if not checkpoints:
        return None
    
    best_checkpoint = None
    best_value = -float('inf')
    
    for ckpt in checkpoints:
        value = ckpt.get('eval_metrics', {}).get(metric_name, -float('inf'))
        if value > best_value:
            best_value = value
            best_checkpoint = ckpt
    
    return best_checkpoint