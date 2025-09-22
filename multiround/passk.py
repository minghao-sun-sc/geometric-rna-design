# multiround/passk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ------------------------------
# Success flag construction
# ------------------------------

@dataclass(frozen=True)
class Rule:
    """
    A threshold rule on a metric array.
    op: one of {">=", ">", "<=", "<"}
    threshold: float
    name: metric name in the per-sample dict (e.g., "tm", "ed_per_nt", "pS0")
    """
    name: str
    op: str
    threshold: float

def _apply_rule(x: np.ndarray, op: str, thr: float) -> np.ndarray:
    if op == ">=": return x >= thr
    if op == ">":  return x >  thr
    if op == "<=": return x <= thr
    if op == "<":  return x <  thr
    raise ValueError(f"Unsupported op: {op}")

def combine_success_flags(flags: List[np.ndarray], mode: str = "all") -> np.ndarray:
    """
    Combine multiple boolean masks (same length) with AND/OR.
    """
    if len(flags) == 0:
        raise ValueError("flags list is empty")
    out = flags[0].astype(bool)
    for f in flags[1:]:
        if mode == "all":
            out = out & f
        elif mode == "any":
            out = out | f
        else:
            raise ValueError("mode must be 'all' or 'any'")
    return out

def success_flags_from_metric_dict(
    metric_dict_list: List[Dict[str, np.ndarray]],
    rules: Sequence[Rule],
    mode: str = "all",
) -> List[np.ndarray]:
    """
    Convert per-item metric dicts into per-item boolean success arrays.

    metric_dict_list: list over items; each item is dict[name -> np.ndarray (n_samples,)]
    rules: list of Rule defining success criteria on individual metrics
    mode: "all" (AND) or "any" (OR) across rules
    """
    all_flags: List[np.ndarray] = []
    for d in metric_dict_list:
        cur_flags = []
        for r in rules:
            if r.name not in d:
                raise KeyError(f"Metric '{r.name}' missing in item dict keys={list(d.keys())}")
            cur_flags.append(_apply_rule(np.asarray(d[r.name]), r.op, r.threshold))
        all_flags.append(combine_success_flags(cur_flags, mode=mode))
    return all_flags

# ------------------------------
# pass@k estimators
# ------------------------------

def _ratio_choose(n: int, c: int, k: int) -> float:
    """
    Compute C(n - c, k) / C(n, k) stably when 0 <= k <= n, using product form:
      Π_{j=0..k-1} (n - c - j) / (n - j)
    If k > n: interpret as 0 (so pass=1 if c>0 else 0 in the unbiased formula).
    """
    if k <= 0:
        return 1.0
    if n <= 0:
        return 0.0
    if k > n:
        return 0.0
    num = 1.0
    for j in range(k):
        num *= (n - c - j) / (n - j)
    return float(max(0.0, min(1.0, num)))

def pass_at_k_unbiased(
    successes_list: List[np.ndarray], 
    k: int
) -> float:
    """
    HumanEval-style unbiased estimator for pass@k when you would select k samples
    uniformly without replacement from n_i samples for each item i.

    For item i with n_i samples and c_i successes:
      hat{p}_i = 1 - C(n_i - c_i, k) / C(n_i, k)      if c_i >= 1 and k <= n_i
               = 0                                     if c_i == 0
               = 1                                     if c_i >= 1 and k > n_i

    Returns mean_i hat{p}_i.
    """
    vals = []
    for flags in successes_list:
        flags = np.asarray(flags).astype(bool)
        n = int(flags.size)
        c = int(flags.sum())
        if c == 0:
            vals.append(0.0)
            continue
        if k > n:
            vals.append(1.0)
            continue
        ratio = _ratio_choose(n, c, k)
        vals.append(1.0 - ratio)
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")

def pass_at_k_topk(
    scores_list: List[np.ndarray],
    k: int,
    threshold: float,
    higher_is_better: bool = True
) -> float:
    """
    Pass@k when you rank each item by a score and keep the best k.
    
    Note: This function is designed for simple single-threshold evaluation.
    For multi-rule evaluation (e.g., TM >= 0.45 AND RMSD <= 8.0), use the 
    'topk' path in pass_at_k_from_metrics() which handles multiple Rule objects.

    Args:
        scores_list: per-item arrays of scores (one score per sample)
        k: number of top samples to consider
        threshold: success if score >= thr (for higher_is_better) or <= thr otherwise
        higher_is_better: True if larger scores are better
        
    Returns:
        float: Pass@k rate (fraction of items with at least one success in top-k)
    """
    hits = []
    for s in scores_list:
        s = np.asarray(s)
        if s.size == 0:
            hits.append(0.0)
            continue
        if higher_is_better:
            idx = np.argsort(-s)[:min(k, s.size)]
            ok = np.any(s[idx] >= threshold)
        else:
            idx = np.argsort(s)[:min(k, s.size)]
            ok = np.any(s[idx] <= threshold)
        hits.append(1.0 if ok else 0.0)
    return float(np.mean(hits)) if len(hits) > 0 else float("nan")

# ------------------------------
# Convenience: one-shot wrapper
# ------------------------------

def pass_at_k_from_metrics(
    metric_dict_list: List[Dict[str, np.ndarray]],
    rules: Sequence[Rule],
    k: int,
    combine_mode: str = "all",
    selection: str = "unbiased",
    rank_metric: Optional[str] = None,
    rank_higher_is_better: Optional[bool] = None,
) -> float:
    """
    Convenience wrapper.

    metric_dict_list: per-item dicts of per-sample arrays, e.g.
        {
          "tm":        np.ndarray(n_samples,),
          "ed_per_nt": np.ndarray(n_samples,),
          "mfe":       np.ndarray(n_samples,),
        }
    rules: e.g. [Rule("tm", ">=", 0.45), Rule("ed_per_nt", "<=", 0.05)]
    k:     pass@k
    combine_mode: "all" / "any" across the rules
    selection:
      - "unbiased" -> unbiased pass@k estimator (no ranking)
      - "topk"     -> rank by `rank_metric` and keep best k
    rank_metric: name of metric to rank by (required if selection="topk")
    rank_higher_is_better: True if larger score is better for ranking
    """
    if selection == "unbiased":
        flags = success_flags_from_metric_dict(metric_dict_list, rules, mode=combine_mode)
        return pass_at_k_unbiased(flags, k)

    elif selection == "topk":
        if rank_metric is None or rank_higher_is_better is None:
            raise ValueError("rank_metric and rank_higher_is_better are required for selection='topk'.")

        # Build a *single* rule if you want 'success' defined on the ranking metric,
        # otherwise 'rules' can include other constraints (e.g., ED_per_nt) and we'll
        # check success among top-k by *rank_metric*.
        out = []
        for d in metric_dict_list:
            # rank indices
            s = d[rank_metric]
            if rank_higher_is_better:
                idx = np.argsort(-s)[:min(k, s.size)]
            else:
                idx = np.argsort(s)[:min(k, s.size)]

            # success among top-k according to the provided rules
            local_flags = []
            for r in rules:
                local_flags.append(_apply_rule(np.asarray(d[r.name])[idx], r.op, r.threshold))
            ok = np.any(combine_success_flags(local_flags, mode=combine_mode))
            out.append(1.0 if ok else 0.0)
        return float(np.mean(out))

    else:
        raise ValueError("selection must be 'unbiased' or 'topk'.")


# ------------------------------
# High-level pass@k analysis functions for eval_full.py integration
# ------------------------------

def calculate_passk_metrics(
    metrics_by_structure: Dict[str, List[Dict[str, Any]]],
    k_values: List[int] = [1, 2, 4, 8, 16, 32, 64],
    thresholds: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate pass@k metrics for multiple k values and thresholds.
    
    Args:
        metrics_by_structure: Dict mapping structure_id -> list of sample metrics
        k_values: List of k values for pass@k analysis
        thresholds: Dict mapping metric names to threshold lists
        
    Returns:
        Dict containing pass@k results for all combinations
    """
    if thresholds is None:
        thresholds = {
            "tm_score": [0.4, 0.45, 0.5, 0.55],
            "rmsd": [8.0, 6.0, 4.0, 2.0],
            "mfe": [-10.0, -15.0, -20.0]
        }
    
    # Convert structure-based metrics to format needed by pass@k functions
    metric_dict_list = []
    structure_ids = []
    
    for struct_id, samples in metrics_by_structure.items():
        if not samples:
            continue
            
        structure_ids.append(struct_id)
        
        # Organize metrics for this structure
        struct_metrics = {}
        metric_names = set()
        for sample in samples:
            for key in sample.keys():
                if key not in ["structure_id", "sample_idx"]:
                    metric_names.add(key)
        
        # Extract arrays for each metric
        for metric_name in metric_names:
            values = []
            valid_count = 0
            for sample in samples:
                val = sample.get(metric_name, np.nan)
                # For Vienna MFE: exclude failed samples (0.0 or NaN) from analysis
                if metric_name == "vienna_mfe":
                    if not np.isnan(val) and val != 0.0:  # Valid Vienna MFE values
                        values.append(val)
                        valid_count += 1
                    # Skip failed samples entirely (don't add to values)
                else:
                    # For other metrics: exclude NaN but keep 0.0 as valid
                    if not np.isnan(val):
                        values.append(val)
                        valid_count += 1
                    # Skip NaN samples entirely
            
            # Only include metrics that have valid samples
            if valid_count > 0:
                struct_metrics[metric_name] = np.array(values)
            else:
                # No valid samples for this metric - exclude from analysis
                print(f"⚠️ Warning: No valid samples for {metric_name} in structure {struct_id}, skipping metric")
                # Don't add this metric to struct_metrics
        
        metric_dict_list.append(struct_metrics)
    
    if not metric_dict_list:
        return {"error": "No valid metrics found"}
    
    # Calculate pass@k for all combinations
    results = {
        "n_structures": len(structure_ids),
        "structure_ids": structure_ids,
        "k_values": k_values,
        "thresholds": thresholds,
        "passk_results": {}
    }
    
    # Map config threshold names to actual metric names
    metric_mapping = {
        "tm_score": "sc_tm",
        "rmsd": "sc_rmsd", 
        "mfe": "vienna_mfe"
    }
    
    for threshold_name, threshold_list in thresholds.items():
        actual_metric_name = metric_mapping.get(threshold_name, threshold_name)
        
        # Check if this metric is available and count structures with it
        structures_with_metric = [i for i, md in enumerate(metric_dict_list) if actual_metric_name in md]
        if not structures_with_metric:
            print(f"⚠️ Warning: Metric '{actual_metric_name}' not found in any structure, skipping {threshold_name}")
            continue
        
        print(f"🔍 Debug: Found {len(structures_with_metric)} structures with '{actual_metric_name}' out of {len(metric_dict_list)} total")
        print(f"   Structures with metric: {[structure_ids[i] for i in structures_with_metric[:3]]}{'...' if len(structures_with_metric) > 3 else ''}")
            
        results["passk_results"][threshold_name] = {}
        
        for threshold in threshold_list:
            # Create appropriate rule
            if threshold_name in ["tm_score"]:
                # Higher is better for TM-score
                rule = Rule(actual_metric_name, ">=", threshold)
            elif threshold_name in ["rmsd"]:
                # Lower is better for RMSD
                rule = Rule(actual_metric_name, "<=", threshold)
            elif threshold_name in ["mfe"]:
                # Lower is better for MFE (more negative)
                rule = Rule(actual_metric_name, "<=", threshold)
            else:
                # Default: higher is better
                rule = Rule(actual_metric_name, ">=", threshold)
            
            # Filter to only structures that have this metric
            filtered_metric_dict_list = [metric_dict_list[i] for i in structures_with_metric]
            
            threshold_results = {}
            for k in k_values:
                try:
                    passk_value = pass_at_k_from_metrics(
                        filtered_metric_dict_list,
                        rules=[rule],
                        k=k,
                        combine_mode="all",
                        selection="unbiased"
                    )
                    threshold_results[f"pass@{k}"] = float(passk_value)
                    print(f"✅ Pass@{k} for {threshold_name}={threshold}: {passk_value:.3f} (based on {len(filtered_metric_dict_list)} structures)")
                except Exception as e:
                    print(f"⚠️ Warning: Pass@{k} calculation failed for {threshold_name}={threshold}: {e}")
                    threshold_results[f"pass@{k}"] = float('nan')
            
            results["passk_results"][threshold_name][str(threshold)] = threshold_results
    
    return results


def plot_metric_distributions(
    individual_metrics: List[Dict[str, Any]],
    output_dir: str,
    metrics_to_plot: List[str] = ["sc_plddt", "sc_rmsd", "sc_tm", "vienna_mfe", "inf_all"],
    checkpoint_name: str = "checkpoint"
) -> None:
    """
    Create distribution plots for key metrics.
    
    Args:
        individual_metrics: List of individual sample metrics
        output_dir: Directory to save plots
        metrics_to_plot: List of metric names to plot (supports both config and actual names)
        checkpoint_name: Name for plot titles and filenames
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        plt.style.use('default')
    except Exception as e:
        print(f"⚠️ Warning: Could not set matplotlib backend: {e}")
        return
    
    # Create metric name mapping for backward compatibility with config files
    metric_name_mapping = {
        "plddt": "sc_plddt",
        "rmsd": "sc_rmsd", 
        "tm_score": "sc_tm",
        "mfe": "vienna_mfe",
        "inf_all": "inf_all",  # no change
        "gdt": "sc_gdt",
        "eternafold": "sc_eternafold",
        "diversity_3mer": "diversity_3mer",  # no change
        "clashscore_pre": "clashscore_pre",  # no change
        "lddt": "lddt",  # no change
        "mcq_abs_deg": "mcq_abs_deg",  # no change
    }
    
    # Organize data by metric
    metric_data = {}
    for metric_name in metrics_to_plot:
        # Map config metric names to actual metric names
        actual_metric_name = metric_name_mapping.get(metric_name, metric_name)
        
        values = []
        for sample in individual_metrics:
            if actual_metric_name in sample:
                val = sample[actual_metric_name]
                if val is not None and not np.isnan(val):
                    values.append(val)
        
        if values:
            # Use the original name for display purposes
            display_name = metric_name if metric_name != actual_metric_name else actual_metric_name
            metric_data[display_name] = np.array(values)
    
    if not metric_data:
        print("⚠️ Warning: No valid metric data found for plotting")
        # Debug: show what metrics are actually available
        if individual_metrics:
            available_metrics = set()
            for sample in individual_metrics[:5]:  # Check first 5 samples
                available_metrics.update(sample.keys())
            print(f"📋 Available metrics in data: {sorted(available_metrics)}")
            print(f"📋 Requested metrics: {metrics_to_plot}")
        return
    
    print(f"📊 Plotting distributions for metrics: {list(metric_data.keys())}")
    
    # Create plots
    n_metrics = len(metric_data)
    if n_metrics == 0:
        return
    
    # Determine plot layout
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_metrics == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if ncols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for metric_name, values in metric_data.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Create histogram with KDE
        try:
            ax.hist(values, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add KDE if we have enough points
            if len(values) > 5:
                from scipy import stats
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                ax.legend()
        except Exception:
            # Fallback to simple histogram
            ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Formatting
        ax.set_title(f'{metric_name}\n(n={len(values)}, μ={np.mean(values):.3f})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density' if 'kde' in locals() else 'Count')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Metric Distributions - {checkpoint_name}', fontsize=16, y=1.02)
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{checkpoint_name}_metric_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Distribution plot saved: {plot_path}")
    
    # Also save summary statistics
    stats_data = {}
    for metric_name, values in metric_data.items():
        stats_data[metric_name] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75))
        }
    
    stats_path = os.path.join(output_dir, f'{checkpoint_name}_metric_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"📊 Statistics saved: {stats_path}")


# ------------------------------
# Minimal usage example (keep or remove)
# ------------------------------
if __name__ == "__main__":
    # Two items; each has 8 samples with two metrics: TM and ED/nt
    rng = np.random.default_rng(0)
    md0 = {
        "tm":        rng.uniform(0.2, 0.9, size=8),
        "ed_per_nt": rng.uniform(0.0, 0.12, size=8),
    }
    md1 = {
        "tm":        rng.uniform(0.2, 0.9, size=8),
        "ed_per_nt": rng.uniform(0.0, 0.12, size=8),
    }
    items = [md0, md1]

    # Success rule: TM ≥ 0.45 AND ED/nt ≤ 0.05
    rules = [Rule("tm", ">=", 0.45), Rule("ed_per_nt", "<=", 0.05)]

    for k in [1, 2, 4]:
        p_unbiased = pass_at_k_from_metrics(items, rules, k, combine_mode="all", selection="unbiased")
        p_topk_tm  = pass_at_k_from_metrics(items, rules, k, combine_mode="all",
                                            selection="topk", rank_metric="tm",
                                            rank_higher_is_better=True)
        print(f"k={k}: unbiased={p_unbiased:.3f}  topk-by-TM={p_topk_tm:.3f}")
