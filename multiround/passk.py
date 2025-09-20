# multiround/passk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np

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
