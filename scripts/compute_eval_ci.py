"""Compute bootstrap 95% CIs from per-structure list arrays persisted by eval_full.py.

Reads `<eval_dir>/<tag>_per_structure.json` (written by the patched eval_full.py)
and writes `<eval_dir>/eval_summary_ci.json` with mean ± 95% percentile-CI for each
metric. Optionally augments the existing `eval_summary.json` in-place.

Usage:
    python scripts/compute_eval_ci.py runs/phase2/eval/grnade_base_v3/
    python scripts/compute_eval_ci.py --inplace runs/phase2/eval/*/
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean. Filters NaN."""
    arr = np.asarray([v for v in values if isinstance(v, (int, float)) and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def process_dir(d, inplace=False):
    """Process a single eval directory."""
    d = Path(d)
    per_files = list(d.glob("*_per_structure.json"))
    if not per_files:
        print(f"[skip] {d}: no *_per_structure.json found")
        return
    # Pick the most recent
    per_path = max(per_files, key=lambda p: p.stat().st_mtime)
    print(f"[{d.name}] reading {per_path.name}")
    data = json.loads(per_path.read_text())
    lists = data.get("lists", {})

    out = {"n_structures": data.get("n_structures"),
           "n_samples_per_structure": data.get("n_samples_per_structure"),
           "metrics": {}}
    for k, v in lists.items():
        if not v:
            continue
        m, lo, hi = bootstrap_ci(v)
        if m is None:
            continue
        out["metrics"][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(v)}

    out_path = d / "eval_summary_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  → {out_path}")

    if inplace:
        # Augment the existing eval_summary.json (the compact one) with per-metric CIs.
        sj = d / "eval_summary.json"
        if sj.exists():
            existing = json.loads(sj.read_text())
            for metric_key, stats in out["metrics"].items():
                # don't overwrite the mean; just add the CI fields with key suffix
                existing[f"{metric_key}_ci_lo"] = stats["ci_lo"]
                existing[f"{metric_key}_ci_hi"] = stats["ci_hi"]
            sj.write_text(json.dumps(existing, indent=2))
            print(f"  augmented {sj}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="eval directories (e.g. runs/phase2/eval/<tag>)")
    ap.add_argument("--inplace", action="store_true",
                    help="also add ci_lo/ci_hi keys to <dir>/eval_summary.json")
    args = ap.parse_args()
    for d in args.dirs:
        try:
            process_dir(d, inplace=args.inplace)
        except Exception as e:
            print(f"[error] {d}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
