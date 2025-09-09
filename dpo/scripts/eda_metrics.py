# dpo/scripts/eda_metrics.py
import os, argparse, json, math, csv
from statistics import mean, median

def _load_pairs_jsonl(p: str):
    rows = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def _get_metric(d: dict, side: str, key: str):
    m = d.get(f"{side}_metrics", {})
    return m.get(key, None)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _quantiles(xs, qs=(0.05,0.25,0.5,0.75,0.95)):
    if not xs: return {}
    xs_sorted = sorted(xs)
    out = {}
    for q in qs:
        idx = max(0, min(len(xs_sorted)-1, int(round(q*(len(xs_sorted)-1)))))
        out[str(q)] = xs_sorted[idx]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True, help="clean JSONL (winner/loser + *_metrics)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = _load_pairs_jsonl(args.pairs_in)

    keys = ["mfe","plddt","rmsd"]
    stats = {}
    csv_rows = []
    for k in keys:
        w_vals = [_safe_float(_get_metric(r, "winner", k)) for r in rows]
        l_vals = [_safe_float(_get_metric(r, "loser",  k)) for r in rows]
        w_vals = [x for x in w_vals if x is not None]
        l_vals = [x for x in l_vals if x is not None]
        d_vals = []
        for r in rows:
            w = _safe_float(_get_metric(r, "winner", k))
            l = _safe_float(_get_metric(r, "loser",  k))
            if w is not None and l is not None:
                d_vals.append(w - l)  # positive = winner better (for MFE, more negative better, so interpret accordingly)

        stats[k] = {
            "winner": {"n": len(w_vals), "mean": mean(w_vals) if w_vals else None, "median": median(w_vals) if w_vals else None, "q": _quantiles(w_vals)},
            "loser":  {"n": len(l_vals), "mean": mean(l_vals) if l_vals else None, "median": median(l_vals) if l_vals else None, "q": _quantiles(l_vals)},
            "delta":  {"n": len(d_vals), "mean": mean(d_vals) if d_vals else None, "median": median(d_vals) if d_vals else None, "q": _quantiles(d_vals)},
        }
        # CSV export
        for r in rows:
            w = _safe_float(_get_metric(r, "winner", k))
            l = _safe_float(_get_metric(r, "loser",  k))
            if w is not None or l is not None:
                csv_rows.append({"metric": k, "winner": w if w is not None else "", "loser": l if l is not None else "", "delta": (w-l) if (w is not None and l is not None) else ""})

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(args.out_dir, "metrics_values.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["metric","winner","loser","delta"])
        wtr.writeheader(); wtr.writerows(csv_rows)

    print("=== EDA metrics ===")
    for k, v in stats.items():
        print(f"{k}: winner_n={v['winner']['n']}, loser_n={v['loser']['n']}, delta_n={v['delta']['n']}")
    print(f"Wrote {os.path.join(args.out_dir, 'metrics_summary.json')} and metrics_values.csv")

if __name__ == "__main__":
    main()
