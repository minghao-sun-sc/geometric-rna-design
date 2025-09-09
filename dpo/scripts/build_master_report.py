# dpo/scripts/build_master_report.py
import os, json, csv, argparse, math, random
from statistics import mean, median
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --------- small IO helpers ---------

def load_metrics_summary(dir_path: str) -> dict:
    path = os.path.join(dir_path, "metrics_summary.json")
    with open(path) as f:
        return json.load(f)

def load_metrics_values(dir_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns: { metric: {"winner": [...], "loser": [...], "delta": [...]} }
    from metrics_values.csv produced by eda_metrics.py
    """
    path = os.path.join(dir_path, "metrics_values.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["metric"]
            out.setdefault(m, {"winner": [], "loser": [], "delta": []})
            for k in ["winner", "loser", "delta"]:
                s = row.get(k, "")
                if s == "" or s is None:
                    continue
                try:
                    out[m][k].append(float(s))
                except:
                    pass
    return out

def load_coverage_csv(coverage_dir: str, split: str) -> List[Tuple[str,int,int]]:
    """
    Reads {split}_coverage.csv : backbone_id, global_index, clean_pairs
    """
    path = os.path.join(coverage_dir, f"{split}_coverage.csv")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["backbone_id"], int(row["global_index"]), int(row["clean_pairs"])))
    return rows

def load_length_hists(lengths_dir: str, split: str):
    """
    Reads {split}_graph_len_hist.csv and {split}_seq_len_hist.csv if present
    """
    gpath = os.path.join(lengths_dir, f"{split}_graph_len_hist.csv")
    spath = os.path.join(lengths_dir, f"{split}_seq_len_hist.csv")
    gh, sh = {}, {}
    if os.path.exists(gpath):
        with open(gpath) as f:
            r = csv.DictReader(f)
            for row in r:
                gh[int(row["Lg"])] = int(row["count"])
    if os.path.exists(spath):
        with open(spath) as f:
            r = csv.DictReader(f)
            for row in r:
                sh[int(row["Lseq"])] = int(row["count"])
    return gh, sh

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# --------- plotting helpers ---------

def hist_multi(ax, series_dict: Dict[str, List[float]], title: str, xlabel: str):
    """
    series_dict: {"train": [...], "val": [...], "test": [...]}
    """
    # Choose common bins across splits for comparability
    all_vals = []
    for v in series_dict.values():
        all_vals.extend(v)
    if not all_vals:
        ax.set_title(title + " (no data)")
        return
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        lo -= 1.0; hi += 1.0
    bins = 60
    for split, vals in series_dict.items():
        if not vals: continue
        ax.hist(vals, bins=bins, range=(lo, hi), density=True, alpha=0.5, label=split)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()

def bar_means(ax, means: Dict[str, float], title: str, ylabel: str):
    splits = ["train","val","test"]
    xs = [i for i,_ in enumerate(splits)]
    ys = [means.get(s, float("nan")) for s in splits]
    ax.bar(xs, ys)
    ax.set_xticks(xs); ax.set_xticklabels(splits)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# --------- narrative helpers ---------

def direction_note(metric: str) -> str:
    # explain sign conventions once
    if metric == "mfe":
        return "More negative MFE (lower) is better (thermostability). Thus negative Δ = winner better."
    if metric == "plddt":
        return "Higher pLDDT is better (confidence). Thus positive Δ = winner better."
    if metric == "rmsd":
        return "Lower RMSD is better (closer to native). Thus negative Δ = winner better."
    return ""

def summarize_stat_block(msum: dict) -> Dict[str, Dict[str, float]]:
    out = {}
    for metric in ["mfe","plddt","rmsd"]:
        if metric not in msum: continue
        block = msum[metric]
        out[metric] = {
            "winner_mean": block["winner"]["mean"],
            "winner_median": block["winner"]["median"],
            "loser_mean": block["loser"]["mean"],
            "loser_median": block["loser"]["median"],
            "delta_mean": block["delta"]["mean"],
            "delta_median": block["delta"]["median"],
        }
    return out

def write_table_md(f, title: str, rowdict: Dict[str, Dict[str, float]]):
    f.write(f"\n### {title}\n\n")
    f.write("| metric | winner_mean | winner_median | loser_mean | loser_median | delta_mean | delta_median |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for metric in ["mfe","plddt","rmsd"]:
        if metric not in rowdict: continue
        r = rowdict[metric]
        fmt = lambda x: "NA" if x is None else f"{x:.4g}" if isinstance(x,float) else str(x)
        f.write(f"| {metric} | {fmt(r['winner_mean'])} | {fmt(r['winner_median'])} | {fmt(r['loser_mean'])} | {fmt(r['loser_median'])} | {fmt(r['delta_mean'])} | {fmt(r['delta_median'])} |\n")

# --------- main pipeline ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help=".../eda/train")
    ap.add_argument("--val_dir",   required=True, help=".../eda/val")
    ap.add_argument("--test_dir",  required=True, help=".../eda/test")
    ap.add_argument("--coverage_dir", default="", help="optional: .../eda/coverage")
    ap.add_argument("--lengths_dir",  default="", help="optional: .../eda/lengths")
    ap.add_argument("--out_dir", required=True, help="where to place report + images")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    img_dir = os.path.join(args.out_dir, "img")
    ensure_dir(img_dir)

    # Load summaries & raw values
    msum = {
        "train": load_metrics_summary(args.train_dir),
        "val":   load_metrics_summary(args.val_dir),
        "test":  load_metrics_summary(args.test_dir),
    }
    mvals = {
        "train": load_metrics_values(args.train_dir),
        "val":   load_metrics_values(args.val_dir),
        "test":  load_metrics_values(args.test_dir),
    }

    # Figures: delta histograms & mean-delta bars per metric
    for metric in ["mfe","plddt","rmsd"]:
        series = {
            split: (mvals[split].get(metric, {}).get("delta", []) if metric in mvals[split] else [])
            for split in ["train","val","test"]
        }
        fig, ax = plt.subplots(figsize=(7,4.2))
        hist_multi(ax, series, f"{metric} Δ (winner - loser)", "delta")
        fig.tight_layout()
        fig_path = os.path.join(img_dir, f"{metric}_delta_hist.png")
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

        # mean-delta bars
        delta_means = {
            split: (msum[split][metric]["delta"]["mean"] if metric in msum[split] else float("nan"))
            for split in ["train","val","test"]
        }
        fig, ax = plt.subplots(figsize=(5.5,3.6))
        bar_means(ax, delta_means, f"{metric} mean Δ by split", "mean delta")
        fig.tight_layout()
        fig_path = os.path.join(img_dir, f"{metric}_delta_means.png")
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

    # Optional: coverage histogram (clean pairs per backbone)
    cov_imgs = {}
    if args.coverage_dir:
        for split in ["train","val","test"]:
            rows = load_coverage_csv(args.coverage_dir, split)
            if rows:
                counts = [c for (_,_,c) in rows]
                fig, ax = plt.subplots(figsize=(6,3.8))
                ax.hist(counts, bins=40, density=False)
                ax.set_title(f"Clean pairs per backbone — {split}")
                ax.set_xlabel("clean pairs per backbone")
                ax.set_ylabel("backbones")
                fig.tight_layout()
                p = os.path.join(img_dir, f"coverage_{split}.png")
                fig.savefig(p, dpi=160)
                plt.close(fig)
                cov_imgs[split] = p

    # Optional: length histograms (graph length vs clean seq length)
    len_imgs = {}
    if args.lengths_dir:
        for split in ["train","val","test"]:
            gh, sh = load_length_hists(args.lengths_dir, split)
            if gh or sh:
                # graph length hist
                if gh:
                    fig, ax = plt.subplots(figsize=(6.4,3.2))
                    xs = sorted(gh.keys()); ys = [gh[x] for x in xs]
                    ax.bar(range(len(xs)), ys)
                    ax.set_xticks(range(0,len(xs), max(1,len(xs)//10)))
                    ax.set_xticklabels([xs[i] for i in range(0,len(xs), max(1,len(xs)//10))], rotation=45)
                    ax.set_title(f"Graph length histogram — {split}")
                    ax.set_ylabel("count")
                    fig.tight_layout()
                    p = os.path.join(img_dir, f"length_graph_{split}.png")
                    fig.savefig(p, dpi=160)
                    plt.close(fig)
                    len_imgs[f"graph_{split}"] = p
                # sequence length hist (clean pairs)
                if sh:
                    fig, ax = plt.subplots(figsize=(6.4,3.2))
                    xs = sorted(sh.keys()); ys = [sh[x] for x in xs]
                    ax.bar(range(len(xs)), ys)
                    ax.set_xticks(range(0,len(xs), max(1,len(xs)//10)))
                    ax.set_xticklabels([xs[i] for i in range(0,len(xs), max(1,len(xs)//10))], rotation=45)
                    ax.set_title(f"Clean sequence length histogram — {split}")
                    ax.set_ylabel("count")
                    fig.tight_layout()
                    p = os.path.join(img_dir, f"length_seq_{split}.png")
                    fig.savefig(p, dpi=160)
                    plt.close(fig)
                    len_imgs[f"seq_{split}"] = p

    # Build report.md
    report_path = os.path.join(args.out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# DPO-RNA — Master EDA Report\n\n")
        f.write("This report aggregates per-split EDA for **MFE**, **pLDDT**, and **RMSD**.\n\n")
        f.write("**Sign conventions:**\n\n")
        f.write("- MFE: more negative is better → negative Δ (winner - loser) means winner better.\n")
        f.write("- pLDDT: higher is better → positive Δ means winner better.\n")
        f.write("- RMSD: lower is better → negative Δ means winner better.\n\n")

        # Tables
        for split in ["train","val","test"]:
            f.write(f"\n## {split.capitalize()} summary\n")
            block = summarize_stat_block(msum[split])
            write_table_md(f, f"{split} — summary stats", block)

        # Delta plots
        f.write("\n## Delta distributions\n")
        for metric in ["mfe","plddt","rmsd"]:
            f.write(f"\n### {metric} Δ (winner - loser)\n")
            f.write(f"{direction_note(metric)}\n\n")
            f.write(f"![{metric} delta hist](img/{metric}_delta_hist.png)\n\n")
            f.write(f"![{metric} mean delta](img/{metric}_delta_means.png)\n\n")

        # Optional sections
        if cov_imgs:
            f.write("\n## Coverage (clean pairs per backbone)\n")
            for split in ["train","val","test"]:
                if split in cov_imgs:
                    f.write(f"\n### {split}\n\n")
                    f.write(f"![coverage {split}]({os.path.relpath(cov_imgs[split], args.out_dir)})\n\n")

        if len_imgs:
            f.write("\n## Length distributions\n")
            for split in ["train","val","test"]:
                gp = len_imgs.get(f"graph_{split}")
                sp = len_imgs.get(f"seq_{split}")
                if gp:
                    f.write(f"\n### {split} — graph length\n\n![graph length {split}]({os.path.relpath(gp, args.out_dir)})\n")
                if sp:
                    f.write(f"\n### {split} — clean sequence length\n\n![seq length {split}]({os.path.relpath(sp, args.out_dir)})\n")

        # High-level interpretation section
        f.write("\n## Interpretation (automatic)\n")
        for metric in ["mfe","plddt","rmsd"]:
            dm = {s: (msum[s][metric]["delta"]["mean"] if metric in msum[s] else None) for s in ["train","val","test"]}
            mm = {s: (msum[s][metric]["delta"]["median"] if metric in msum[s] else None) for s in ["train","val","test"]}
            f.write(f"\n**{metric}** — mean Δ (train/val/test): {dm['train']:.4g}, {dm['val']:.4g}, {dm['test']:.4g}; "
                    f"median Δ: {mm['train']:.4g}, {mm['val']:.4g}, {mm['test']:.4g}.\n")
            if metric == "mfe":
                f.write("Winners are more stable (more negative MFE) across splits.\n")
            elif metric == "plddt":
                f.write("Winners have higher pLDDT (structure confidence).\n")
            elif metric == "rmsd":
                f.write("Winners are closer to native (lower RMSD).\n")

    print(f"Wrote report -> {report_path}")

if __name__ == "__main__":
    main()
