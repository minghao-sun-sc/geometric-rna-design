"""Plot Stage-2 Pareto front from corner-w + centroid evals.

After running scripts/eval_pareto_front.sh at w in {(1,0,0), (0,1,0), (0,0,1), centroid},
this script reads the 4 eval_summary.json files and plots three 2D projections of the
achievable Pareto front: (scMCC vs sc_rmsd), (scMCC vs vienna_pS0), (sc_rmsd vs vienna_pS0).

Optional overlay: thermo-surplus, Stage-1 (Pareto-DPO) for context.

Usage:
    python scripts/plot_stage2_pareto_front.py \\
        --centroid runs/eval/pareto_stage2_b012/eval_summary.json \\
        --w_rmsd   runs/eval/pareto_stage2_b012_w_rmsd/eval_summary.json \\
        --w_plddt  runs/eval/pareto_stage2_b012_w_plddt/eval_summary.json \\
        --w_mfe    runs/eval/pareto_stage2_b012_w_mfe/eval_summary.json \\
        --overlay  runs/eval/thermo_surplus_m25/eval_summary.json \\
                   runs/eval/pareto_dpo_b012/eval_summary.json \\
        --out manuscript/ribopo_nips2026/figs/stage2_pareto_front.pdf
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(p):
    return json.loads(Path(p).read_text()) if p else None


def _key(d, *keys, default=float("nan")):
    """Pick the first present numeric key."""
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroid", required=True)
    ap.add_argument("--w_rmsd",   required=True, help="eval at w=(1,0,0) — RMSD-only preference")
    ap.add_argument("--w_plddt",  required=True, help="eval at w=(0,1,0) — pLDDT-only preference")
    ap.add_argument("--w_mfe",    required=True, help="eval at w=(0,0,1) — MFE-only preference")
    ap.add_argument("--overlay", nargs="*", default=[], help="extra eval_summary.json files to plot as context")
    ap.add_argument("--overlay_labels", nargs="*", default=[], help="labels for overlay points (matched by index)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pts = [
        ("centroid",          _load(args.centroid),  "C0", "o"),
        ("w=(1,0,0) RMSD",    _load(args.w_rmsd),    "C1", "s"),
        ("w=(0,1,0) pLDDT",   _load(args.w_plddt),   "C2", "^"),
        ("w=(0,0,1) MFE",     _load(args.w_mfe),     "C3", "v"),
    ]
    overlays = []
    labels = list(args.overlay_labels) + [Path(p).parent.name for p in args.overlay[len(args.overlay_labels):]]
    for p, lab in zip(args.overlay, labels):
        overlays.append((lab, _load(p), "0.55", "x"))

    # Extract metrics from each
    def m(d):
        return {
            "scMCC":   _key(d, "scMCC", "sc_eternafold"),
            "sc_rmsd": _key(d, "RMSD",  "sc_rmsd"),
            "vienna_pS0": _key(d, "vienna_pS0"),
            "pLDDT":    _key(d, "pLDDT", "sc_plddt"),
        }
    pts_m   = [(lab, m(d), c, mk) for lab, d, c, mk in pts if d is not None]
    over_m  = [(lab, m(d), c, mk) for lab, d, c, mk in overlays if d is not None]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("scMCC", "sc_rmsd", "EternaFold scMCC ↑", "RhoFold sc_rmsd (Å) ↓", False, True),
        ("scMCC", "vienna_pS0", "EternaFold scMCC ↑", "Vienna P(S₀) ↑", False, False),
        ("sc_rmsd", "vienna_pS0", "RhoFold sc_rmsd (Å) ↓", "Vienna P(S₀) ↑", True, False),
    ]
    for ax, (xk, yk, xl, yl, x_inv, y_inv) in zip(axes, panels):
        for lab, d, color, marker in pts_m:
            ax.scatter(d[xk], d[yk], color=color, marker=marker, s=110, label=lab, zorder=3,
                       edgecolors="black", linewidths=0.6)
        for lab, d, color, marker in over_m:
            ax.scatter(d[xk], d[yk], color=color, marker=marker, s=70, label=lab,
                       alpha=0.7, zorder=2)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if x_inv: ax.invert_xaxis()
        if y_inv: ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    # Legend on right of last subplot
    handles, plabels = axes[0].get_legend_handles_labels()
    fig.legend(handles, plabels, loc="upper center", ncol=min(7, len(plabels)),
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=9)
    fig.suptitle("Stage-2 FiLM-conditioned Pareto front", y=1.12, fontsize=12)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
