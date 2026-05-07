#!/usr/bin/env python3
"""Build the β-recovery trade-off figure from phase-2 evaluation outputs.

Expects each β run to have written a JSON eval summary under:
    runs/phase2/eval/beta_<tag>/eval_summary.json
with at least the keys: recovery, scMCC, MFE, RMSD, GC, diversity.

Output: scripts/beta_recovery_figure.pdf — a 2x2 figure with β on the x-axis.

This script tolerates missing runs and just plots what's available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".")

# β values and their tag suffixes. The tag is the β value with the decimal point removed
# (matches dpo/configs/experiments_phase2/beta_<tag>.yaml).
BETA_VALUES = [(0.01, "001"), (0.05, "005"), (0.12, "012"), (0.5, "05"), (1.0, "10")]


def load_summary(tag: str) -> dict | None:
    """Load eval summary for a single β run; return None if missing."""
    p = PROJECT_ROOT / f"runs/phase2/eval/beta_{tag}/eval_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def main():
    rows = []
    for beta, tag in BETA_VALUES:
        s = load_summary(tag)
        if s is None:
            print(f"[skip] beta={beta} (no eval_summary.json at runs/phase2/eval/beta_{tag}/)")
            continue
        rows.append((beta, s))
    if not rows:
        print("No β-sweep eval summaries found. Run the sweep first:")
        print("  bash scripts/run_phase2_sweep.sh beta_sweep")
        print("Then evaluate each checkpoint and write per-run eval_summary.json.")
        return

    betas = np.asarray([r[0] for r in rows])
    metrics = {
        "recovery":  np.asarray([r[1].get("recovery", np.nan) for r in rows]),
        "scMCC":     np.asarray([r[1].get("scMCC", np.nan)    for r in rows]),
        "MFE":       np.asarray([r[1].get("MFE", np.nan)      for r in rows]),
        "RMSD":      np.asarray([r[1].get("RMSD", np.nan)     for r in rows]),
        "diversity": np.asarray([r[1].get("diversity", np.nan) for r in rows]),
        "GC":        np.asarray([r[1].get("GC", np.nan)        for r in rows]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_specs = [
        (axes[0, 0], "recovery",   "↓ Recovery (sequence identity)", True),
        (axes[0, 1], "scMCC",      "↑ scMCC (2D self-consistency)",  False),
        (axes[1, 0], "MFE",        "↓ MFE (kcal/mol, lower=better)", False),
        (axes[1, 1], "RMSD",       "↓ scRMSD (Å)",                   False),
    ]
    for ax, key, title, _ in plot_specs:
        y = metrics[key]
        ax.plot(betas, y, "o-", color="steelblue")
        for x, v in zip(betas, y):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}" if abs(v) < 100 else f"{v:.2f}",
                            (x, v), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("DPO β (log scale)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("β-recovery trade-off: how DPO temperature trades sequence recovery for structure/thermodynamics", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PROJECT_ROOT / "scripts/beta_recovery_figure.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"figure -> {out}")

    # also write a compact JSON summary for the manuscript
    summary = {
        "betas": betas.tolist(),
        "metrics": {k: v.tolist() for k, v in metrics.items()},
    }
    out_json = PROJECT_ROOT / "scripts/beta_recovery_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_json}")


if __name__ == "__main__":
    main()
