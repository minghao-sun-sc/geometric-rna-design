#!/usr/bin/env python3
"""Aggregate eval_summary.json across all phase-2 trainings and produce
manuscript-ready tables and figures.

Outputs:
  - scripts/phase2_results.json         : combined results table (one row per tag)
  - scripts/phase2_results.tex          : LaTeX table of loss-ablation + β-sweep
  - scripts/phase2_beta_recovery.pdf    : the β-recovery trade-off figure
  - scripts/phase2_loss_ablation.pdf    : the DPO/IPO/KTO/Pareto-DPO ablation bar chart

Run after `auto_eval_watcher.sh` reports ALL_SETTLED.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")

TAGS = [
    "thermo_surplus_m25",
    "beta_001",
    "beta_005",
    "ipo_b012",
    "kto_b012",
    "pareto_dpo_b012",
    "pareto_stage2_b012",
]

# Original-paper β values (from manuscript Table 4) so the β-recovery curve has 5 points
EXISTING_BETA = [
    {"tag": "beta_012_paper", "beta": 0.12, "recovery": 0.50, "scMCC": 0.71, "MFE": -35.83, "RMSD": 10.40},
    {"tag": "beta_05_paper",  "beta": 0.5,  "recovery": 0.47, "scMCC": 0.57, "MFE": -41.05, "RMSD": 11.96},
    {"tag": "beta_10_paper",  "beta": 1.0,  "recovery": 0.49, "scMCC": 0.59, "MFE": -41.86, "RMSD": 13.90},
]


def load_eval_summary(tag: str) -> dict | None:
    p = ROOT / f"runs/phase2/eval/{tag}/eval_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def main():
    rows = []
    for tag in TAGS:
        s = load_eval_summary(tag)
        if s is None:
            print(f"[skip] {tag}: no eval_summary.json yet")
            continue
        rows.append({"tag": tag, **s})

    out_dir = ROOT / "scripts"
    out_json = out_dir / "phase2_results.json"
    with out_json.open("w") as f:
        json.dump({"phase2": rows, "existing_beta": EXISTING_BETA}, f, indent=2)
    print(f"results -> {out_json}")

    # ---- LaTeX ablation table ----
    if rows:
        tex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Phase-2 loss ablation and \texttt{β}-sweep results on DAS test (T=0.1, $n=8$ samples per backbone). All runs use the same pair set, $\beta=0.12$ unless noted, and $\lambda_{\text{SFT}}=0.10$.}",
            r"\label{tab:phase2_results}",
            r"\small",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Run} & \textbf{Recovery} $\uparrow$ & \textbf{scMCC} $\uparrow$ & \textbf{MFE} $\downarrow$ & \textbf{RMSD} $\downarrow$ & \textbf{TM} $\uparrow$ & \textbf{GC} \\",
            r"\midrule",
        ]
        def fmt(v, dec=3):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
            return f"{v:.{dec}f}"
        for r in rows:
            tag_safe = r['tag'].replace('_', '\\_')
            tex_lines.append(
                f"{tag_safe} & "
                f"{fmt(r.get('recovery'))} & "
                f"{fmt(r.get('scMCC'))} & "
                f"{fmt(r.get('MFE'),2)} & "
                f"{fmt(r.get('RMSD'),2)} & "
                f"{fmt(r.get('TM'))} & "
                f"{fmt(r.get('GC'))} \\\\"
            )
        tex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        out_tex = out_dir / "phase2_results.tex"
        out_tex.write_text("\n".join(tex_lines) + "\n")
        print(f"latex table -> {out_tex}")

    # ---- β-recovery figure ----
    beta_rows = []
    for r in rows:
        if r["tag"].startswith("beta_") and r["tag"] != "beta_001_paper":
            tag = r["tag"]
            # extract the β value from the tag suffix
            tail = tag.replace("beta_", "")
            if tail in {"001": 0.01, "005": 0.05, "012": 0.12, "05": 0.5, "10": 1.0} or True:
                map_ = {"001": 0.01, "005": 0.05, "012": 0.12, "05": 0.5, "10": 1.0}
                beta_val = map_.get(tail.split("_")[0])
                if beta_val is not None:
                    beta_rows.append({"beta": beta_val, **r})
    # Add existing-paper values
    for r in EXISTING_BETA:
        beta_rows.append(r)
    if beta_rows:
        beta_rows.sort(key=lambda x: x["beta"])
        betas = np.array([r["beta"] for r in beta_rows])
        rec = np.array([r.get("recovery", np.nan) for r in beta_rows], dtype=float)
        scmcc = np.array([r.get("scMCC", np.nan) for r in beta_rows], dtype=float)
        mfe = np.array([r.get("MFE", np.nan) for r in beta_rows], dtype=float)
        rmsd = np.array([r.get("RMSD", np.nan) for r in beta_rows], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, y, ylabel, lower in zip(
            axes.flatten(), [rec, scmcc, mfe, rmsd],
            ["Recovery", "scMCC", "MFE (kcal/mol)", "scRMSD (Å)"],
            [True, False, False, True],
        ):
            ax.plot(betas, y, "o-", color="steelblue")
            ax.set_xscale("log")
            ax.set_xlabel("DPO β")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        fig.suptitle("β-recovery trade-off", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_pdf = out_dir / "phase2_beta_recovery.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"β-recovery figure -> {out_pdf}")

    # ---- loss-ablation bar chart ----
    abl_tags = ["thermo_surplus_m25", "ipo_b012", "kto_b012", "pareto_dpo_b012", "pareto_stage2_b012"]
    abl_rows = [r for r in rows if r["tag"] in abl_tags]
    if abl_rows:
        names = [r["tag"].replace("_b012", "").replace("_m25", "") for r in abl_rows]
        scmcc_vals = [r.get("scMCC", np.nan) for r in abl_rows]
        mfe_vals = [r.get("MFE", np.nan) for r in abl_rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(names, scmcc_vals, color="seagreen")
        axes[0].set_ylabel("scMCC ↑"); axes[0].set_title("Loss ablation: scMCC")
        axes[0].tick_params(axis='x', rotation=20)
        axes[1].bar(names, mfe_vals, color="firebrick")
        axes[1].set_ylabel("MFE (kcal/mol) ↓"); axes[1].set_title("Loss ablation: MFE")
        axes[1].tick_params(axis='x', rotation=20)
        fig.tight_layout()
        out_pdf = out_dir / "phase2_loss_ablation.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"loss-ablation figure -> {out_pdf}")


if __name__ == "__main__":
    main()
