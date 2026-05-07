#!/usr/bin/env python3
"""Analyse the GC-controlled thermodynamic-surplus pair set vs the raw-MFE pair set.

Outputs:
  - scripts/thermo_surplus_analysis.json   : numerical summary
  - scripts/thermo_surplus_figure.pdf      : 2x2 figure suitable for the appendix
       (a) gRNAde MFE vs GC*L (regression)
       (b) Δ(MFE) vs Δ(MFE residual): which pairs are kept (consistent) vs dropped (GC-driven)
       (c) GC distributions of winners and losers
       (d) MFE residual distributions of winners and losers (kept pairs only)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".")


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    upper = seq.upper().replace("T", "U")
    gc = sum(1 for c in upper if c in {"G", "C"})
    return gc / len(upper)


def load_pairs(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    margin = "25"
    raw_path = PROJECT_ROOT / f"data/pairs_margin{margin}/by_das/clean/train.clean.jsonl"
    surplus_path = PROJECT_ROOT / f"data/pairs_thermo_surplus_margin{margin}/by_das/clean/train.clean.jsonl"
    report_path = PROJECT_ROOT / "scripts/thermo_surplus_report.json"
    out_json = PROJECT_ROOT / "scripts/thermo_surplus_analysis.json"
    out_pdf = PROJECT_ROOT / "scripts/thermo_surplus_figure.pdf"

    raw = load_pairs(raw_path)
    surplus = load_pairs(surplus_path)
    print(f"raw pairs: {len(raw)}  surplus-kept pairs: {len(surplus)}  ({len(surplus)/len(raw):.1%})")

    # Reload regression coefficients from the report.
    with report_path.open() as f:
        report = json.load(f)
    coef = report[f"margin{margin}"]["regression"]
    a, b, c = coef["a"], coef["b"], coef["c"]

    def residual(seq: str, mfe: float) -> float:
        L = len(seq)
        gc = gc_fraction(seq)
        return mfe - (a + b * L + c * (gc * L))

    def gather(records: list[dict]) -> dict[str, np.ndarray]:
        wL, lL, wMFE, lMFE, wGC, lGC, wRES, lRES = [], [], [], [], [], [], [], []
        for r in records:
            ws, ls = r["winner_seq"], r["loser_seq"]
            wm, lm = float(r["winner_metrics"]["mfe"]), float(r["loser_metrics"]["mfe"])
            wL.append(len(ws))
            lL.append(len(ls))
            wMFE.append(wm)
            lMFE.append(lm)
            wGC.append(gc_fraction(ws))
            lGC.append(gc_fraction(ls))
            wRES.append(residual(ws, wm))
            lRES.append(residual(ls, lm))
        return {k: np.asarray(v) for k, v in locals().items() if k.startswith(("wL", "lL", "wMFE", "lMFE", "wGC", "lGC", "wRES", "lRES"))}

    raw_d = gather(raw)
    surplus_d = gather(surplus)

    # Statistics
    delta_mfe_raw = raw_d["wMFE"] - raw_d["lMFE"]
    delta_res_raw = raw_d["wRES"] - raw_d["lRES"]
    n_consistent  = int((delta_res_raw < 0).sum())
    n_inverted    = int((delta_res_raw > 0).sum())

    summary = {
        "n_raw_pairs": int(len(raw)),
        "n_surplus_kept": int(len(surplus)),
        "fraction_kept": float(len(surplus) / max(len(raw), 1)),
        "fraction_gc_driven_dropped": 1.0 - float(len(surplus) / max(len(raw), 1)),
        "regression_coefficients": {"a": a, "b": b, "c": c},
        "raw_pairs": {
            "n_consistent_with_residual": n_consistent,
            "n_inverted_by_residual": n_inverted,
            "mean_delta_mfe":      float(np.mean(delta_mfe_raw)),
            "mean_delta_residual": float(np.mean(delta_res_raw)),
            "mean_winner_gc":      float(np.mean(raw_d["wGC"])),
            "mean_loser_gc":       float(np.mean(raw_d["lGC"])),
        },
        "surplus_pairs": {
            "mean_delta_mfe":      float(np.mean(surplus_d["wMFE"] - surplus_d["lMFE"])),
            "mean_delta_residual": float(np.mean(surplus_d["wRES"] - surplus_d["lRES"])),
            "mean_winner_gc":      float(np.mean(surplus_d["wGC"])),
            "mean_loser_gc":       float(np.mean(surplus_d["lGC"])),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"summary written -> {out_json}")

    # ----- Figure -----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) MFE vs GC*L on the raw winners + regression line
    GCL_raw = raw_d["wGC"] * raw_d["wL"]
    GCL_grid = np.linspace(GCL_raw.min(), GCL_raw.max(), 50)
    L_med = np.median(raw_d["wL"])
    pred_grid = a + b * L_med + c * GCL_grid
    axes[0, 0].scatter(GCL_raw, raw_d["wMFE"], s=2, alpha=0.2, color="steelblue", label="winners")
    axes[0, 0].plot(GCL_grid, pred_grid, "k--", lw=1.5, label=f"a+bL+c(GC·L), L={L_med:.0f}")
    axes[0, 0].set_xlabel("GC × length")
    axes[0, 0].set_ylabel("MFE (kcal/mol)")
    axes[0, 0].set_title(f"(a) Regression on winners: MFE = {a:.2f} + {b:.3f}L + {c:.3f}(GC·L)")
    axes[0, 0].legend(fontsize=8)

    # (b) ΔMFE vs Δresidual, colour by kept/dropped
    consistent_mask = delta_res_raw < 0
    axes[0, 1].scatter(
        delta_mfe_raw[consistent_mask], delta_res_raw[consistent_mask],
        s=2, alpha=0.3, color="seagreen", label=f"kept ({consistent_mask.sum()})"
    )
    axes[0, 1].scatter(
        delta_mfe_raw[~consistent_mask], delta_res_raw[~consistent_mask],
        s=2, alpha=0.3, color="firebrick", label=f"GC-driven, dropped ({(~consistent_mask).sum()})"
    )
    axes[0, 1].axhline(0, color="k", lw=0.6)
    axes[0, 1].axvline(0, color="k", lw=0.6)
    axes[0, 1].set_xlabel("Δ MFE (winner − loser)")
    axes[0, 1].set_ylabel("Δ MFE-residual (winner − loser)")
    axes[0, 1].set_title("(b) Pair classification: kept vs GC-driven")
    axes[0, 1].legend(fontsize=8)

    # (c) GC distributions
    axes[1, 0].hist(raw_d["wGC"], bins=40, alpha=0.5, label="raw winners", density=True, color="steelblue")
    axes[1, 0].hist(raw_d["lGC"], bins=40, alpha=0.5, label="raw losers", density=True, color="orange")
    axes[1, 0].axvline(np.mean(raw_d["wGC"]), color="steelblue", ls="--", lw=1)
    axes[1, 0].axvline(np.mean(raw_d["lGC"]), color="orange", ls="--", lw=1)
    axes[1, 0].set_xlabel("GC fraction")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("(c) GC distribution of raw winners vs losers")
    axes[1, 0].legend(fontsize=8)

    # (d) Residual distribution of kept pairs (winners vs losers)
    axes[1, 1].hist(surplus_d["wRES"], bins=40, alpha=0.5, label="kept winners", density=True, color="seagreen")
    axes[1, 1].hist(surplus_d["lRES"], bins=40, alpha=0.5, label="kept losers", density=True, color="lightcoral")
    axes[1, 1].axvline(np.mean(surplus_d["wRES"]), color="seagreen", ls="--", lw=1)
    axes[1, 1].axvline(np.mean(surplus_d["lRES"]), color="lightcoral", ls="--", lw=1)
    axes[1, 1].set_xlabel("MFE residual (kcal/mol)")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("(d) GC-controlled residual: kept winners vs losers")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Thermodynamic-surplus pair construction (margin {margin}σ): "
        f"{n_consistent}/{len(raw)} ({n_consistent/len(raw):.1%}) pairs kept, "
        f"{n_inverted}/{len(raw)} ({n_inverted/len(raw):.1%}) GC-driven and dropped",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"figure written -> {out_pdf}")


if __name__ == "__main__":
    main()
