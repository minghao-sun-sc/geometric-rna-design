#!/usr/bin/env python3
"""Build GC-controlled thermodynamic-surplus preference pairs.

Replaces the raw-MFE inequality in the preference filter with a GC-controlled
residual: MFE_residual = MFE - (a + b*L + c*(GC*L)), where (a,b,c) are fit on
the candidate pool. Pairs where the winner has a more-negative residual than
the loser are kept; others are dropped.

Inputs:  data/pairs_margin{25,125}/by_das/clean/{train,val,test}.clean.jsonl
Outputs: data/pairs_thermo_surplus_margin{25,125}/by_das/clean/*.clean.jsonl
         scripts/thermo_surplus_report_margin{25,125}.json (kept/dropped stats)

Usage: python scripts/build_thermo_surplus_pairs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path("/mnt/rna01/smh/projects/ribopo")


def gc_fraction(seq: str) -> float:
    """GC fraction of an RNA sequence (treats T == U; case-insensitive)."""
    if not seq:
        return 0.0
    upper = seq.upper().replace("T", "U")
    gc = sum(1 for c in upper if c in {"G", "C"})
    return gc / len(upper)


def collect_sequences(pair_records: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (length, gc, mfe) arrays for *all* sequences (winner + loser) in the pair set."""
    Ls, GCs, MFEs = [], [], []
    for rec in pair_records:
        for side in ("winner", "loser"):
            seq = rec[f"{side}_seq"]
            mfe = rec[f"{side}_metrics"]["mfe"]
            if seq is None or mfe is None:
                continue
            L = len(seq)
            if L < 2:
                continue
            Ls.append(L)
            GCs.append(gc_fraction(seq))
            MFEs.append(float(mfe))
    return np.asarray(Ls, dtype=float), np.asarray(GCs, dtype=float), np.asarray(MFEs, dtype=float)


def fit_regression(L: np.ndarray, GC: np.ndarray, MFE: np.ndarray) -> tuple[float, float, float, float]:
    """Fit MFE = a + b*L + c*(GC*L). Returns (a, b, c, R^2)."""
    X = np.column_stack([np.ones_like(L), L, GC * L])
    coef, *_ = np.linalg.lstsq(X, MFE, rcond=None)
    pred = X @ coef
    ss_res = np.sum((MFE - pred) ** 2)
    ss_tot = np.sum((MFE - MFE.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    a, b, c = coef.tolist()
    return a, b, c, float(r2)


def residual(seq: str, mfe: float, a: float, b: float, c: float) -> float:
    L = len(seq)
    GC = gc_fraction(seq)
    pred = a + b * L + c * (GC * L)
    return float(mfe - pred)


def process_pair_file(
    in_path: Path, out_path: Path, a: float, b: float, c: float
) -> dict:
    """Filter pairs by thermodynamic-surplus consistency. Returns stats dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_consistent = 0    # winner residual < loser residual (favored)
    n_inverted = 0      # winner residual > loser residual (drop — was GC-driven)
    n_tied = 0          # |residual diff| ~ 0
    delta_mfe = []
    delta_resid = []
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1
            ws = rec["winner_seq"]
            ls = rec["loser_seq"]
            wm = rec["winner_metrics"]["mfe"]
            lm = rec["loser_metrics"]["mfe"]
            if ws is None or ls is None or wm is None or lm is None:
                continue
            wr = residual(ws, wm, a, b, c)
            lr = residual(ls, lm, a, b, c)
            delta_mfe.append(wm - lm)
            delta_resid.append(wr - lr)
            if wr < lr - 1e-6:
                # winner has more-negative residual → genuinely better surplus
                rec["winner_metrics"]["mfe_residual"] = wr
                rec["loser_metrics"]["mfe_residual"] = lr
                fout.write(json.dumps(rec) + "\n")
                n_consistent += 1
            elif wr > lr + 1e-6:
                n_inverted += 1
            else:
                n_tied += 1
    return {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "n_input_pairs": n_in,
        "n_kept_consistent": n_consistent,
        "n_dropped_gc_driven": n_inverted,
        "n_dropped_tied": n_tied,
        "kept_fraction": n_consistent / max(n_in, 1),
        "mean_delta_mfe": float(np.mean(delta_mfe)) if delta_mfe else None,
        "mean_delta_residual_kept": (
            float(np.mean([d for d in delta_resid if d < -1e-6])) if delta_resid else None
        ),
    }


def main():
    out_root = PROJECT_ROOT / "data"
    summary = {}
    for margin in ("25", "125"):
        in_dir = PROJECT_ROOT / f"data/pairs_margin{margin}/by_das/clean"
        out_dir = PROJECT_ROOT / f"data/pairs_thermo_surplus_margin{margin}/by_das/clean"
        if not in_dir.exists():
            print(f"[skip] {in_dir} does not exist")
            continue

        # Step 1: pool all sequences to fit regression.
        all_records = []
        for split in ("train", "val", "test"):
            f = in_dir / f"{split}.clean.jsonl"
            if not f.exists():
                continue
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
        if not all_records:
            print(f"[skip] no records under {in_dir}")
            continue
        L, GC, MFE = collect_sequences(all_records)
        a, b, c, r2 = fit_regression(L, GC, MFE)
        n_seqs = int(L.size)
        gc_min, gc_max = float(GC.min()), float(GC.max())
        L_min, L_max = float(L.min()), float(L.max())
        print(
            f"[margin={margin}] regression on {n_seqs} sequences: "
            f"MFE = {a:.3f} + ({b:.4f})*L + ({c:.4f})*(GC*L), R^2 = {r2:.3f}; "
            f"GC range [{gc_min:.3f}, {gc_max:.3f}], L range [{L_min:.0f}, {L_max:.0f}]"
        )

        # Step 2: filter each split.
        per_split = {}
        for split in ("train", "val", "test"):
            in_f = in_dir / f"{split}.clean.jsonl"
            out_f = out_dir / f"{split}.clean.jsonl"
            if not in_f.exists():
                print(f"[skip] {in_f} missing")
                continue
            stats = process_pair_file(in_f, out_f, a, b, c)
            print(
                f"[margin={margin}/{split}] {stats['n_input_pairs']} -> kept={stats['n_kept_consistent']} "
                f"(GC-driven dropped={stats['n_dropped_gc_driven']}, tied={stats['n_dropped_tied']}), "
                f"kept_fraction={stats['kept_fraction']:.3f}"
            )
            per_split[split] = stats

        summary[f"margin{margin}"] = {
            "regression": {
                "a": a, "b": b, "c": c, "r2": r2,
                "n_seqs": n_seqs,
                "gc_range": [gc_min, gc_max],
                "L_range": [L_min, L_max],
            },
            "per_split": per_split,
        }

    report = PROJECT_ROOT / "scripts/thermo_surplus_report.json"
    report.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] full report: {report}")


if __name__ == "__main__":
    main()
