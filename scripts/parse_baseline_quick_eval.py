"""Parse a baseline's full_eval_test.csv into eval_summary.json — same format
as the per-checkpoint summaries written by scripts/the SSTT eval driver.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True,
                   help="e.g. runs/baselines/rhodesign")
    p.add_argument("--tag", required=True, help="e.g. rhodesign")
    args = p.parse_args()

    base = Path(args.baseline_dir)
    csv_path = base / "full_eval_test.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {csv_path}")
    row = rows[0]

    def fl(k):
        v = row.get(k)
        if v is None or v == "":
            return None
        try:
            x = float(v)
            return None if math.isnan(x) else x
        except ValueError:
            return None

    summary = {
        "tag": args.tag,
        "ckpt": "from_fasta_dir",
        "recovery":         fl("recovery"),
        "scMCC":            fl("sc_eternafold"),
        "MFE":              fl("vienna_mfe"),
        "RMSD":             fl("sc_rmsd"),
        "TM":               fl("sc_tm"),
        "pLDDT":            fl("sc_plddt"),
        "diversity":        fl("diversity_3mer"),
        "inf_all":          fl("inf_all"),
        "inf_wc":           fl("inf_wc"),
        "inf_nwc":          fl("inf_nwc"),
        "rmsd_within_8A":   fl("rmsd_within_8A"),
        "plddt_above_070":  fl("plddt_above_070"),
        "vienna_pS0":       fl("vienna_pS0"),
        "vienna_Tm":        fl("vienna_Tm"),
        "vienna_ED_per_nt": fl("vienna_ED_per_nt"),
        "perplexity":       fl("perplexity"),
    }
    out = base / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
