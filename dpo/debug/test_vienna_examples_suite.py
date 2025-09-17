# dpo/debug/test_vienna_examples_suite.py
import os, sys, glob, json, csv

# Keep env clean
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluator import vienna_mfe, vienna_ensemble_metrics, vienna_Tm_by_pS0

EX_DIR = os.path.join(THIS_DIR, "example_data")
OUT_DIR = os.path.join(THIS_DIR, "out")
os.makedirs(EX_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def read_fasta_one(path: str) -> str:
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).replace("T","U").upper()

def read_dbn_or_none(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.readline().strip()

def main():
    fas = sorted(glob.glob(os.path.join(EX_DIR, "vienna_*.fa")))
    if not fas:
        print("No vienna_*.fa files in example_data/. Run run_generate_dbn_for_examples first if needed.")
        sys.exit(0)

    summary = []
    print(f"Discovered {len(fas)} files.\n")
    for fa in fas:
        name = os.path.splitext(os.path.basename(fa))[0]
        seq = read_fasta_one(fa)
        dbn_file = os.path.join(EX_DIR, name + ".dbn")
        target_db = read_dbn_or_none(dbn_file)

        # If no target DBN exists, use the sequence's own MFE structure as the target by default
        if target_db is None or len(target_db) != len(seq):
            _, target_db = vienna_mfe(seq, 37.0)

        # 37°C metrics
        e = vienna_ensemble_metrics(seq, target_db=target_db, T=37.0, return_positional_entropy=False)
        # MFE for reporting (already present in e, but also capture MFE db)
        mfe, mfe_db = e["mfe"], e["mfe_db"]

        # Tm (p(S0)≈0.5)
        Tm = vienna_Tm_by_pS0(seq, target_db, Tmin=10, Tmax=95, step=1.0, threshold=0.5)

        row = dict(
            name=name,
            N=len(seq),
            mfe=mfe,
            mfe_db=mfe_db,
            ED=e["ED"],
            ED_per_nt=e["ED_per_nt"],
            pS0=e["pS0"],
            entropy_mean=e["entropy_mean"],
            diversity=e["diversity"],
            Tm=Tm
        )
        summary.append(row)

        print(f"[{name}] N={len(seq)}  MFE={mfe:.2f}  ED/nt={row['ED_per_nt']:.4f}  pS0={row['pS0']:.3f}  "
              f"S-entropy={row['entropy_mean']:.3f}  div={row['diversity']:.3f}  Tm≈{Tm:.1f}°C")

    # Save CSV + JSON
    csv_path = os.path.join(OUT_DIR, "vienna_examples_summary.csv")
    json_path = os.path.join(OUT_DIR, "vienna_examples_summary.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"Wrote: {os.path.relpath(json_path, REPO_ROOT)}")

if __name__ == "__main__":
    main()
