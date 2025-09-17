# dpo/debug/test_vienna_ps0_trends.py
import os, sys, glob

# Clean env
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluator import _vienna_fc, vienna_mfe

EX_DIR = os.path.join(THIS_DIR, "example_data")

def read_fasta_one(path: str) -> str:
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).replace("T","U").upper()

def pS0_at_T(seq: str, db: str, T: float) -> float:
    fc, _ = _vienna_fc(seq, T)
    fc.pf()
    return float(fc.pr_structure(db))

def main():
    fas = sorted(glob.glob(os.path.join(EX_DIR, "vienna_*.fa")))
    if not fas:
        print("No vienna_*.fa files in example_data/.")
        sys.exit(0)

    Ts = list(range(15, 96, 5))  # 15..95°C step 5
    for fa in fas:
        name = os.path.splitext(os.path.basename(fa))[0]
        seq = read_fasta_one(fa)
        # Use each sequence's own MFE structure as the target — this yields interpretable trends
        _, db = vienna_mfe(seq, 20.0)
        vals = [(T, pS0_at_T(seq, db, T)) for T in Ts]
        overall_down = vals[0][1] >= vals[-1][1]  # loose, qualitative

        print(f"\n[{name}] len={len(seq)}  target=MFE@20°C")
        for T, p in vals:
            print(f"  T={T:>3}°C  p(S0)={p:.3f}")
        print("Trend:", "✔️ overall decrease" if overall_down else "⚠️ not overall decreasing")

if __name__ == "__main__":
    main()
