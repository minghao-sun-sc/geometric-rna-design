# dpo/debug/run_generate_dbn_for_examples.py
import os, sys, glob

# Harden env (avoid networkx hiccups when evaluator imports biotite elsewhere)
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluator import vienna_mfe

EX_DIR = os.path.join(THIS_DIR, "example_data")
os.makedirs(EX_DIR, exist_ok=True)

def read_fasta_one(path: str) -> str:
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).replace("T", "U").upper()

def main():
    fa_paths = sorted(glob.glob(os.path.join(EX_DIR, "vienna_*.fa")))
    if not fa_paths:
        print("No vienna_*.fa files found in example_data/.")
        sys.exit(0)

    print(f"Found {len(fa_paths)} FASTA files. Generating .dbn for each (target = MFE @37°C)...\n")
    for fa in fa_paths:
        base = os.path.splitext(os.path.basename(fa))[0]
        dbn = os.path.join(EX_DIR, base + ".dbn")
        if os.path.exists(dbn):
            print(f"✓ {base}: .dbn already exists -> {os.path.relpath(dbn, EX_DIR)}")
            continue
        seq = read_fasta_one(fa)
        mfe, mfe_db = vienna_mfe(seq, 37.0)
        with open(dbn, "w") as f:
            f.write(mfe_db + "\n")
        print(f"+ {base}: wrote {os.path.relpath(dbn, EX_DIR)}  (MFE={mfe:.2f} kcal/mol)")

    print("\nDone. You can now run the suite tests.")
if __name__ == "__main__":
    main()
