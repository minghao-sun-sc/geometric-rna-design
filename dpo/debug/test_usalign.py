# dpo/debug/test_usalign.py
import os, sys, argparse

# --- (A) Sanitize NetworkX env (defensive; see Option B for the root fix) ---
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

# --- (B) Set repo paths so we can import your USalign utils directly ---
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Your wrapper lives here (based on what you pasted):
# tools/usalign_utils.py  (the module that loads configs/... and finds USalign)
from tools.usalign_utils import run_rna_usalign

def get_usalign_tmscore_local(model_pdb, native_pdb, aggregate="avg"):
    """Tiny local helper so we don't import src/evaluator.py"""
    res = run_rna_usalign(model_pdb, native_pdb)
    if res is None:
        return -1.0, None
    tm1 = float(res.tmscore_chain1)
    tm2 = float(res.tmscore_chain2)
    if aggregate == "avg":
        tm = 0.5*(tm1+tm2)
    elif aggregate == "max":
        tm = max(tm1, tm2)
    else:
        tm = min(tm1, tm2)
    return tm, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=False)
    ap.add_argument("--native", required=False)
    ap.add_argument("--agg", default="avg", choices=["avg", "max", "min"])
    args = ap.parse_args()

    model_pdb  = args.model
    native_pdb = args.native
    if model_pdb is None or native_pdb is None:
        example_dir = os.path.join(REPO_ROOT, "dpo", "debug", "example_data")
        model_pdb   = os.path.join(example_dir, "usalign_rna_example_1.pdb")
        native_pdb  = os.path.join(example_dir, "usalign_rna_example_2.pdb")
        if not (os.path.exists(model_pdb) and os.path.exists(native_pdb)):
            print("Provide --model and --native, or put model.pdb/native.pdb in dpo/debug/example_data/")
            sys.exit(1)

    tm, res = get_usalign_tmscore_local(model_pdb, native_pdb, args.agg)
    if res is None:
        sys.exit(2)

    print("\n--- US-align RESULTS ---")
    print(f"Aggregated TM-score ({args.agg}): {tm:.4f}")
    print(f"TM-score (chain1 norm): {res.tmscore_chain1:.4f}")
    print(f"TM-score (chain2 norm): {res.tmscore_chain2:.4f}")
    print(f"RMSD (Å):               {res.rmsd:.3f}")
    print(f"Aligned length:         {res.align_len}")
    print(f"Seq identity:           {res.seq_identity:.2f}")

if __name__ == "__main__":
    main()
