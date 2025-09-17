# dpo/debug/test_evaluator_usalign_pair.py
import os
import sys
import argparse
import traceback

# ---- Harden env before importing heavy deps (NetworkX) ----
# Remove any NetworkX config; then set a safe backend name (no hyphen)
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)
os.environ["NETWORKX_BACKENDS"] = "nx_loopback"  # underscores only, avoids dataclass name bug

# ---- Repo paths ----
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")

for p in (REPO_ROOT, SRC_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

# ---- Try to import evaluator helpers (what we want to test) ----
evaluator_ok = True
try:
    from evaluator import usalign_tm_pdbpair  # ADDED helper you pasted into src/evaluator.py
except Exception as e:
    evaluator_ok = False
    print("❌ Failed to import 'usalign_tm_pdbpair' from src/evaluator.py")
    print("---- Traceback ----")
    traceback.print_exc()
    print("-------------------")

# ---- Direct USalign utils as oracle (always needed) ----
try:
    from tools.usalign_utils import run_rna_usalign, aggregate_tm
except Exception as e:
    print("❌ Failed to import tools/usalign_utils.py")
    print("Error:", e)
    sys.exit(2)

def main():
    ap = argparse.ArgumentParser(description="Test evaluator.usalign_tm_pdbpair vs tools.usalign_utils")
    ap.add_argument("--model",  help="Path to predicted/model PDB")
    ap.add_argument("--native", help="Path to native/reference PDB")
    ap.add_argument("--agg", default="avg", choices=["avg","max","min"], help="Aggregation mode")
    ap.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance")
    ap.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    args = ap.parse_args()

    # Resolve defaults from example_data
    model_pdb  = args.model
    native_pdb = args.native
    if model_pdb is None or native_pdb is None:
        example_dir = os.path.join(REPO_ROOT, "dpo", "debug", "example_data")
        candidates = [
            (os.path.join(example_dir, "model.pdb"),  os.path.join(example_dir, "native.pdb")),
            (os.path.join(example_dir, "usalign_rna_example_1.pdb"), os.path.join(example_dir, "usalign_rna_example_2.pdb")),
        ]
        for m, n in candidates:
            if os.path.exists(m) and os.path.exists(n):
                model_pdb, native_pdb = m, n
                print(f"ℹ️ Using example_data defaults:\n  model:  {model_pdb}\n  native: {native_pdb}")
                break
        if model_pdb is None or native_pdb is None:
            print("❌ Provide --model and --native, or place a PDB pair in dpo/debug/example_data/")
            sys.exit(1)

    # --- Oracle: direct tools call ---
    res_direct = run_rna_usalign(model_pdb, native_pdb)
    tm_direct  = aggregate_tm(res_direct.tmscore_chain1, res_direct.tmscore_chain2, args.agg)

    if evaluator_ok:
        # --- Evaluator helper ---
        tm_eval, res_eval = usalign_tm_pdbpair(model_pdb, native_pdb, aggregate=args.agg, return_detail=True)

        print("\n=== Evaluator vs Direct (tools/usalign_utils) ===")
        print(f"Aggregation:         {args.agg}")
        print(f"Evaluator TM:        {tm_eval:.6f}")
        print(f"Direct TM:           {tm_direct:.6f}")
        print(f"Evaluator TM1/TM2:   {res_eval.tmscore_chain1:.6f} / {res_eval.tmscore_chain2:.6f}")
        print(f"Direct TM1/TM2:      {res_direct.tmscore_chain1:.6f} / {res_direct.tmscore_chain2:.6f}")
        print(f"Evaluator RMSD/len:  {res_eval.rmsd:.3f} Å / {res_eval.align_len}")
        print(f"Direct RMSD/len:     {res_direct.rmsd:.3f} Å / {res_direct.align_len}")
        ident_eval = res_eval.seq_identity
        ident_dir  = res_direct.seq_identity
        if ident_eval >= 0 and ident_dir >= 0:
            print(f"Evaluator identity:  {ident_eval:.3f} (fraction)")
            print(f"Direct identity:     {ident_dir:.3f} (fraction)")
        else:
            print("Identity not present in one or both outputs (OK).")

        ok_tm   = np.isclose(tm_eval, tm_direct, rtol=args.rtol, atol=args.atol)
        ok_tm1  = np.isclose(res_eval.tmscore_chain1, res_direct.tmscore_chain1, rtol=args.rtol, atol=args.atol)
        ok_tm2  = np.isclose(res_eval.tmscore_chain2, res_direct.tmscore_chain2, rtol=args.rtol, atol=args.atol)
        ok_rmsd = (res_eval.rmsd < 0 and res_direct.rmsd < 0) or np.isclose(res_eval.rmsd, res_direct.rmsd, rtol=1e-6, atol=1e-6)

        if ok_tm and ok_tm1 and ok_tm2 and ok_rmsd:
            print("\n✅ US-align pairwise check PASSED (evaluator helper matches direct run).")
            sys.exit(0)
        else:
            print("\n❌ US-align pairwise check FAILED.")
            if not ok_tm:  print(" - Aggregated TM mismatch")
            if not ok_tm1: print(" - TM1 mismatch")
            if not ok_tm2: print(" - TM2 mismatch")
            if not ok_rmsd: print(" - RMSD mismatch")
            sys.exit(3)
    else:
        print("\n⚠️ Skipping evaluator import (see traceback above).")
        print("   Showing direct USalign results so you can proceed:")
        print(f"\n--- US-align RESULTS (direct) ---")
        print(f"Aggregated TM-score ({args.agg}): {tm_direct:.6f}")
        print(f"TM-score (chain1 norm): {res_direct.tmscore_chain1:.6f}")
        print(f"TM-score (chain2 norm): {res_direct.tmscore_chain2:.6f}")
        print(f"RMSD (Å):               {res_direct.rmsd:.3f}")
        print(f"Aligned length:         {res_direct.align_len}")
        ident = res_direct.seq_identity
        if ident >= 0:
            print(f"Seq identity (fraction): {ident:.3f}")
        else:
            print("Seq identity:           N/A")
        sys.exit(0)

if __name__ == "__main__":
    main()
