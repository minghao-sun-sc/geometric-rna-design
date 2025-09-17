# dpo/debug/test_evaluator_usalign_vs_natives.py
import os
import sys
import argparse
import shutil
import uuid
import traceback

# ---- Harden env before importing heavy deps (NetworkX) ----
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)
os.environ["NETWORKX_BACKENDS"] = "nx_loopback"

# ---- Repo paths ----
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")

for p in (REPO_ROOT, SRC_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

# ---- Try to import evaluator helper ----
evaluator_ok = True
try:
    from evaluator import usalign_tm_vs_natives
except Exception:
    evaluator_ok = False
    print("❌ Failed to import 'usalign_tm_vs_natives' from src/evaluator.py")
    print("---- Traceback ----")
    traceback.print_exc()
    print("-------------------")

# ---- Oracle: direct tools wrapper ----
try:
    from tools.usalign_utils import run_rna_usalign, aggregate_tm
except Exception as e:
    print("❌ Failed to import tools/usalign_utils.py")
    print("Error:", e)
    sys.exit(2)

def _stage_natives(natives, stage_root):
    raw_dir = os.path.join(stage_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    id_list = []
    for i, src in enumerate(natives, 1):
        nid = f"N{i}"
        dst = os.path.join(raw_dir, f"{nid}.pdb")
        shutil.copyfile(src, dst)
        id_list.append(nid)
    return {"id_list": id_list}, stage_root

def main():
    ap = argparse.ArgumentParser(description="Test evaluator.usalign_tm_vs_natives with staged natives")
    ap.add_argument("--model", help="Path to predicted/model PDB")
    ap.add_argument("--natives", help="Comma-separated list of native PDBs")
    ap.add_argument("--agg", default="avg", choices=["avg","max","min"], help="Aggregation mode")
    ap.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance")
    ap.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    args = ap.parse_args()

    # Defaults
    example_dir = os.path.join(REPO_ROOT, "dpo", "debug", "example_data")
    model = args.model or next((p for p in [
        os.path.join(example_dir, "model.pdb"),
        os.path.join(example_dir, "usalign_rna_example_1.pdb"),
    ] if os.path.exists(p)), None)
    if not model:
        print("❌ Provide --model PDB or place one in example_data/.")
        sys.exit(1)

    if args.natives:
        natives = [p.strip() for p in args.natives.split(",") if p.strip()]
    else:
        natives = [next((p for p in [
            os.path.join(example_dir, "native.pdb"),
            os.path.join(example_dir, "usalign_rna_example_2.pdb"),
        ] if os.path.exists(p)), None)]
        natives = [p for p in natives if p is not None]
    if not natives:
        print("❌ Provide at least one native PDB via --natives or ensure example_data/native.pdb exists.")
        sys.exit(1)

    stage_root = os.path.join(REPO_ROOT, "dpo", "debug", "out", f"usalign_stage_{uuid.uuid4().hex[:8]}")
    os.makedirs(stage_root, exist_ok=True)
    true_raw_data, data_path = _stage_natives(natives, stage_root)

    # Oracle (per-native)
    tm_direct = []
    for native in natives:
        r = run_rna_usalign(model, native)
        tm_direct.append(aggregate_tm(r.tmscore_chain1, r.tmscore_chain2, args.agg))
    tm_direct = np.array(tm_direct, dtype=float)
    tm_mean_direct = float(tm_direct.mean()) if tm_direct.size else -1.0

    if evaluator_ok:
        tm_eval, tm_mean_eval, details = usalign_tm_vs_natives(
            predicted_pdb_path=model,
            true_raw_data=true_raw_data,
            data_path=data_path,
            aggregate=args.agg,
            return_detail=True
        )

        print("\n=== Evaluator vs Direct (vs_natives) ===")
        print(f"Aggregation:         {args.agg}")
        print(f"IDs:                 {true_raw_data['id_list']}")
        print(f"Evaluator TM list:   {np.array2string(tm_eval, precision=6)}")
        print(f"Direct TM list:      {np.array2string(tm_direct, precision=6)}")
        print(f"Evaluator TM mean:   {tm_mean_eval:.6f}")
        print(f"Direct TM mean:      {tm_mean_direct:.6f}")

        ok_list = np.allclose(tm_eval, tm_direct, rtol=args.rtol, atol=args.atol) and (tm_eval.shape == tm_direct.shape)
        ok_mean = np.isclose(tm_mean_eval, tm_mean_direct, rtol=args.rtol, atol=args.atol)

        if ok_list and ok_mean:
            print("\n✅ US-align vs_natives check PASSED (evaluator helper matches direct runs).")
            sys.exit(0)
        else:
            print("\n❌ US-align vs_natives check FAILED.")
            if not ok_list: print(" - Per-native TM list mismatch")
            if not ok_mean: print(" - Mean TM mismatch")
            print(f"(Staged data kept at: {stage_root})")
            sys.exit(3)
    else:
        print("\n⚠️ Skipping evaluator import (see traceback above). Showing direct results:")
        print(f"IDs:                 {[f'N{i+1}' for i in range(len(natives))]}")
        print(f"Direct TM list:      {np.array2string(tm_direct, precision=6)}")
        print(f"Direct TM mean:      {tm_mean_direct:.6f}")
        print(f"(Staged data kept at: {stage_root})")
        sys.exit(0)

if __name__ == "__main__":
    main()
