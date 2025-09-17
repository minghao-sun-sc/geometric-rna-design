# dpo/debug/test_mcq.py
import os, sys

# Clean NETWORKX_* to avoid unrelated import issues
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluator import mcq_pseudotorsion, mcq_pseudotorsion_stats

EX_DIR = os.path.join(THIS_DIR, "example_data")
model  = os.path.join(EX_DIR, "model.pdb")
native = os.path.join(EX_DIR, "native.pdb")
if not (os.path.exists(model) and os.path.exists(native)):
    print("Provide example PDBs at dpo/debug/example_data/model.pdb and native.pdb.")
    sys.exit(1)

# Original mean-direction "MCQ" (not a magnitude)
res_dir = mcq_pseudotorsion(model, native)

# New evaluation-friendly stats
res = mcq_pseudotorsion_stats(model, native)

print("\n--- MCQ (η/θ pseudo-torsions) ---")
print(f"Positions used (N):     {res['n_positions']}")
print(f"MCQ_abs (deg):          {res['mcq_abs_deg']:.2f}   <-- use this for evaluation (lower is better)")
print(f"Mean |Δη| (deg):        {res['mean_abs_diff_eta']:.2f}")
print(f"Mean |Δθ| (deg):        {res['mean_abs_diff_theta']:.2f}")
print(f"Mean direction |Δ| (°): {res['mean_dir_deg_abs']:.2f}   (original 'MCQ'; not a magnitude)")
print(f"Resultant length R:     {res['R']:.3f}   (0..1, higher is better)")
print(f"Circular SD (deg):      {res['circ_sd_deg']:.2f}   (lower is better)")

# quick consistency check
if res['n_positions'] == 0 or res['mcq_abs_deg'] != res['mcq_abs_deg']:
    print("⚠️  MCQ could not be computed.")
else:
    print("✅  MCQ computed successfully.")
