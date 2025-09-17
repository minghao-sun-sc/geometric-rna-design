# dpo/debug/test_vienna_ensemble.py
import os, sys, json

# Keep NetworkX env clean (defense vs earlier import issue)
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")

for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import evaluator helpers
from evaluator import vienna_mfe, vienna_ensemble_metrics, vienna_Tm_by_pS0

EX_DIR = os.path.join(THIS_DIR, "example_data")
os.makedirs(EX_DIR, exist_ok=True)

# Example sequence (length 16) and a consistent target dot-bracket:
SEQ = "GGGGAAAAUUUUCCCC"           # 16 nt
DB  = "((((....))))...."           # 16 chars

# Write example files if missing (handy for future scripts/tools)
fa_path  = os.path.join(EX_DIR, "vienna_example.fa")
dbn_path = os.path.join(EX_DIR, "vienna_example.dbn")
if not os.path.exists(fa_path):
    with open(fa_path, "w") as f:
        f.write(">vienna_example\n" + SEQ + "\n")
if not os.path.exists(dbn_path):
    with open(dbn_path, "w") as f:
        f.write(DB + "\n")

print("Example data written to:", EX_DIR)

# --- Run metrics
print("\n== MFE at 37°C ==")
mfe, mfe_db = vienna_mfe(SEQ, 37.0)
print(f"MFE: {mfe:.2f} kcal/mol")
print(f"MFE structure: {mfe_db}")

print("\n== Ensemble metrics at 37°C ==")
metrics = vienna_ensemble_metrics(SEQ, target_db=DB, T=37.0, return_positional_entropy=True)
for k, v in metrics.items():
    if k == "entropy_list":
        print(f"{k}: (len={len(v)}) first5={v[:5]}")
    else:
        print(f"{k}: {v}")

# sanity checks
assert isinstance(metrics["mfe"], float)
assert 0.0 <= metrics["pS0"] <= 1.0 or (metrics["pS0"] != metrics["pS0"])  # allow nan
assert metrics["ED"] >= 0.0 or (metrics["ED"] != metrics["ED"])

print("\n== Tm sweep (p(S0)≈0.5) ==")
Tm = vienna_Tm_by_pS0(SEQ, DB, Tmin=10.0, Tmax=90.0, step=1.0, threshold=0.5)
print(f"Estimated Tm: {Tm:.1f} °C")

print("\n== JSON dump ==")
print(json.dumps({
    "mfe": mfe,
    "mfe_db": mfe_db,
    "ensemble": {k: (v if k != "entropy_list" else f"[{len(v)} entries]") for k, v in metrics.items()},
    "Tm": Tm
}, indent=2))
