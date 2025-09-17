# dpo/debug/test_vienna_sensitivity.py
import os, sys
for k in list(os.environ):
    if k.startswith("NETWORKX_"):
        os.environ.pop(k, None)

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluator import _vienna_fc, vienna_Tm_by_pS0

SEQ = "GGGGAAAAUUUUCCCC"
DB  = "((((....))))...."

def pS0_at_T(T):
    fc, _ = _vienna_fc(SEQ, T)
    fc.pf()
    return float(fc.pr_structure(DB))

print("p(S0) vs T sweep:")
Ts = [20, 30, 37, 45, 60, 75]
vals = [(T, pS0_at_T(T)) for T in Ts]
for T, p in vals:
    print(f"  T={T:>3} °C  p(S0)={p:.3f}")

Tm = vienna_Tm_by_pS0(SEQ, DB, Tmin=10.0, Tmax=90.0, step=1.0, threshold=0.5)
print(f"\nEstimated Tm ~ {Tm:.1f} °C")

# loose monotonic trend check (not strictly monotone in all RNAs, so keep soft)
if vals[0][1] >= vals[-1][1]:
    print("✔️  p(S0) decreased overall with temperature (expected trend).")
else:
    print("⚠️  p(S0) did not show overall decrease; inspect sequence/target.")
