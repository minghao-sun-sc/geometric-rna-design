# tools/print_dpo_api.py
import os, inspect, importlib, types, textwrap, time
import dpo.losses as losses

fn = losses.dpo_sft_step
sig = inspect.signature(fn)
src_path = inspect.getsourcefile(fn) or inspect.getfile(fn)
mtime = time.ctime(os.path.getmtime(src_path))

print("== dpo_sft_step location ==")
print(src_path)
print("last modified:", mtime)
print()

print("== signature ==")
print(sig)
print()

print("== parameter kinds ==")
for name, p in sig.parameters.items():
    print(f"{name:15} kind={p.kind} default={p.default!r}")
print()

print("== first 80 lines of source ==")
src_lines, start = inspect.getsourcelines(fn)
print("".join(src_lines[:80]))
print()

# Cheap token sniffing to see how 'models' is used:
src = "".join(src_lines).lower()
tokens = ["policy_model", "ref_model", "policy", "reference", "ref", "pi", "pi_ref"]
print("== token hits in function body ==")
for tok in tokens:
    print(f"{tok:15}: {src.count(tok)}")
