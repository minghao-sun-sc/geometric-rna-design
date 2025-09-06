# dpo/debug/inspect_len_mismatch.py
import argparse, json, torch
from collections import Counter, defaultdict
from dpo.common_id import canonicalize_id, extract_backbone_id_from_pair

def load_pairs_any(p):
    if p.endswith(".jsonl"):
        with open(p) as f: return [json.loads(x) for x in f if x.strip()]
    with open(p) as f:
        obj = json.load(f)
        return obj if isinstance(obj, list) else obj["pairs"]

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--processed_pt", default="data/processed.pt")
ap.add_argument("--n", type=int, default=15)
args = ap.parse_args()

raws = torch.load(args.processed_pt)
raws = list(raws.values()) if isinstance(raws, dict) else list(raws)

# id -> graph (sequence) length
id2L = {}
for r in raws:
    L = len(r.get("sequence",""))
    for it in r.get("id_list", []):
        id2L[canonicalize_id(it)] = L

pairs = load_pairs_any(args.pairs)

cats = Counter()
by_backbone = Counter()
examples = defaultdict(list)

def norm_rna(s): return s.replace("t","U").replace("T","U")

for i,p in enumerate(pairs):
    gid = extract_backbone_id_from_pair(p)
    gL  = id2L.get(gid, None)

    w = p.get("winner_seq") or p.get("winner") or ""
    l = p.get("loser_seq")  or p.get("loser")  or ""
    wL, lL = len(w), len(l)

    if wL != lL:
        cats["unequal_winner_loser"] += 1
        if len(examples["unequal_winner_loser"]) < args.n:
            examples["unequal_winner_loser"].append((gid, wL, lL))
        continue

    if gL is None:
        cats["unknown_backbone"] += 1
        if len(examples["unknown_backbone"]) < args.n:
            examples["unknown_backbone"].append((gid, wL))
        continue

    if wL == gL:
        cats["perfect_match"] += 1
    elif wL < gL:
        cats["shorter_than_graph"] += 1
        by_backbone[gid] += 1
        if len(examples["shorter_than_graph"]) < args.n:
            examples["shorter_than_graph"].append((gid, wL, gL))
    else: # wL > gL
        cats["longer_than_graph"] += 1
        by_backbone[gid] += 1
        if len(examples["longer_than_graph"]) < args.n:
            examples["longer_than_graph"].append((gid, wL, gL))

print("=== mismatch categories ===")
for k,v in cats.items():
    print(f"{k:22s}: {v}")

print("\n=== examples ===")
for k,rows in examples.items():
    print(f"[{k}]")
    for r in rows: print("  ", r)

print("\n=== backbones with most mismatches ===")
for gid,c in by_backbone.most_common(args.n):
    print(f"{gid:15s}  {c}")
