# dpo/debug/mismatch_summary_by_id.py
import os, json, argparse, torch, statistics
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dpo.common_id import (
    canonicalize_id, canonical_from_id_list,
    extract_index_from_pair, extract_backbone_id_from_pair
)

def _load_pairs_any(path: str) -> List[dict]:
    rows = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s: rows.append(json.loads(s))
    elif path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        rows = obj if isinstance(obj, list) else obj.get("pairs", [])
    else:
        raise ValueError(f"unsupported: {path}")
    return rows

def _load_processed(ppt: str) -> List[dict]:
    data = torch.load(ppt)
    return list(data.values()) if isinstance(data, dict) else list(data)

def _load_split(sp: str):
    tr, va, te = torch.load(sp)
    return set(map(int, tr)), set(map(int, va)), set(map(int, te))

def _id2global(raws: List[dict]) -> Dict[str, int]:
    m = {}
    for gi, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and cid not in m: m[cid] = gi
    return m

def _which(gi: int, TR: set, VA: set, TE: set) -> str:
    if gi in TR: return "train"
    if gi in VA: return "val"
    if gi in TE: return "test"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    raws  = _load_processed(args.processed_pt)
    id2g  = _id2global(raws)
    TR, VA, TE = _load_split(args.split_pt)

    by_id = defaultdict(lambda: {"n":0,"n_ok":0,"n_mm":0,"deltas":[], "splits":Counter(), "Lg":None})
    for p in pairs:
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)
            gi = id2g.get(cid, None)
            key = cid
        else:
            key = canonical_from_id_list(raws[gi].get("id_list", []))

        if gi is None or gi<0 or gi>=len(raws):
            continue

        sp = _which(gi, TR, VA, TE)
        raw = raws[gi]
        Lg = len(raw.get("sequence",""))
        w = p.get("winner_seq") or p.get("winner") or ""
        l = p.get("loser_seq")  or p.get("loser")  or ""
        Lw, Ll = len(w), len(l)

        rec = by_id[key]
        rec["n"] += 1
        rec["splits"][sp] += 1
        if rec["Lg"] is None: rec["Lg"] = Lg

        if Lw == Ll == Lg:
            rec["n_ok"] += 1
        elif Lw == Ll:
            rec["n_mm"] += 1
            rec["deltas"].append(Lg - Lw)  # positive if graph is longer
        # else (W!=L): very rare in your data → ignore here

    # Rank by number of mismatches
    ranked = sorted(by_id.items(), key=lambda kv: kv[1]["n_mm"], reverse=True)
    print("backbone_id, total_pairs, mismatches, ok, Lg, delta_median, delta_min, delta_max, split_counts")
    for bid, rec in ranked[:args.top]:
        deltas = rec["deltas"]
        dmed = statistics.median(deltas) if deltas else 0
        dmin = min(deltas) if deltas else 0
        dmax = max(deltas) if deltas else 0
        print(f"{bid}, {rec['n']}, {rec['n_mm']}, {rec['n_ok']}, {rec['Lg']}, {dmed}, {dmin}, {dmax}, {dict(rec['splits'])}")

if __name__ == "__main__":
    main()
