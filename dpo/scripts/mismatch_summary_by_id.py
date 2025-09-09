# dpo/scripts/mismatch_summary_by_id.py
import argparse, json, torch
from collections import defaultdict, Counter
from statistics import median
from typing import List, Dict, Tuple
from dpo.common_id import canonicalize_id, canonical_from_id_list, extract_index_from_pair, extract_backbone_id_from_pair

def _load_pairs_any(p: str) -> List[dict]:
    pairs = []
    if p.endswith(".jsonl"):
        with open(p) as f:
            for line in f:
                s = line.strip()
                if s: pairs.append(json.loads(s))
    else:
        with open(p) as f:
            obj = json.load(f)
        pairs = obj if isinstance(obj, list) else obj.get("pairs", [])
    return pairs

def _load_processed(pt: str) -> List[dict]:
    d = torch.load(pt)
    return list(d.values()) if isinstance(d, dict) else list(d)

def _load_das_split(pt: str):
    tr, va, te = torch.load(pt)
    return set(tr), set(va), set(te)

def _index_all_ids(raws: List[dict]) -> Dict[str, int]:
    id2i = {}
    for i, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and cid not in id2i:
                id2i[cid] = i
    return id2i

def _split_of(i: int, s_tr: set, s_va: set, s_te: set) -> str:
    if i in s_tr: return "train"
    if i in s_va: return "val"
    if i in s_te: return "test"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    raws = _load_processed(args.processed_pt)
    s_tr, s_va, s_te = _load_das_split(args.split_pt)
    id2g = _index_all_ids(raws)

    # graph length map
    Lg = {i: len(raws[i].get("sequence","")) for i in range(len(raws))}

    per_id = defaultdict(lambda: {"total": 0, "mism": 0, "ok": 0, "deltas": [], "splits": Counter()})

    for p in pairs:
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)
            gi = id2g.get(cid, None)
        if gi is None:  # unknown; skip
            continue
        w = p.get("winner_seq") or p.get("winner") or ""
        l = p.get("loser_seq")  or p.get("loser")  or ""
        lw = len(w or ""); ll = len(l or ""); lg = Lg.get(gi, 0)
        cid_hint = canonical_from_id_list(raws[gi].get("id_list", []))
        rec = per_id[cid_hint]
        rec["total"] += 1
        rec["splits"][_split_of(gi, s_tr, s_va, s_te)] += 1
        if lw == ll == lg:
            rec["ok"] += 1
        else:
            rec["mism"] += 1
            if lw == ll and lg:
                rec["deltas"].append(abs(lw - lg))

    rows = []
    for cid, r in per_id.items():
        deltas = r["deltas"]
        rows.append((
            cid, r["total"], r["mism"], r["ok"], Lg[id2g[cid]] if cid in id2g else -1,
            median(deltas) if deltas else 0, min(deltas) if deltas else 0, max(deltas) if deltas else 0,
            dict(r["splits"])
        ))

    rows.sort(key=lambda x: (-x[2], x[0]))  # sort by mismatches desc
    print("\nbackbone_id, total_pairs, mismatches, ok, Lg, delta_median, delta_min, delta_max, split_counts")
    for r in rows[:args.top]:
        print(f"{r[0]}, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}, {r[6]}, {r[7]}, {r[8]}")

if __name__ == "__main__":
    main()
