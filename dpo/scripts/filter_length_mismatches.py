# dpo/scripts/filter_length_mismatches.py
import os, json, argparse, torch
from typing import List, Dict
from dpo.common_id import (
    canonicalize_id, canonical_from_id_list,
    extract_index_from_pair, extract_backbone_id_from_pair
)

def _load_pairs_any(pairs_path: str) -> List[dict]:
    pairs: List[dict] = []
    if pairs_path.endswith(".jsonl"):
        with open(pairs_path) as f:
            for line in f:
                s = line.strip()
                if s: pairs.append(json.loads(s))
    elif pairs_path.endswith(".json"):
        with open(pairs_path) as f:
            obj = json.load(f)
        if isinstance(obj, list): pairs = obj
        elif isinstance(obj, dict) and "pairs" in obj: pairs = obj["pairs"]
        else: raise ValueError("JSON must be list or {'pairs': [...]} format.")
    else:
        raise ValueError(f"Unsupported pairs file format: {pairs_path}")
    return pairs

def _load_processed(processed_pt: str) -> List[dict]:
    data = torch.load(processed_pt)
    return list(data.values()) if isinstance(data, dict) else list(data)

def _id2global_from_all_raws(raws: List[dict]) -> Dict[str, int]:
    id2g: Dict[str, int] = {}
    for gi, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and (cid not in id2g):
                id2g[cid] = gi
    return id2g

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _write_jsonl(path: str, rows: List[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--out_clean", required=True, help="Output JSONL with length-consistent pairs")
    ap.add_argument("--out_mismatches", default="", help="Optional JSONL of dropped mismatches")
    ap.add_argument("--out_unknown", default="", help="Optional JSONL of unresolved pairs")
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    raws = _load_processed(args.processed_pt)
    id2g = _id2global_from_all_raws(raws)

    keep, mismatches, unknowns = [], [], []

    for p in pairs:
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)
            gi = id2g.get(cid, None)
        if gi is None or gi < 0 or gi >= len(raws):
            unknowns.append(p)
            continue

        raw = raws[gi]
        Lg = len(raw.get("sequence",""))
        w = p.get("winner_seq") or p.get("winner") or ""
        l = p.get("loser_seq")  or p.get("loser")  or ""
        if (len(w) == len(l) == Lg):
            keep.append(p)
        else:
            mismatches.append(p)

    _ensure_dir(os.path.dirname(args.out_clean) or ".")
    _write_jsonl(args.out_clean, keep)

    if args.out_mismatches:
        _ensure_dir(os.path.dirname(args.out_mismatches) or ".")
        _write_jsonl(args.out_mismatches, mismatches)
    if args.out_unknown:
        _ensure_dir(os.path.dirname(args.out_unknown) or ".")
        _write_jsonl(args.out_unknown, unknowns)

    print("=== filter_length_mismatches ===")
    print(f"input: {len(pairs)}  kept: {len(keep)}  mismatches: {len(mismatches)}  unknown: {len(unknowns)}")
    print(f"clean -> {args.out_clean}")
    if args.out_mismatches: print(f"mismatches -> {args.out_mismatches}")
    if args.out_unknown:    print(f"unknown    -> {args.out_unknown}")

if __name__ == "__main__":
    main()
