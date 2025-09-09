# dpo/debug/length_mismatch_report.py
import os, json, argparse, statistics, torch
from typing import List, Dict, Tuple, Optional
from collections import Counter
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

def _load_das_split(split_pt: str) -> Tuple[List[int], List[int], List[int]]:
    train_idx, val_idx, test_idx = torch.load(split_pt)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

def _id2global_from_all_raws(raws: List[dict]) -> Dict[str, int]:
    id2g: Dict[str, int] = {}
    for gi, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and (cid not in id2g):
                id2g[cid] = gi
    return id2g

def _which_split(i: int, tr: List[int], va: List[int], te: List[int]) -> str:
    if i in tr: return "train"
    if i in va: return "val"
    if i in te: return "test"
    return "unknown"

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _write_jsonl(path: str, rows: List[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="", help="Optional: data/das_split.pt to annotate split")
    ap.add_argument("--out_dir", default="", help="Optional: directory to dump mismatches/unknowns JSONL")
    ap.add_argument("--print_examples", type=int, default=20)
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    raws = _load_processed(args.processed_pt)
    id2g = _id2global_from_all_raws(raws)

    tr = va = te = []
    if args.split_pt:
        tr, va, te = _load_das_split(args.split_pt)

    stats = Counter()
    mismatches = []
    unknowns = []
    ok_count = 0

    for k, p in enumerate(pairs):
        # resolve to global index
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)  # handles pdb_file
            gi = id2g.get(cid, None)

        if gi is None or gi < 0 or gi >= len(raws):
            stats["unknown"] += 1
            if len(unknowns) < args.print_examples:
                unknowns.append({"idx": k, "cid_or_hint": p.get("pdb_file", p.get("backbone_id","")), "reason": "unresolved"})
            continue

        raw = raws[gi]
        Lg = len(raw.get("sequence", ""))

        w = p.get("winner_seq") or p.get("winner") or ""
        l = p.get("loser_seq")  or p.get("loser")  or ""
        Lw, Ll = len(w), len(l)

        sp = _which_split(gi, tr, va, te) if args.split_pt else "n/a"

        if Lw != Ll:
            stats["winner_loser_len_diff"] += 1
            if len(mismatches) < args.print_examples:
                mismatches.append({
                    "idx": k, "global_idx": gi, "split": sp,
                    "type": "W!=L", "Lg": Lg, "Lw": Lw, "Ll": Ll,
                    "cid_hint": canonical_from_id_list(raw.get("id_list", [])) or p.get("pdb_file","")
                })
            continue

        if Lw != Lg:
            stats["graph_len_mismatch"] += 1
            if len(mismatches) < args.print_examples:
                mismatches.append({
                    "idx": k, "global_idx": gi, "split": sp,
                    "type": "W==L!=G", "Lg": Lg, "Lw": Lw,
                    "cid_hint": canonical_from_id_list(raw.get("id_list", [])) or p.get("pdb_file","")
                })
            continue

        ok_count += 1

    total = len(pairs)
    print("=== Length Mismatch Report ===")
    print(f"total_pairs: {total}")
    print(f"resolved_ok : {ok_count}")
    print(f"winner_loser_len_diff : {stats['winner_loser_len_diff']}")
    print(f"graph_len_mismatch    : {stats['graph_len_mismatch']}")
    print(f"unknown               : {stats['unknown']}")

    if mismatches:
        print("\nExamples (mismatches):")
        for m in mismatches[:args.print_examples]:
            print(m)
    if unknowns:
        print("\nExamples (unknowns):")
        for u in unknowns[:args.print_examples]:
            print(u)

    if args.out_dir:
        _ensure_dir(args.out_dir)
        # dump full lists
        # collect all mismatches again fully
        mm_full, un_full = [], []
        for k, p in enumerate(pairs):
            gi = extract_index_from_pair(p)
            if gi is None:
                cid = extract_backbone_id_from_pair(p)
                gi = id2g.get(cid, None)
            if gi is None or gi < 0 or gi >= len(raws):
                un_full.append(p)
                continue
            raw = raws[gi]
            Lg = len(raw.get("sequence",""))
            w = p.get("winner_seq") or p.get("winner") or ""
            l = p.get("loser_seq")  or p.get("loser")  or ""
            if (len(w) != len(l)) or (len(w) != Lg):
                mm_full.append(p)
        _write_jsonl(os.path.join(args.out_dir, "mismatches.jsonl"), mm_full)
        _write_jsonl(os.path.join(args.out_dir, "unknown.jsonl"),    un_full)
        print(f"\nWrote: {os.path.join(args.out_dir,'mismatches.jsonl')} and unknown.jsonl")

if __name__ == "__main__":
    main()
