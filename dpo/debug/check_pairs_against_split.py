# dpo/debug/check_pairs_against_split.py
import argparse, statistics, torch
from collections import Counter, defaultdict
from typing import List, Dict
from dpo.common_id import (
    canonicalize_id, canonical_from_id_list,
    extract_index_from_pair, extract_backbone_id_from_pair
)

def load_pairs(path: str) -> List[dict]:
    import json
    pairs = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    pairs.append(json.loads(s))
    elif path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
            if isinstance(obj, list): pairs = obj
            elif isinstance(obj, dict) and "pairs" in obj: pairs = obj["pairs"]
            else: raise ValueError("JSON must be list or {'pairs': [...]} format.")
    else:
        raise ValueError("Use .jsonl or .json for pairs.")
    return pairs

def load_processed(processed_pt: str) -> List[dict]:
    data = torch.load(processed_pt)
    return list(data.values()) if isinstance(data, dict) else list(data)

def load_das_split(split_pt: str):
    train_idx, val_idx, test_idx = torch.load(split_pt)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

def index_all_backbones(raws: List[dict]) -> Dict[str, int]:
    id2i: Dict[str, int] = {}
    for i, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and (cid not in id2i):
                id2i[cid] = i
    return id2i

def which_split(i: int, train_idx: List[int], val_idx: List[int], test_idx: List[int]) -> str:
    if i in train_idx: return "train"
    if i in val_idx:   return "val"
    if i in test_idx:  return "test"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_path", default="data/pairs_margin125/dpo_pairs_margin125.json")
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--allow_test", action="store_true")
    ap.add_argument("--print_examples", type=int, default=20)
    args = ap.parse_args()

    pairs = load_pairs(args.pairs_path)
    print(f"Pairs: {args.pairs_path}")
    print(f"Loaded {len(pairs)} pairs.")

    raws = load_processed(args.processed_pt)
    train_idx, val_idx, test_idx = load_das_split(args.split_pt)
    id2i = index_all_backbones(raws)
    n_raw = len(raws)

    counts = Counter()
    problems = []
    weights, lengths, masks_present = [], [], 0
    split_bk_counter = Counter()

    for k, p in enumerate(pairs):
        # Try direct index
        i = extract_index_from_pair(p)
        if i is not None and 0 <= i < n_raw:
            sp = which_split(i, train_idx, val_idx, test_idx)
            # derive canonical id from the referenced raw
            cid = canonical_from_id_list(raws[i].get("id_list", []))
        else:
            # Try id-based resolution (now including pdb_file)
            cid = extract_backbone_id_from_pair(p)
            if cid in id2i:
                i = id2i[cid]
                sp = which_split(i, train_idx, val_idx, test_idx)
            else:
                i = None
                sp = "unknown"

        counts[sp] += 1
        if sp != "unknown":
            split_bk_counter[(sp, cid)] += 1
            if (sp == "test") and (not args.allow_test):
                problems.append((k, cid or "<empty>", "pair_in_test_split"))
        else:
            # show the best guess that failed (cid could be from pdb_file)
            problems.append((k, cid or p.get("pdb_file",""), "backbone_id_not_found_in_processed"))

        w = float(p.get("weight", 1.0)); weights.append(w)
        y_w, y_l = p.get("winner_seq",""), p.get("loser_seq","")
        if y_w and y_l and (len(y_w) == len(y_l)): lengths.append(len(y_w))
        if p.get("seq_mask", None) is not None: masks_present += 1

    print("\n=== Pairs by DAS split ===")
    for sp in ["train","val","test","unknown"]:
        print(f"{sp:7s}: {counts[sp]}")

    uniq_bk_per_split = defaultdict(set)
    for (sp, gid), _ in split_bk_counter.items():
        uniq_bk_per_split[sp].add(gid)
    print("\nUnique backbones per split (in pairs):")
    for sp in ["train","val","test"]:
        print(f"  {sp}: {len(uniq_bk_per_split[sp])}")

    if weights:
        print("\nWeights: min={:.3f}  mean={:.3f}  median={:.3f}  max={:.3f}".format(
            min(weights), statistics.mean(weights), statistics.median(weights), max(weights)
        ))
    if lengths:
        print("Seq length: min={}  mean={:.1f}  median={}  max={}".format(
            min(lengths), statistics.mean(lengths), statistics.median(lengths), max(lengths)
        ))
    print(f"Pairs with seq_mask: {masks_present} / {len(pairs)}")

    if problems:
        print(f"\n[warn] Found {len(problems)} potential issues:")
        for r in problems[:args.print_examples]:
            print("  idx={}, backbone_id={}, issue={}".format(*r))
    else:
        print("\nNo issues detected.")

if __name__ == "__main__":
    main()
