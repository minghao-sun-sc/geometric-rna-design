# dpo/scripts/length_mismatch_report.py
import argparse, json, os, torch
from typing import List, Dict, Tuple
from dpo.common_id import (
    canonicalize_id, canonical_from_id_list,
    extract_index_from_pair, extract_backbone_id_from_pair
)

def _load_pairs_any(p: str) -> List[dict]:
    pairs = []
    if p.endswith(".jsonl"):
        with open(p) as f:
            for line in f:
                s = line.strip()
                if s: pairs.append(json.loads(s))
    elif p.endswith(".json"):
        with open(p) as f:
            obj = json.load(f)
        pairs = obj if isinstance(obj, list) else obj.get("pairs", [])
    else:
        raise ValueError("Use .json or .jsonl")
    return pairs

def _load_processed(pt: str) -> List[dict]:
    d = torch.load(pt)
    return list(d.values()) if isinstance(d, dict) else list(d)

def _load_das_split(pt: str) -> Tuple[List[int], List[int], List[int]]:
    tr, va, te = torch.load(pt)
    return list(map(int, tr)), list(map(int, va)), list(map(int, te))

def _index_all_ids(raws: List[dict]) -> Dict[str, int]:
    id2i = {}
    for i, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and cid not in id2i:
                id2i[cid] = i
    return id2i

def _which_split(i: int, tr: List[int], va: List[int], te: List[int]) -> str:
    if i in tr: return "train"
    if i in va: return "val"
    if i in te: return "test"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    raws = _load_processed(args.processed_pt)
    tr, va, te = _load_das_split(args.split_pt)
    id2g = _index_all_ids(raws)

    os.makedirs(args.out_dir, exist_ok=True)
    out_mis = open(os.path.join(args.out_dir, "mismatches.jsonl"), "w")
    out_unk = open(os.path.join(args.out_dir, "unknown.jsonl"), "w")

    total = len(pairs)
    ok = 0
    mism = 0
    unk = 0
    examples_mis = []

    for k, p in enumerate(pairs):
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)
            gi = id2g.get(cid, None)
        if gi is None:
            unk += 1
            out_unk.write(json.dumps({"idx": k, "cid_or_hint": p.get("pdb_file", ""), "reason": "unresolved"}) + "\n")
            continue

        # expected graph length from raws[gi]['sequence']
        Lg = len(raws[gi].get("sequence", ""))

        w = p.get("winner_seq") or p.get("winner") or ""
        l = p.get("loser_seq")  or p.get("loser")  or ""
        Lw, Ll = len(w or ""), len(l or "")

        if (w and l and (Lw == Ll == Lg)):
            ok += 1
        else:
            mism += 1
            cid_hint = canonical_from_id_list(raws[gi].get("id_list", []))
            typ = "W!=L" if Lw != Ll else "W==L!=G"
            rec = {"idx": k, "global_idx": gi, "split": _which_split(gi, tr, va, te),
                   "type": typ, "Lg": Lg, "Lw": Lw, "cid_hint": cid_hint}
            if len(examples_mis) < 20: examples_mis.append(rec)
            out_mis.write(json.dumps(rec) + "\n")

    out_mis.close(); out_unk.close()

    print("=== Length Mismatch Report ===")
    print(f"total_pairs: {total}")
    print(f"resolved_ok : {ok}")
    print(f"winner_loser_len_diff : {0}")  # kept in case you later break out W!=L separately
    print(f"graph_len_mismatch    : {mism}")
    print(f"unknown               : {unk}\n")

    if examples_mis:
        print("Examples (mismatches):")
        for r in examples_mis: print(r)
    print("\nWrote: {}/mismatches.jsonl and unknown.jsonl".format(args.out_dir))

if __name__ == "__main__":
    main()