# dpo/scripts/filter_pairs_by_split.py
import os, json, argparse, torch
from typing import List, Dict, Iterable, Optional, Tuple
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
                if s:
                    pairs.append(json.loads(s))
    elif pairs_path.endswith(".json"):
        with open(pairs_path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            pairs = obj
        elif isinstance(obj, dict) and "pairs" in obj and isinstance(obj["pairs"], list):
            pairs = obj["pairs"]
        else:
            raise ValueError("JSON must be a list or {'pairs': [...]} format.")
    else:
        raise ValueError(f"Unsupported pairs file format: {pairs_path}")
    return pairs

def _load_processed(processed_pt: str) -> List[dict]:
    data = torch.load(processed_pt)
    return list(data.values()) if isinstance(data, dict) else list(data)

def _load_das_split(split_pt: str) -> Tuple[List[int], List[int], List[int]]:
    train_idx, val_idx, test_idx = torch.load(split_pt)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

def _load_ids_file(path: str) -> List[str]:
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(canonicalize_id(s))
    return out

def _split_sets_from_ids_dir(ids_dir: str) -> Tuple[set, set, set]:
    tr = _load_ids_file(os.path.join(ids_dir, "train_ids_das.txt"))
    va = _load_ids_file(os.path.join(ids_dir, "val_ids_das.txt"))
    te = _load_ids_file(os.path.join(ids_dir, "test_ids_das.txt"))
    return set(tr), set(va), set(te)

def _index_all_ids_for_split(raws: List[dict]) -> Dict[str, int]:
    # Map every canonical id in id_list to its local index within this split
    id2i: Dict[str, int] = {}
    for i, raw in enumerate(raws):
        for it in raw.get("id_list", []):
            cid = canonicalize_id(it)
            if cid and (cid not in id2i):
                id2i[cid] = i
    return id2i

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True, help="Input pairs file (.json or .jsonl)")
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--ids_dir", default="", help="Directory containing train/val/test_ids_das.txt")
    ap.add_argument("--split_source", choices=["das","ids"], default="das",
                    help="Use splits from das_split.pt (default) or from ids_dir text files.")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSONLs")
    ap.add_argument("--emit_unknown", action="store_true", help="Also write unknown.jsonl")
    args = ap.parse_args()

    pairs = _load_pairs_any(args.pairs_in)
    all_raws = _load_processed(args.processed_pt)
    n_all = len(all_raws)

    # Build split → local raws and index maps
    if args.split_source == "das":
        tr_idx, va_idx, te_idx = _load_das_split(args.split_pt)
        # local raw lists
        raw_tr = [all_raws[i] for i in tr_idx]
        raw_va = [all_raws[i] for i in va_idx]
        raw_te = [all_raws[i] for i in te_idx]
        # global->local
        g2l_tr = {g:i for i,g in enumerate(tr_idx)}
        g2l_va = {g:i for i,g in enumerate(va_idx)}
        g2l_te = {g:i for i,g in enumerate(te_idx)}
        # id→local
        id2l_tr = _index_all_ids_for_split(raw_tr)
        id2l_va = _index_all_ids_for_split(raw_va)
        id2l_te = _index_all_ids_for_split(raw_te)

        def resolve_pair(p: dict) -> str:
            # try explicit global index
            gi = extract_index_from_pair(p)
            if gi is not None:
                if gi in g2l_tr: return "train"
                if gi in g2l_va: return "val"
                if gi in g2l_te: return "test"
                return "unknown"
            # else resolve by id (includes pdb_file)
            cid = extract_backbone_id_from_pair(p)
            if cid in id2l_tr: return "train"
            if cid in id2l_va: return "val"
            if cid in id2l_te: return "test"
            return "unknown"

    else:  # split_source == "ids"
        s_tr, s_va, s_te = _split_sets_from_ids_dir(args.ids_dir)
        # Build id→global index map from all_raws (so we can still sanity-check length etc if needed)
        id2g: Dict[str,int] = {}
        for gi, raw in enumerate(all_raws):
            for it in raw.get("id_list", []):
                cid = canonicalize_id(it)
                if cid and (cid not in id2g):
                    id2g[cid] = gi

        def resolve_pair(p: dict) -> str:
            # prefer explicit index if given
            gi = extract_index_from_pair(p)
            if gi is not None:
                # place by external ids if the raw's cid matches a split set
                # fall back to unknown if no id matches
                cids = {canonicalize_id(it) for it in all_raws[gi].get("id_list", [])}
                if cids & s_tr: return "train"
                if cids & s_va: return "val"
                if cids & s_te: return "test"
                return "unknown"
            # else place by cid
            cid = extract_backbone_id_from_pair(p)
            if cid in s_tr: return "train"
            if cid in s_va: return "val"
            if cid in s_te: return "test"
            return "unknown"

    # Partition
    out = {"train": [], "val": [], "test": [], "unknown": []}
    for p in pairs:
        bucket = resolve_pair(p)
        out[bucket].append(p)

    # Write
    _ensure_dir(args.out_dir)
    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), out["train"])
    _write_jsonl(os.path.join(args.out_dir, "val.jsonl"),   out["val"])
    _write_jsonl(os.path.join(args.out_dir, "test.jsonl"),  out["test"])
    if args.emit_unknown:
        _write_jsonl(os.path.join(args.out_dir, "unknown.jsonl"), out["unknown"])

    # Stats
    print("=== filter_pairs_by_split ===")
    for k in ["train","val","test","unknown"]:
        print(f"{k:7s}: {len(out[k])}")
    if args.split_source == "das":
        print(f"(source: das_split.pt → train={len(raw_tr)} val={len(raw_va)} test={len(raw_te)})")
    else:
        print(f"(source: ids_dir={args.ids_dir})")

if __name__ == "__main__":
    main()
