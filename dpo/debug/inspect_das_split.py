import os, argparse, torch
from typing import List
from dpo.common_id import canonicalize_id, canonical_from_id_list

def load_split(split_pt: str):
    train_idx, val_idx, test_idx = torch.load(split_pt)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

def load_processed(processed_pt: str) -> List[dict]:
    data = torch.load(processed_pt)
    return list(data.values()) if isinstance(data, dict) else list(data)

def ids_from_indices(raws: List[dict], indices: List[int]) -> List[str]:
    ids = []
    for i in indices:
        if 0 <= i < len(raws):
            cid = canonical_from_id_list(raws[i].get("id_list", []))
            ids.append(cid if cid else f"<missing_{i}>")
        else:
            ids.append(f"<oob_{i}>")
    return ids

def read_ids_file(path: str) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(canonicalize_id(s))
    return ids

def write_list(path: str, items: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x in items:
            f.write(str(x) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--test_ids_file", default="")
    ap.add_argument("--out_dir", default="dpo/debug/out")
    args = ap.parse_args()

    raws = load_processed(args.processed_pt)
    train_idx, val_idx, test_idx = load_split(args.split_pt)
    print("=== DAS SPLIT ===")
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}\n")

    train_ids = ids_from_indices(raws, train_idx)
    val_ids   = ids_from_indices(raws, val_idx)
    test_ids  = ids_from_indices(raws, test_idx)

    print("Sample IDs:")
    print("  train:", train_ids[:5])
    print("  val:  ", val_ids[:5])
    print("  test: ", test_ids[:5])
    print()

    write_list(os.path.join(args.out_dir, "das_train_ids.txt"), train_ids)
    write_list(os.path.join(args.out_dir, "das_val_ids.txt"),   val_ids)
    write_list(os.path.join(args.out_dir, "das_test_ids.txt"),  test_ids)

    if args.test_ids_file:
        ext = set(read_ids_file(args.test_ids_file))
        das = {x for x in test_ids if not x.startswith("<")}
        missing_in_das = sorted(list(ext - das))
        extra_in_das   = sorted(list(das - ext))
        print("=== Compare with external test_ids_das.txt ===")
        print(f"external test count: {len(ext)}   das test count: {len(das)}")
        print(f"missing in DAS test (present in external): {len(missing_in_das)}")
        if missing_in_das[:10]: print("  e.g.,", missing_in_das[:10])
        print(f"extra in DAS test (not in external):       {len(extra_in_das)}")
        if extra_in_das[:10]: print("  e.g.,", extra_in_das[:10])

if __name__ == "__main__":
    main()
