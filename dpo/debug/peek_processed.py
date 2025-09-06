import os, argparse, pprint, torch
from itertools import islice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--n", type=int, default=5, help="number of samples to print")
    args = ap.parse_args()

    data = torch.load(args.processed_pt)
    # data can be dict or list; normalize to list of raws
    raws = list(data.values()) if isinstance(data, dict) else list(data)
    print(f"loaded {len(raws)} entries")
    print("---- keys present in first item ----")
    print(list(raws[0].keys()))
    print("---- first few items (trimmed) ----")
    for i, raw in enumerate(islice(raws, 0, args.n)):
        print(f"[{i}] keys:", list(raw.keys()))
        # print a few most likely ID-ish fields if present
        for k in ["backbone_id", "id", "pdb_id", "pdb_filepath", "chain", "model", "chain_id"]:
            if k in raw:
                print(f"    {k} =", raw[k])
        print()

if __name__ == "__main__":
    main()
