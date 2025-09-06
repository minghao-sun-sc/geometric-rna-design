# dpo/debug/peek_processed_ids.py
import argparse, torch
from itertools import islice

ap = argparse.ArgumentParser()
ap.add_argument("--processed_pt", default="data/processed.pt")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--m", type=int, default=5, help="how many id_list items to show per raw")
args = ap.parse_args()

data = torch.load(args.processed_pt)
raws = list(data.values()) if isinstance(data, dict) else list(data)
print(f"loaded {len(raws)} entries")
for i, raw in enumerate(islice(raws, 0, args.n)):
    ids = raw.get("id_list", [])
    print(f"\n[{i}] sequence_len={len(raw.get('sequence',''))}, id_list_len={len(ids)}")
    for j, it in enumerate(islice(ids, 0, args.m)):
        print(f"  id_list[{j}] = {it}")
