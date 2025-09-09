# dpo/scripts/eda_clean_coverage.py
import os, json, argparse, torch, csv
from collections import defaultdict, Counter
from dpo.common_id import canonicalize_id, canonical_from_id_list, extract_index_from_pair, extract_backbone_id_from_pair

def _load_jsonl(p):
    rows=[]
    with open(p) as f:
        for line in f:
            s=line.strip()
            if s: rows.append(json.loads(s))
    return rows

def _load_processed(pt):
    d=torch.load(pt)
    return list(d.values()) if isinstance(d,dict) else list(d)

def _index_all_ids(raws):
    id2g={}
    for gi,raw in enumerate(raws):
        for it in raw.get("id_list",[]):
            cid=canonicalize_id(it)
            if cid and cid not in id2g: id2g[cid]=gi
    return id2g

def run(clean_path, processed_pt, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_jsonl(clean_path)
    raws = _load_processed(processed_pt)
    id2g  = _index_all_ids(raws)
    g2cid = {gi:canonical_from_id_list(raws[gi].get("id_list",[])) for gi in range(len(raws))}

    per_g = defaultdict(int)
    for p in rows:
        gi = extract_index_from_pair(p)
        if gi is None:
            cid = extract_backbone_id_from_pair(p)
            gi = id2g.get(cid, None)
        if gi is not None:
            per_g[gi]+=1

    csv_path = os.path.join(out_dir, f"{split_name}_coverage.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["backbone_id","global_index","clean_pairs"])
        for gi,count in sorted(per_g.items(), key=lambda x:(-x[1],x[0])):
            w.writerow([g2cid.get(gi,""), gi, count])

    md_path = os.path.join(out_dir, f"{split_name}_coverage.md")
    counts=list(per_g.values())
    total=len(rows); uniq=len(per_g)
    avg = (sum(counts)/uniq) if uniq>0 else 0.0
    with open(md_path,"w") as f:
        f.write(f"# Clean coverage — {split_name}\n\n")
        f.write(f"- clean pairs: **{total}**\n- unique backbones with ≥1 clean pair: **{uniq}**\n")
        f.write(f"- avg clean pairs per covered backbone: **{avg:.2f}**\n")
        if counts:
            f.write(f"- min/max: **{min(counts)} / {max(counts)}**\n")
    print(f"[{split_name}] wrote {csv_path} and {md_path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--train_clean", required=True)
    ap.add_argument("--val_clean", required=True)
    ap.add_argument("--test_clean", required=True)
    ap.add_argument("--out_dir", required=True)
    args=ap.parse_args()

    run(args.train_clean, args.processed_pt, args.out_dir, "train")
    run(args.val_clean,   args.processed_pt, args.out_dir, "val")
    run(args.test_clean,  args.processed_pt, args.out_dir, "test")

if __name__=="__main__":
    main()
