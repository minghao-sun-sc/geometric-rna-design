# dpo/scripts/eda_length_distribution.py
import os, json, argparse, torch, csv
from collections import Counter
from dpo.common_id import extract_index_from_pair, extract_backbone_id_from_pair, canonicalize_id

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

def hist(rows, raws, id2g):
    hg=Counter(); hs=Counter()
    for gi,raw in enumerate(raws):
        Lg=len(raw.get("sequence","")); hg[Lg]+=1
    for p in rows:
        w=p.get("winner_seq") or p.get("winner") or ""
        hs[len(w or "")]+=1
    return hg, hs

def run(clean_path, processed_pt, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    rows=_load_jsonl(clean_path)
    raws=_load_processed(processed_pt)
    id2g=_index_all_ids(raws)

    hg,hs=hist(rows, raws, id2g)
    with open(os.path.join(out_dir,f"{split_name}_graph_len_hist.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Lg","count"])
        for L,c in sorted(hg.items()): w.writerow([L,c])
    with open(os.path.join(out_dir,f"{split_name}_seq_len_hist.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Lseq","count"])
        for L,c in sorted(hs.items()): w.writerow([L,c])
    print(f"[{split_name}] wrote length hist CSVs")

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
