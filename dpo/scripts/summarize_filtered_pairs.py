# dpo/scripts/summarize_filtered_pairs.py
import os, argparse, json, glob

def _count_jsonl(path: str) -> int:
    if not os.path.exists(path): return 0
    n = 0
    with open(path) as f:
        for _ in f: n += 1
    return n

def _summ_split(root: str):
    split = os.path.join(root, "split")
    return {
        "train": _count_jsonl(os.path.join(split, "train.jsonl")),
        "val":   _count_jsonl(os.path.join(split, "val.jsonl")),
        "test":  _count_jsonl(os.path.join(split, "test.jsonl")),
        "unknown": _count_jsonl(os.path.join(split, "unknown.jsonl")),
    }

def _summ_clean(root: str):
    clean = os.path.join(root, "clean")
    return {
        "train": _count_jsonl(os.path.join(clean, "train.clean.jsonl")),
        "val":   _count_jsonl(os.path.join(clean, "val.clean.jsonl")),
        "test":  _count_jsonl(os.path.join(clean, "test.clean.jsonl")),
        "train_mismatch": _count_jsonl(os.path.join(clean, "train.mismatch.jsonl")),
        "val_mismatch":   _count_jsonl(os.path.join(clean, "val.mismatch.jsonl")),
        "test_mismatch":  _count_jsonl(os.path.join(clean, "test.mismatch.jsonl")),
        "train_unknown":  _count_jsonl(os.path.join(clean, "train.unknown.jsonl")),
        "val_unknown":    _count_jsonl(os.path.join(clean, "val.unknown.jsonl")),
        "test_unknown":   _count_jsonl(os.path.join(clean, "test.unknown.jsonl")),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., data/pairs_margin125/by_das")
    ap.add_argument("--report_md", required=True)
    args = ap.parse_args()

    sp = _summ_split(args.root)
    cl = _summ_clean(args.root)

    os.makedirs(os.path.dirname(args.report_md), exist_ok=True)
    with open(args.report_md, "w") as f:
        f.write("# Filtered Pairs Summary\n\n")
        f.write("## Split counts (before length filtering)\n\n")
        for k in ["train","val","test","unknown"]:
            f.write(f"- {k}: **{sp.get(k,0)}**\n")
        f.write("\n## Clean counts (after length filtering)\n\n")
        for k in ["train","val","test"]:
            f.write(f"- {k}: **{cl.get(k,0)}**\n")
        f.write("\n## Mismatch counts\n\n")
        for k in ["train_mismatch","val_mismatch","test_mismatch"]:
            f.write(f"- {k}: **{cl.get(k,0)}**\n")
        f.write("\n## Unknown counts\n\n")
        for k in ["train_unknown","val_unknown","test_unknown"]:
            f.write(f"- {k}: **{cl.get(k,0)}**\n")

    print("=== Summary ===")
    print("split:", sp)
    print("clean:", cl)
    print(f"Wrote report -> {args.report_md}")

if __name__ == "__main__":
    main()
