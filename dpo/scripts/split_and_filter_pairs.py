# dpo/scripts/split_and_filter_pairs.py
import os, argparse, subprocess, sys, json

def run(cmd: list):
    print(">>", " ".join(cmd)); rv = subprocess.run(cmd)
    if rv.returncode != 0:
        print(f"[error] command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(rv.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--processed_pt", default="data/processed.pt")
    ap.add_argument("--split_pt", default="data/das_split.pt")
    ap.add_argument("--ids_dir", default="")
    ap.add_argument("--split_source", choices=["das","ids"], default="das")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--emit_unknown", action="store_true")
    args = ap.parse_args()

    split_dir = os.path.join(args.out_dir, "split")
    clean_dir = os.path.join(args.out_dir, "clean")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # 1) Split
    cmd = [
        sys.executable, "-m", "dpo.scripts.filter_pairs_by_split",
        "--pairs_in", args.pairs_in,
        "--processed_pt", args.processed_pt,
        "--out_dir", split_dir,
        "--split_source", args.split_source,
    ]
    if args.split_source == "das":
        cmd += ["--split_pt", args.split_pt]
    else:
        cmd += ["--ids_dir", args.ids_dir]
    if args.emit_unknown:
        cmd += ["--emit_unknown"]
    run(cmd)

    # 2) Filter each split for length consistency
    for split in ["train","val","test"]:
        inp  = os.path.join(split_dir, f"{split}.jsonl")
        outc = os.path.join(clean_dir, f"{split}.clean.jsonl")
        outm = os.path.join(clean_dir, f"{split}.mismatch.jsonl")
        outu = os.path.join(clean_dir, f"{split}.unknown.jsonl")
        if not os.path.exists(inp):
            print(f"[warn] not found: {inp} (skipping)"); continue
        cmd = [
            sys.executable, "-m", "dpo.scripts.filter_length_mismatches",
            "--pairs_in", inp,
            "--processed_pt", args.processed_pt,
            "--out_clean", outc,
            "--out_mismatches", outm,
            "--out_unknown", outu,
        ]
        run(cmd)

    print("\n=== Done ===")
    print(f"split -> {split_dir}")
    print(f"clean -> {clean_dir}")

if __name__ == "__main__":
    main()
