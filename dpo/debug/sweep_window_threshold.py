# dpo/debug/sweep_window_threshold.py
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import argparse, json, yaml, os
from dpo.data import PreferencePairDataset

def count_lines(path):
    if path.endswith(".jsonl"):
        return sum(1 for _ in open(path))
    return len(json.load(open(path)))

def run(cfg, thresholds):
    dc = cfg["data"]
    total = count_lines(dc["pairs_path_train"])
    print(f"total pairs (train jsonl): {total}")
    for t in thresholds:
        cfg_t = yaml.safe_load(open(args.config))
        cfg_t["data"]["min_window_identity"] = float(t)
        ds = PreferencePairDataset(
            processed_pt=dc["processed_pt"],
            split_file=dc["split_file"],
            pairs_path=dc["pairs_path_train"],
            split="train",
            max_num_conformers=dc["max_num_conformers"],
            radius=dc["radius"],
            top_k=dc["top_k"],
            num_rbf=dc["num_rbf"],
            num_posenc=dc["num_posenc"],
            noise_scale=dc["noise_scale"],
            device="cpu",
            use_seq_mask=dc.get("use_seq_mask", True),
            strict_length_check=bool(dc.get("strict_length_check", True)),
            window_align=bool(dc.get("window_align", True)),
            min_window_identity=float(t),
        )
        kept = len(ds)
        dropped = total - kept
        print(f"t={t:.2f}  kept={kept}  dropped={dropped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dpo/configs/default.yaml")
    ap.add_argument("--thresholds", default="0.85,0.80,0.75,0.70,0.65,0.60")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    thresholds = [float(x) for x in args.thresholds.split(",")]
    run(cfg, thresholds)
