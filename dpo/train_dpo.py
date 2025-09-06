# dpo/train_dpo.py
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import argparse, os, yaml, torch, wandb
from torch.utils.data import DataLoader
from dpo.data import PreferencePairDataset, collate_pairs
from dpo.trainer import train_dpo
from dpo.utils import set_seed

import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_loaders(cfg: dict, device: torch.device):
    dcfg = cfg["data"]
    train_ds = PreferencePairDataset(
        processed_pt = dcfg["processed_pt"],
        split_file   = dcfg["split_file"],
        pairs_path   = dcfg["pairs_path_train"],
        split        = "train",
        max_num_conformers=dcfg["max_num_conformers"],
        radius=dcfg["radius"], top_k=dcfg["top_k"],
        num_rbf=dcfg["num_rbf"], num_posenc=dcfg["num_posenc"],
        noise_scale=dcfg["noise_scale"],
        device=str(device),
        use_seq_mask=bool(dcfg.get("use_seq_mask", True)),
    )
    val_ds = PreferencePairDataset(
        processed_pt = dcfg["processed_pt"],
        split_file   = dcfg["split_file"],
        pairs_path   = dcfg["pairs_path_val"],
        split        = "val",
        max_num_conformers=dcfg["max_num_conformers"],
        radius=dcfg["radius"], top_k=dcfg["top_k"],
        num_rbf=dcfg["num_rbf"], num_posenc=dcfg["num_posenc"],
        noise_scale=dcfg["noise_scale"],
        device=str(device),
        use_seq_mask=bool(dcfg.get("use_seq_mask", True)),
    )

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True,
        num_workers=int(tcfg["num_workers"]), pin_memory=True, collate_fn=collate_pairs
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=int(tcfg["num_workers"]), pin_memory=True, collate_fn=collate_pairs
    )
    return train_loader, val_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="dpo/configs/defaults.yaml")
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    # New W&B convenience flags
    ap.add_argument("--run_name", type=str, default=None, help="W&B run name")
    ap.add_argument("--project",  type=str, default=None, help="W&B project (overrides config)")
    ap.add_argument("--entity",   type=str, default=None, help="W&B entity/team (optional)")
    ap.add_argument("--group",    type=str, default=None, help="W&B group (optional)")
    ap.add_argument("--tags",     type=str, default=None, help="Comma-separated tags")
    ap.add_argument("--notes",    type=str, default=None, help="Short notes for the run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(f"cuda:{cfg.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")

    # Ensure logging section exists
    cfg.setdefault("logging", {})
    log_cfg = cfg["logging"]

    # CLI overrides
    if args.run_name is not None: log_cfg["run_name"] = args.run_name
    if args.project  is not None: log_cfg["project"]  = args.project
    if args.entity   is not None: log_cfg["entity"]   = args.entity
    if args.group    is not None: log_cfg["group"]    = args.group
    if args.tags     is not None: log_cfg["tags"]     = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.notes    is not None: log_cfg["notes"]    = args.notes

    if args.wandb and log_cfg.get("project"):
        wandb.init(
            project=log_cfg.get("project"),
            entity=log_cfg.get("entity", None),
            name=log_cfg.get("run_name", None),
            group=log_cfg.get("group", None),
            notes=log_cfg.get("notes", None),
            tags=log_cfg.get("tags", None),
            config=cfg,
            resume="never"  # <-- always start a new run
        )
    else:
        wandb.init(mode="disabled")

    train_loader, val_loader = build_loaders(cfg, device)
    train_dpo(cfg, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
