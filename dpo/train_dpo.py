# dpo/train_dpo.py
import argparse, os, yaml, torch, wandb
from torch.utils.data import DataLoader
from dpo.data import PreferencePairDataset, collate_pairs
from dpo.trainer import train_dpo
from dpo.utils import set_seed

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
    ap.add_argument("--config", type=str, default="dpo/configs/default.yaml")
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(f"cuda:{cfg.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")

    if args.wandb and cfg["logging"]["wandb"]:
        wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["run_name"], config=cfg)
    else:
        wandb.init(mode="disabled")

    train_loader, val_loader = build_loaders(cfg, device)
    train_dpo(cfg, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
