from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import argparse
import yaml
import wandb
import torch
from types import SimpleNamespace as SN

from dpo.trainer import DPOTrainer


def _to_sn(o):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o


def load_cfg(path: str) -> SN:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_sn(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None, help="Override wandb run name from config")
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"], 
                       help="Override wandb mode from config")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # allow CLI to override wandb settings
    if args.run_name:
        if hasattr(cfg, "wandb"):
            cfg.wandb.run_name = args.run_name
    
    if args.wandb_mode:
        if hasattr(cfg, "wandb"):
            if args.wandb_mode == "disabled":
                cfg.wandb.enable = False
            else:
                cfg.wandb.mode = args.wandb_mode

    # init wandb from config
    if hasattr(cfg, "wandb") and getattr(cfg.wandb, "enable", False):
        wandb_config = {
            "project": getattr(cfg.wandb, "project", "DPO-RNA"),
            "entity": getattr(cfg.wandb, "entity", None),
            "name": getattr(cfg.wandb, "run_name", None),
            "tags": getattr(cfg.wandb, "tags", None),
            "mode": getattr(cfg.wandb, "mode", "online"),
            "notes": getattr(cfg.wandb, "notes", None),
            "group": getattr(cfg.wandb, "group", None),
            "config": {
                # Log key hyperparameters for easy filtering
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.optimizer.lr,
                "dpo_beta": cfg.dpo.beta,
                "sft_lambda": cfg.dpo.sft_lambda,
                "grad_accum_steps": cfg.training.grad_accum_steps,
                "num_workers": cfg.training.num_workers,
                "precision": cfg.training.precision,
            }
        }
        
        wandb.init(**wandb_config)
        print(f"🚀 wandb run: {wandb.run.name} ({wandb.run.id})")
        print(f"📊 View at: {wandb.run.url}")
    else:
        print("📝 wandb disabled - metrics will not be logged")

    trainer = DPOTrainer(cfg)
    trainer.train()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
