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
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # allow CLI to set run name
    if args.run_name:
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "run_name"):
            cfg.wandb.run_name = args.run_name

    # init wandb
    if hasattr(cfg, "wandb") and getattr(cfg.wandb, "enable", False):
        wandb.init(
            project=getattr(cfg.wandb, "project", "DPO-RNA"),
            entity=getattr(cfg.wandb, "entity", None),
            name=getattr(cfg.wandb, "run_name", None),
            tags=getattr(cfg.wandb, "tags", None),
            mode=getattr(cfg.wandb, "mode", "online"),
            config=None,  # we log stepwise metrics elsewhere
        )

    trainer = DPOTrainer(cfg)
    trainer.train()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
