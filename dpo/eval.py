import argparse, yaml
from types import SimpleNamespace as SN
import torch
import wandb

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

from dpo.data import build_dataloaders
from dpo.ref_manager import build_policy_and_reference, load_checkpoint
from dpo.losses import dpo_step_losses
from dpo.utils import set_seed


def load_cfg(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SN(**raw)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.run_name:
        if "wandb" in cfg.__dict__:
            cfg.wandb.run_name = args.run_name

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # loaders
    _, _, test_loader, _ = build_dataloaders(cfg, device=device)

    # model (init from base then load trained ckpt)
    policy, reference = build_policy_and_reference(cfg, device=device)
    if cfg.paths.checkpoint:
        load_checkpoint(cfg.paths.checkpoint, policy, optimizer=None, scheduler=None, device=device)

    if getattr(cfg.eval, "log_to_wandb", False):
        wandb.init(
            project=cfg.eval.wandb.project,
            entity=cfg.eval.wandb.entity,
            name=cfg.eval.wandb.run_name,
            tags=cfg.eval.wandb.get("tags", ["eval"])
        )

    n = 0
    agg = {"loss_dpo": 0.0, "pref_acc": 0.0, "margin": 0.0}
    for batch in test_loader:
        out = dpo_step_losses(
            model=policy, ref_model=reference, batch=batch,
            beta=getattr(cfg, "dpo", SN(beta=0.1)).beta
        )
        for k in agg:
            agg[k] += float(out[k])
        n += 1

    for k in agg:
        agg[k] /= max(1, n)

    print("=== EVAL (test) ===")
    for k, v in agg.items():
        print(f"{k}: {v:.6f}")

    if getattr(cfg.eval, "log_to_wandb", False):
        wandb.log({f"test/{k}": v for k, v in agg.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
