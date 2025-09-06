import os, yaml, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()
from dpo.lightning_module import DpoLightningModule
from dpo.lightning_datamodule import DpoDataModule

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dpo/configs/default.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--precision", default="bf16-mixed")  # or "16-mixed" if Ampere+
    ap.add_argument("--devices", type=int, default=-1)    # -1 = all
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    logger = None
    if args.wandb:
        logger = WandbLogger(project=args.project or cfg["logging"]["project"],
                             name=args.run_name,
                             log_model=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["train"].get("save_dir", "runs/offline_dpo_full"),
        filename="policy-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2, monitor="val/loss", mode="min", save_last=True
    )

    module = DpoLightningModule(cfg)
    datamodule = DpoDataModule(cfg)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=int(cfg["train"]["rounds"]) * int(cfg["train"]["epochs_per_round"]),
        accumulate_grad_batches=int(cfg["train"]["grad_accum_steps"]),
        gradient_clip_val=float(cfg["train"].get("grad_clip", 1.0)),
        precision=args.precision,
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        callbacks=[ckpt_cb],
        log_every_n_steps=int(cfg["train"].get("val_every_steps", 100)),
    )
    trainer.fit(module, datamodule=datamodule)

if __name__ == "__main__":
    main()
