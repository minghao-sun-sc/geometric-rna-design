# dpo/train_lightning.py
from __future__ import annotations
import argparse, yaml, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from dpo.patches import patch_featurizer_three_bead

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

# # NEW: apply compatibility patches before anything builds the featurizer
# from dpo.compat_patches import patch_featurizer_internal_coords
# patch_featurizer_internal_coords()

from dpo.lightning_module import DpoLightningModule
from dpo.lightning_datamodule import DpoDataModule

import torch
torch.set_float32_matmul_precision("high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dpo/configs/default.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--devices", type=int, default=-1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    logger = None
    if args.wandb:
        logger = WandbLogger(project=args.project or cfg["logging"]["project"],
                             name=args.run_name, log_model=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["train"].get("save_dir", "runs/offline_dpo_full"),
        filename="policy-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2, monitor="val/loss", mode="min", save_last=True
    )

    patch_featurizer_three_bead()

    module = DpoLightningModule(cfg)
    datamodule = DpoDataModule(cfg)
    
    # PARAMETER VERIFICATION: Log trainable vs frozen parameters for debugging
    print(f"\n=== PARAMETER VERIFICATION ===")
    trainable_params = []
    frozen_params = []
    lora_params = []
    
    for name, param in module.model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            if "lora" in name.lower() or "A" in name or "B" in name:  # LoRA parameter patterns
                lora_params.append(name)
        else:
            frozen_params.append(name)
    
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")
    print(f"LoRA parameters: {len(lora_params)}")
    
    if len(trainable_params) > 0:
        print(f"Sample trainable params: {trainable_params[:5]}")
    if len(lora_params) > 0:
        print(f"Sample LoRA params: {lora_params[:3]}")
    
    # Verify reference model is completely frozen
    ref_trainable = sum(1 for p in module.ref_model.parameters() if p.requires_grad)
    print(f"Reference model trainable params: {ref_trainable} (should be 0)")
    
    if ref_trainable > 0:
        print("WARNING: Reference model has trainable parameters!")

    # CRITICAL FIX: Use DDPStrategy with find_unused_parameters=True to handle LoRA + reference model
    # This allows DDP to tolerate parameters that don't contribute to loss in every forward pass
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=int(cfg["train"]["rounds"]) * int(cfg["train"]["epochs_per_round"]),
        accumulate_grad_batches=int(cfg["train"]["grad_accum_steps"]),
        gradient_clip_val=float(cfg["train"].get("grad_clip", 1.0)),
        precision=args.precision,
        accelerator="gpu",
        devices=args.devices,
        strategy=ddp_strategy,
        callbacks=[ckpt_cb],
        log_every_n_steps=int(cfg["train"].get("val_every_steps", 100)),
    )
    trainer.fit(module, datamodule=datamodule)

if __name__ == "__main__":
    main()
