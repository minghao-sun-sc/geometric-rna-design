#!/usr/bin/env python
# dpo/train_simple.py
"""
Simplified DPO training script.
- No LoRA, trains full model
- Uses exact-match pairs only
- Cleaner implementation without complex masking
"""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

# Setup environment
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Apply patches
from dpo.patches import patch_featurizer_three_bead
patch_featurizer_three_bead()

# Import our simplified modules
from dpo.lightning_module_simple import SimpleDpoLightningModule
from dpo.data_simple import SimplePreferencePairDataset, collate_simple_pairs


class SimpleDataModule(pl.LightningDataModule):
    """Simple data module for exact-match pairs."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["train"]["batch_size"]
        self.num_workers = cfg["train"].get("num_workers", 2)
        
    def setup(self, stage=None):
        """Setup datasets."""
        data_cfg = self.cfg["data"]
        
        # Common dataset arguments
        dataset_kwargs = {
            "processed_pt": data_cfg["processed_pt"],
            "split_file": data_cfg["split_file"],
            "max_num_conformers": data_cfg.get("max_num_conformers", 1),
            "radius": data_cfg.get("radius", 0.0),
            "top_k": data_cfg.get("top_k", 32),
            "num_rbf": data_cfg.get("num_rbf", 32),
            "num_posenc": data_cfg.get("num_posenc", 32),
            "noise_scale": data_cfg.get("noise_scale", 0.1),
            "device": "cpu",
        }
        
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = SimplePreferencePairDataset(
                pairs_path=data_cfg["pairs_path_train"],
                split="train",
                **dataset_kwargs
            )
            
            # Validation dataset
            self.val_dataset = SimplePreferencePairDataset(
                pairs_path=data_cfg["pairs_path_val"],
                split="val",
                **dataset_kwargs
            )
            
            print(f"[DataModule] Train dataset: {len(self.train_dataset)} pairs")
            print(f"[DataModule] Val dataset: {len(self.val_dataset)} pairs")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_simple_pairs,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_simple_pairs,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Simplified DPO Training")
    parser.add_argument("--config", default="dpo/configs/simple_nolora.yaml",
                        help="Path to config file")
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Enable wandb logging (default: True)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--run-name", default=None,
                        help="Name for this run")
    parser.add_argument("--precision", default="bf16-mixed",
                        help="Training precision")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    
    # Setup logging
    loggers = []
    
    # CSV logger (always enabled)
    csv_logger = CSVLogger(
        save_dir=cfg["train"]["save_dir"],
        name="csv_logs"
    )
    loggers.append(csv_logger)
    
    # Wandb logger (optional)
    if not args.no_wandb:
        try:
            project = cfg["logging"]["project"]
            run_name = args.run_name or f"simple_dpo_{pl.seed_everything(workers=True)}"
            
            wandb_logger = WandbLogger(
                project=project,
                name=run_name,
                save_dir=cfg["train"]["save_dir"],
                log_model=False,
            )
            loggers.append(wandb_logger)
            print(f"✅ Wandb logging enabled: {project}/{run_name}")
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            print("   Continuing with CSV logging only")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(cfg["train"]["save_dir"], "checkpoints"),
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_cb)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create model and data module
    print("\n" + "="*60)
    print("Initializing model and data...")
    print("="*60)
    
    model = SimpleDpoLightningModule(cfg)
    datamodule = SimpleDataModule(cfg)
    
    # Print training info
    print("\n" + "="*60)
    print("Training Configuration:")
    print(f"  Model: Full model training (no LoRA)")
    print(f"  Batch size: {cfg['train']['batch_size']}")
    print(f"  Gradient accumulation: {cfg['train']['grad_accum_steps']}")
    print(f"  Effective batch size: {cfg['train']['batch_size'] * cfg['train']['grad_accum_steps']}")
    print(f"  Learning rate: {cfg['optimizer']['lr']}")
    print(f"  Rounds: {cfg['train']['rounds']}")
    print(f"  Epochs per round: {cfg['train']['epochs_per_round']}")
    print(f"  Total epochs: {cfg['train']['rounds'] * cfg['train']['epochs_per_round']}")
    print("="*60 + "\n")
    
    # Setup trainer
    trainer_kwargs = {
        "max_epochs": cfg["train"]["rounds"] * cfg["train"]["epochs_per_round"],
        "accelerator": "gpu",
        "devices": args.devices,
        "precision": args.precision,
        "logger": loggers,
        "callbacks": callbacks,
        "accumulate_grad_batches": cfg["train"]["grad_accum_steps"],
        "gradient_clip_val": cfg["train"].get("grad_clip", 1.0),
        "val_check_interval": cfg["train"].get("val_every_steps", 200),
        "log_every_n_steps": 50,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Use DDP if multiple GPUs
    if args.devices > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(model, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved to: {checkpoint_cb.dirpath}")
    print("="*60)


if __name__ == "__main__":
    main()