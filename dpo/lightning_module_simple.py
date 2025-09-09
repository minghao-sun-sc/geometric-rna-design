# dpo/lightning_module_simple.py
"""
Simplified Lightning module for DPO training without LoRA.
Trains the full model directly.
"""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from typing import Any, Dict, Optional

from dpo.losses import dpo_sft_step
from dpo.model_factory import build_model


class SimpleDpoLightningModule(pl.LightningModule):
    """
    Simplified DPO training module without LoRA.
    Trains full model parameters.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        
        # Load base model
        print("[SimpleDPO] Loading base model...")
        # Prepare model config with checkpoint path
        model_cfg = dict(cfg.get("model", {}))
        if "model_checkpoint" in cfg:
            model_cfg["model_path"] = cfg["model_checkpoint"]
        self.model = build_model(model_cfg)
        
        # Create reference model (frozen copy)
        print("[SimpleDPO] Creating reference model...")
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        
        # Store config
        self.cfg = cfg
        self.beta = cfg["loss"]["beta"]
        self.lambda_sft = cfg["loss"]["lambda_sft"]
        self.length_norm = cfg["loss"].get("length_norm", True)
        
        # Learning parameters
        self.lr = cfg["optimizer"]["lr"]
        self.weight_decay = cfg["optimizer"].get("weight_decay", 0.01)
        
        # Round management for reference refresh
        self.current_round = 0
        self.epochs_per_round = cfg["train"]["epochs_per_round"]
        self.rounds = cfg["train"]["rounds"]
        
        # Count parameters
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[SimpleDPO] Model parameters: {n_train:,} trainable / {n_total:,} total")
        
        # Verify reference is frozen
        n_ref_train = sum(p.numel() for p in self.ref_model.parameters() if p.requires_grad)
        assert n_ref_train == 0, f"Reference model has {n_ref_train} trainable params!"
        print(f"[SimpleDPO] Reference model: frozen ({n_total:,} parameters)")
    
    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)
    
    def _compute_loss(self, batch, train: bool = True):
        """Compute DPO + SFT loss."""
        # Move batch to device if needed
        if hasattr(batch, 'to'):
            batch = batch.to(self.device)
        
        # Compute DPO + SFT loss
        out = dpo_sft_step(
            model=self.model,
            ref_model=self.ref_model,
            data_batch=batch,
            y_w=batch.y_w,
            y_l=batch.y_l,
            beta=self.beta,
            lambda_sft=self.lambda_sft,
            node_mask=None,  # No masking needed for exact matches
            weight=batch.weight,
            length_norm=self.length_norm,
        )
        
        return out
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        out = self._compute_loss(batch, train=True)
        
        # Get batch size
        if hasattr(batch, 'num_graphs'):
            batch_size = batch.num_graphs
        elif hasattr(batch, 'batch'):
            batch_size = batch.batch.max().item() + 1
        else:
            batch_size = 1
        
        # Log metrics
        self.log("train/loss", out["loss"], on_step=True, on_epoch=True, 
                 sync_dist=True, batch_size=batch_size)
        self.log("train/dpo_loss", out["dpo_loss"], on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("train/sft_loss", out["sft_loss"], on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("train/advmean", out["advmean"], on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("train/advstd", out["advstd"], on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        
        # Log additional metrics periodically
        if batch_idx % 50 == 0:
            self.log("train/pol_logprob_w", out["pol_logprob_w"], sync_dist=True)
            self.log("train/pol_logprob_l", out["pol_logprob_l"], sync_dist=True)
            self.log("train/ref_logprob_w", out["ref_logprob_w"], sync_dist=True)
            self.log("train/ref_logprob_l", out["ref_logprob_l"], sync_dist=True)
        
        return out["loss"]
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        out = self._compute_loss(batch, train=False)
        
        # Get batch size
        if hasattr(batch, 'num_graphs'):
            batch_size = batch.num_graphs
        elif hasattr(batch, 'batch'):
            batch_size = batch.batch.max().item() + 1
        else:
            batch_size = 1
        
        # Log metrics with on_epoch=True for checkpointing
        self.log("val/loss", out["loss"], on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("val/dpo_loss", out["dpo_loss"], on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("val/sft_loss", out["sft_loss"], on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("val/advmean", out["advmean"], on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        
        return out["loss"]
    
    def on_train_epoch_end(self):
        """Handle end of training epoch."""
        current_epoch = self.current_epoch
        
        # Check if we should refresh reference model
        if (current_epoch + 1) % self.epochs_per_round == 0:
            round_num = (current_epoch + 1) // self.epochs_per_round
            if round_num < self.rounds:
                print(f"\n[SimpleDPO] Epoch {current_epoch + 1}: Refreshing reference model (round {round_num} -> {round_num + 1})")
                
                # Copy current policy to reference
                self.ref_model.load_state_dict(self.model.state_dict())
                
                # Ensure reference remains frozen
                for p in self.ref_model.parameters():
                    p.requires_grad_(False)
                self.ref_model.eval()
                
                self.current_round = round_num
                print(f"[SimpleDPO] Reference model refreshed for round {round_num + 1}")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Simple AdamW optimizer for full model
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )
        
        # Cosine annealing scheduler
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.lr * 0.1,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }