import os, math, json, time, copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dpo.data import build_dataloaders
from dpo.losses import dpo_step_losses, ce_on_sequence, SimPOLoss
from dpo.ref_manager import build_policy_and_reference, save_checkpoint, load_checkpoint
from dpo.utils import WarmupCosine, AverageMeter, now_str
import wandb


class DPOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # data
        self.train_loader, self.val_loader, self.test_loader, self.meta = build_dataloaders(cfg, device=self.device)

        # models
        self.policy, self.reference = build_policy_and_reference(cfg, self.device)

        # opt/sched
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
        total_steps = cfg.training.epochs * len(self.train_loader)
        self.scheduler = WarmupCosine(
            self.optimizer, warmup_steps=cfg.scheduler.warmup_steps,
            total_steps=total_steps, min_lr_ratio=cfg.scheduler.min_lr_ratio
        )

        self.scaler = GradScaler(enabled=cfg.training.precision in ["fp16", "bf16"])
        self.global_step = 0
        self.best_metric = -1.0
        self.save_root = os.path.join(cfg.paths.save_dir, cfg.wandb.run_name or now_str())
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(os.path.join(self.save_root, "steps"), exist_ok=True)

        if cfg.training.compile and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy)

    def train(self):
        cfg = self.cfg
        self.policy.train()

        for epoch in range(cfg.training.epochs):
            meters = defaultdict(AverageMeter)

            for batch in self.train_loader:
                self.global_step += 1

                with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
                    # DPO forward
                    out = dpo_step_losses(
                        model=self.policy,
                        ref_model=self.reference,
                        batch=batch,
                        beta=cfg.dpo.beta,
                        label_smoothing=cfg.dpo.label_smoothing,
                        max_len=cfg.dpo.max_len
                    )
                    loss = out["loss_dpo"]

                    # optional SFT on winners
                    if cfg.dpo.sft_lambda and cfg.dpo.sft_lambda > 0:
                        loss_sft = ce_on_sequence(self.policy, batch.graph, batch.winner_seq, max_len=cfg.dpo.max_len)
                        loss = loss + cfg.dpo.sft_lambda * loss_sft
                        out["loss_sft"] = loss_sft.detach()

                # backward (grad-accum)
                self.scaler.scale(loss / cfg.training.grad_accum_steps).backward()

                if self.global_step % cfg.training.grad_accum_steps == 0:
                    if cfg.optimizer.grad_clip_norm and cfg.optimizer.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.optimizer.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                # meters
                meters["train/loss"].update(loss.detach().item(), 1)
                for k in ["loss_dpo", "pref_acc", "margin"]:
                    meters[f"train/{k}"].update(out[k].item(), 1)
                meters["train/lr"].update(self.optimizer.param_groups[0]["lr"], 1)

                # log every step
                if (self.global_step % cfg.training.log_every) == 0 and wandb.run is not None:
                    log = {k: v.avg for k, v in meters.items()}
                    log["epoch"] = epoch
                    log["step"] = self.global_step
                    wandb.log(log, step=self.global_step)

                # periodic eval & save
                if (self.global_step % cfg.training.val_every) == 0:
                    val_stats = self.evaluate(self.val_loader, split="val")
                    if wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=self.global_step)

                    # save best by val/pref_acc
                    if val_stats.get("pref_acc", -1.0) > self.best_metric:
                        self.best_metric = val_stats["pref_acc"]
                        save_checkpoint(self.save_root, "best", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

                if (self.global_step % cfg.training.save_every) == 0:
                    save_checkpoint(self.save_root, f"steps/step_{self.global_step:07d}", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

            # end epoch
        # save "latest" symlink-ish
        save_checkpoint(self.save_root, "latest", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        cfg = self.cfg
        self.policy.eval()
        agg = {"loss_dpo": 0.0, "pref_acc": 0.0, "margin": 0.0}
        n = 0
        for batch in loader:
            out = dpo_step_losses(
                model=self.policy, ref_model=self.reference,
                batch=batch, beta=cfg.dpo.beta,
                label_smoothing=cfg.dpo.label_smoothing,
                max_len=cfg.dpo.max_len
            )
            # Handle both single and batched data
            if hasattr(batch.graph, 'num_graphs'):
                bs = batch.graph.num_graphs  # Batched graphs
            else:
                bs = 1  # Single graph
            
            agg["loss_dpo"] += out["loss_dpo"].item() * bs
            agg["pref_acc"] += out["pref_acc"].item() * bs
            
            # Handle margin which might be tensor or scalar
            if hasattr(out["margin"], "item"):
                agg["margin"] += out["margin"].item() * bs
            else:
                agg["margin"] += out["margin"] * bs
            n += bs
        self.policy.train()
        return {k: (v / max(1, n)) for k, v in agg.items()}


class SimPOTrainer:
    """SimPO Trainer - reference-free preference optimization."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # data
        self.train_loader, self.val_loader, self.test_loader, self.meta = build_dataloaders(cfg, device=self.device)
        
        # model - only policy, no reference needed for SimPO
        from dpo.ref_manager import build_policy_only
        self.policy = build_policy_only(cfg, self.device)
        
        # SimPO loss
        self.simpo_loss = SimPOLoss(
            beta=cfg.simpo.beta,
            gamma=cfg.simpo.gamma,
            ignore_index=-1  # matches padding value
        )
        
        # optimizer/scheduler
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
        total_steps = cfg.training.epochs * len(self.train_loader)
        self.scheduler = WarmupCosine(
            self.optimizer, warmup_steps=cfg.scheduler.warmup_steps,
            total_steps=total_steps, min_lr_ratio=cfg.scheduler.min_lr_ratio
        )
        
        self.scaler = GradScaler(enabled=cfg.training.precision in ["fp16", "bf16"])
        self.global_step = 0
        self.best_metric = -1.0
        self.save_root = os.path.join(cfg.paths.save_dir, cfg.wandb.run_name or now_str())
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(os.path.join(self.save_root, "steps"), exist_ok=True)
        
        if cfg.training.compile and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy)
    
    def forward_batch(self, graph, seq):
        """Forward pass for a sequence given a graph."""
        # Handle both single and batched graphs
        if hasattr(graph, 'num_graphs'):  # Batched
            # For batched graphs, we need to process each graph-seq pair
            # This is complex due to autoregressive nature - simplify by assuming batch_size=1 for now
            # TODO: Implement proper batched forward pass
            pass
        
        # Set sequence into graph (teacher forcing)
        graph = graph.clone()
        graph.seq = seq
        logits = self.policy.forward(graph)  # [L, 4] or [B, L, 4]
        return logits
    
    def train(self):
        cfg = self.cfg
        self.policy.train()
        
        for epoch in range(cfg.training.epochs):
            meters = defaultdict(AverageMeter)
            
            for batch in self.train_loader:
                self.global_step += 1
                
                with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], 
                             dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
                    
                    # Forward passes for winner and loser
                    if batch.winner_seq.dim() == 2:  # Batched [B, L]
                        # For now, process each sample individually
                        # TODO: Implement proper batched processing
                        graphs = batch.graph.to_data_list()
                        all_win_logits = []
                        all_lose_logits = []
                        
                        for i in range(len(graphs)):
                            win_seq = batch.winner_seq[i]
                            lose_seq = batch.loser_seq[i]
                            
                            # Remove padding
                            win_mask = win_seq != -1
                            lose_mask = lose_seq != -1
                            win_seq = win_seq[win_mask]
                            lose_seq = lose_seq[lose_mask]
                            
                            # Forward passes
                            win_logits = self.forward_batch(graphs[i], win_seq)
                            lose_logits = self.forward_batch(graphs[i], lose_seq)
                            
                            all_win_logits.append(win_logits.unsqueeze(0))
                            all_lose_logits.append(lose_logits.unsqueeze(0))
                        
                        # Combine and pad
                        # For SimPOLoss, we need [B, T, V] format
                        # This is simplified - proper implementation would handle padding better
                        max_len = max(l.size(1) for l in all_win_logits + all_lose_logits)
                        
                        def pad_logits(logits_list, max_len):
                            padded = []
                            for logits in logits_list:
                                if logits.size(1) < max_len:
                                    pad_len = max_len - logits.size(1)
                                    padding = torch.zeros(1, pad_len, logits.size(-1), 
                                                         device=logits.device, dtype=logits.dtype)
                                    logits = torch.cat([logits, padding], dim=1)
                                padded.append(logits)
                            return torch.cat(padded, dim=0)
                        
                        win_logits_batch = pad_logits(all_win_logits, max_len)
                        lose_logits_batch = pad_logits(all_lose_logits, max_len)
                        
                        # Prepare labels with padding
                        win_labels = batch.winner_seq
                        lose_labels = batch.loser_seq
                        
                    else:  # Single sample
                        win_logits_batch = self.forward_batch(batch.graph, batch.winner_seq).unsqueeze(0)
                        lose_logits_batch = self.forward_batch(batch.graph, batch.loser_seq).unsqueeze(0)
                        win_labels = batch.winner_seq.unsqueeze(0)
                        lose_labels = batch.loser_seq.unsqueeze(0)
                    
                    # Compute SimPO loss
                    loss, metrics = self.simpo_loss(win_logits_batch, win_labels, 
                                                    lose_logits_batch, lose_labels)
                    
                    # Optional SFT regularization on winners
                    if cfg.simpo.sft_lambda and cfg.simpo.sft_lambda > 0:
                        # Simple cross-entropy on winner sequences
                        sft_loss = F.cross_entropy(
                            win_logits_batch.view(-1, win_logits_batch.size(-1)),
                            win_labels.view(-1),
                            ignore_index=-1,
                            reduction="mean"
                        )
                        loss = loss + cfg.simpo.sft_lambda * sft_loss
                        metrics["simpo/sft_loss"] = sft_loss.item()
                
                # Backward (with gradient accumulation)
                self.scaler.scale(loss / cfg.training.grad_accum_steps).backward()
                
                if self.global_step % cfg.training.grad_accum_steps == 0:
                    if cfg.optimizer.grad_clip_norm and cfg.optimizer.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.optimizer.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                
                # Update meters
                meters["train/loss"].update(loss.detach().item(), 1)
                for k, v in metrics.items():
                    meters[f"train/{k}"].update(v, 1)
                meters["train/lr"].update(self.optimizer.param_groups[0]["lr"], 1)
                
                # Log every step
                if (self.global_step % cfg.training.log_every) == 0 and wandb.run is not None:
                    log = {k: v.avg for k, v in meters.items()}
                    log["epoch"] = epoch
                    log["step"] = self.global_step
                    wandb.log(log, step=self.global_step)
                
                # Periodic evaluation and save
                if (self.global_step % cfg.training.val_every) == 0:
                    val_stats = self.evaluate(self.val_loader, split="val")
                    if wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=self.global_step)
                    
                    # Save best by validation reward accuracy
                    if val_stats.get("simpo/reward_acc", -1.0) > self.best_metric:
                        self.best_metric = val_stats["simpo/reward_acc"]
                        save_checkpoint(self.save_root, "best", self.policy, self.optimizer, 
                                      self.scheduler, self.global_step, self.best_metric, self.cfg)
                
                if (self.global_step % cfg.training.save_every) == 0:
                    save_checkpoint(self.save_root, f"steps/step_{self.global_step:07d}", 
                                  self.policy, self.optimizer, self.scheduler, 
                                  self.global_step, self.best_metric, self.cfg)
        
        # Save final checkpoint
        save_checkpoint(self.save_root, "latest", self.policy, self.optimizer, 
                       self.scheduler, self.global_step, self.best_metric, self.cfg)
    
    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        cfg = self.cfg
        self.policy.eval()
        agg = defaultdict(float)
        n = 0
        
        for batch in loader:
            # Similar forward logic as training
            if batch.winner_seq.dim() == 2:  # Batched
                # Simplified for now - proper implementation would be more efficient
                bs = batch.graph.num_graphs
                # Process individually for simplicity
                graphs = batch.graph.to_data_list()
                all_win_logits = []
                all_lose_logits = []
                
                for i in range(len(graphs)):
                    win_seq = batch.winner_seq[i]
                    lose_seq = batch.loser_seq[i]
                    
                    # Remove padding
                    win_mask = win_seq != -1
                    lose_mask = lose_seq != -1
                    win_seq = win_seq[win_mask]
                    lose_seq = lose_seq[lose_mask]
                    
                    # Forward passes
                    win_logits = self.forward_batch(graphs[i], win_seq)
                    lose_logits = self.forward_batch(graphs[i], lose_seq)
                    
                    all_win_logits.append(win_logits.unsqueeze(0))
                    all_lose_logits.append(lose_logits.unsqueeze(0))
                
                # Combine (simplified)
                max_len = max(l.size(1) for l in all_win_logits + all_lose_logits)
                
                def pad_logits(logits_list, max_len):
                    padded = []
                    for logits in logits_list:
                        if logits.size(1) < max_len:
                            pad_len = max_len - logits.size(1)
                            padding = torch.zeros(1, pad_len, logits.size(-1), 
                                                 device=logits.device, dtype=logits.dtype)
                            logits = torch.cat([logits, padding], dim=1)
                        padded.append(logits)
                    return torch.cat(padded, dim=0)
                
                win_logits_batch = pad_logits(all_win_logits, max_len)
                lose_logits_batch = pad_logits(all_lose_logits, max_len)
                win_labels = batch.winner_seq
                lose_labels = batch.loser_seq
                
            else:  # Single
                bs = 1
                win_logits_batch = self.forward_batch(batch.graph, batch.winner_seq).unsqueeze(0)
                lose_logits_batch = self.forward_batch(batch.graph, batch.loser_seq).unsqueeze(0)
                win_labels = batch.winner_seq.unsqueeze(0)
                lose_labels = batch.loser_seq.unsqueeze(0)
            
            # Compute loss and metrics
            loss, metrics = self.simpo_loss(win_logits_batch, win_labels, 
                                           lose_logits_batch, lose_labels)
            
            agg["loss"] += loss.item() * bs
            for k, v in metrics.items():
                agg[k] += v * bs
            n += bs
        
        self.policy.train()
        return {k: (v / max(1, n)) for k, v in agg.items()}
