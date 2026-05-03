import os, math, json, time, copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dpo.data import build_dataloaders
from dpo.losses import dpo_step_losses, ce_on_sequence, SimPOLoss, step_losses, _LOSS_REGISTRY
from dpo.ref_manager import build_policy_and_reference, save_checkpoint, load_checkpoint
from dpo.utils import WarmupCosine, AverageMeter, now_str
import wandb

# Pareto-DPO Stage-2 support: if cfg.pareto_stage2 is set, the policy is
# upgraded to a WeightConditionedAutoregressiveGNN whose forward takes an
# additional w argument. The reference policy stays as the un-conditioned base.
def _maybe_upgrade_to_stage2(policy, cfg, device):
    """If cfg.pareto_stage2 is true, wrap policy with FiLM head (preserving weights)."""
    if not getattr(cfg, "pareto_stage2", False):
        return policy, False
    from dpo.pareto_dpo import WeightConditionedAutoregressiveGNN
    base_kwargs = dict(
        node_in_dim=tuple(cfg.model.node_in_dim),
        node_h_dim=tuple(cfg.model.node_h_dim),
        edge_in_dim=tuple(cfg.model.edge_in_dim),
        edge_h_dim=tuple(cfg.model.edge_h_dim),
        num_layers=cfg.model.num_layers,
        drop_rate=cfg.model.drop_rate,
        out_dim=cfg.model.out_dim,
    )
    wrapped = WeightConditionedAutoregressiveGNN(base_kwargs=base_kwargs, w_dim=3)
    # Copy the existing policy's weights into the base of the wrapper.
    wrapped.base.load_state_dict(policy.state_dict(), strict=True)
    wrapped = wrapped.to(device)
    return wrapped, True


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

        # Pareto-DPO Stage-2 upgrade: wrap policy with FiLM head if enabled.
        # Must be done AFTER policy/reference are built so we copy current weights
        # into the wrapper. Optimizer is rebuilt on the wrapped model.
        self.policy, self.is_stage2 = _maybe_upgrade_to_stage2(self.policy, cfg, self.device)
        if self.is_stage2:
            print(f"[Pareto-DPO Stage 2] FiLM-conditioned policy active "
                  f"(extra params={sum(p.numel() for p in self.policy.film.parameters())})", flush=True)
            # Rebuild optimizer to include the FiLM head parameters.
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=cfg.optimizer.lr,
                betas=tuple(cfg.optimizer.betas),
                eps=cfg.optimizer.eps,
                weight_decay=cfg.optimizer.weight_decay,
            )

        if cfg.training.compile and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy)

    def train(self):
        cfg = self.cfg
        self.policy.train()
        
        # Track best metrics for HPO
        self.best_metrics = {}
        
        # Support max_steps for HPO
        max_steps = getattr(cfg.training, 'max_steps', float('inf'))

        for epoch in range(cfg.training.epochs):
            meters = defaultdict(AverageMeter)

            for batch in self.train_loader:
                self.global_step += 1
                
                # Early stopping for HPO
                if self.global_step > max_steps:
                    print(f"Reached max_steps ({max_steps}), stopping training")
                    return self.best_metrics
                
                # Move batch to device (batch comes from DataLoader on CPU)
                batch.graph = batch.graph.to(self.device)
                batch.winner_seq = batch.winner_seq.to(self.device)
                batch.loser_seq = batch.loser_seq.to(self.device)

                with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
                    # Loss dispatch — the canonical loss is "dpo"; alternatives
                    # (ipo, kto, pareto_dpo, dpo_is) share the same DPO infrastructure
                    # and the dpo.* hyperparameters (β, sft_lambda, max_len).
                    loss_type = getattr(cfg, "loss_type", "dpo")

                    # Pareto-DPO Stage 2: weight-conditioned policy. Sample w per
                    # batch and call the dedicated step function. Overrides loss_type.
                    if self.is_stage2:
                        from dpo.pareto_dpo import pareto_stage2_step_losses, sample_dirichlet_w
                        w = sample_dirichlet_w(
                            batch_size=1,
                            w_dim=getattr(cfg, "pareto_w_dim", 3),
                            alpha=getattr(cfg, "pareto_dirichlet_alpha", 1.0),
                            device=self.device,
                        )
                        out = pareto_stage2_step_losses(
                            policy_w=self.policy,
                            ref_model=self.reference,
                            batch=batch,
                            w=w,
                            beta=cfg.dpo.beta,
                            max_len=cfg.dpo.max_len,
                        )
                        loss = out["loss_pareto2"]
                    elif loss_type == "dpo":
                        # Preserve original code path (label_smoothing only used by DPO).
                        out = dpo_step_losses(
                            model=self.policy,
                            ref_model=self.reference,
                            batch=batch,
                            beta=cfg.dpo.beta,
                            label_smoothing=cfg.dpo.label_smoothing,
                            max_len=cfg.dpo.max_len
                        )
                        loss = out["loss_dpo"]
                    elif loss_type in _LOSS_REGISTRY:
                        loss, out = step_losses(
                            loss_type,
                            model=self.policy,
                            ref_model=self.reference,
                            batch=batch,
                            beta=cfg.dpo.beta,
                            max_len=cfg.dpo.max_len,
                        )
                        # Add aliases so downstream meters always find canonical keys.
                        if "loss_dpo" not in out:
                            out["loss_dpo"] = loss.detach()
                    else:
                        raise ValueError(
                            f"Unknown loss_type {loss_type!r}; "
                            f"valid: {['dpo', 'simpo'] + list(_LOSS_REGISTRY)}"
                        )

                    # optional SFT on winners (applies to all loss types using DPOTrainer)
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

                # lightweight stdout progress (every 25 steps) for nohup-tailing
                if self.global_step > 0 and (self.global_step % 25) == 0:
                    try:
                        print(
                            f"[ep={epoch} step={self.global_step}] "
                            f"loss={meters['train/loss'].avg:.4f} "
                            f"loss_dpo={meters['train/loss_dpo'].avg:.4f} "
                            f"pref_acc={meters['train/pref_acc'].avg:.3f} "
                            f"margin={meters['train/margin'].avg:.3f} "
                            f"lr={self.optimizer.param_groups[0]['lr']:.2e}",
                            flush=True,
                        )
                    except Exception:
                        pass

                # periodic eval & save
                if (self.global_step % cfg.training.val_every) == 0:
                    val_stats = self.evaluate(self.val_loader, split="val")
                    if wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=self.global_step)

                    # save best by val/pref_acc
                    if val_stats.get("pref_acc", -1.0) > self.best_metric:
                        self.best_metric = val_stats["pref_acc"]
                        self.best_metrics = val_stats  # Store for HPO
                        save_checkpoint(self.save_root, "best", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

                if (self.global_step % cfg.training.save_every) == 0:
                    save_checkpoint(self.save_root, f"steps/step_{self.global_step:07d}", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

            # end epoch
        # save "latest" symlink-ish
        save_checkpoint(self.save_root, "latest", self.policy, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)
        
        return self.best_metrics

    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        cfg = self.cfg
        self.policy.eval()
        agg = {"loss_dpo": 0.0, "pref_acc": 0.0, "margin": 0.0}
        n = 0
        for batch in loader:
            # Move batch to device (batch comes from DataLoader on CPU)
            batch.graph = batch.graph.to(self.device)
            batch.winner_seq = batch.winner_seq.to(self.device)
            batch.loser_seq = batch.loser_seq.to(self.device)
            
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
        
        # Track best metrics for HPO
        self.best_metrics = {}
        
        # Support max_steps for HPO
        max_steps = getattr(cfg.training, 'max_steps', float('inf'))
        
        for epoch in range(cfg.training.epochs):
            meters = defaultdict(AverageMeter)
            
            for batch in self.train_loader:
                self.global_step += 1
                
                # Early stopping for HPO
                if self.global_step > max_steps:
                    print(f"Reached max_steps ({max_steps}), stopping training")
                    return self.best_metrics
                
                # Move batch to device (batch comes from DataLoader on CPU)
                batch.graph = batch.graph.to(self.device)
                batch.winner_seq = batch.winner_seq.to(self.device)
                batch.loser_seq = batch.loser_seq.to(self.device)
                
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
                        self.best_metrics = val_stats  # Store for HPO
                        save_checkpoint(self.save_root, "best", self.policy, self.optimizer, 
                                      self.scheduler, self.global_step, self.best_metric, self.cfg)
                
                if (self.global_step % cfg.training.save_every) == 0:
                    save_checkpoint(self.save_root, f"steps/step_{self.global_step:07d}", 
                                  self.policy, self.optimizer, self.scheduler, 
                                  self.global_step, self.best_metric, self.cfg)
        
        # Save final checkpoint
        save_checkpoint(self.save_root, "latest", self.policy, self.optimizer, 
                       self.scheduler, self.global_step, self.best_metric, self.cfg)
        
        return self.best_metrics
    
    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        cfg = self.cfg
        self.policy.eval()
        agg = defaultdict(float)
        n = 0
        
        for batch in loader:
            # Move batch to device (batch comes from DataLoader on CPU)
            batch.graph = batch.graph.to(self.device)
            batch.winner_seq = batch.winner_seq.to(self.device)
            batch.loser_seq = batch.loser_seq.to(self.device)
            
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
