from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import argparse
import yaml
import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from types import SimpleNamespace as SN
import os

from dpo.trainer import DPOTrainer
from dpo.data import build_dataloaders
from dpo.ref_manager import build_policy_and_reference


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


def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class DistributedDPOTrainer(DPOTrainer):
    """DPO Trainer with distributed data parallel support."""
    
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Setup device
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")
        cfg.device = f"cuda:{rank}"
        
        # Build distributed data loaders
        self.train_loader, self.val_loader, self.test_loader, self.meta = self._build_distributed_dataloaders(cfg)
        
        # Build models
        self.policy, self.reference = build_policy_and_reference(cfg, self.device)
        
        # Wrap policy with DDP - enable find_unused_parameters for batched processing
        self.policy = DDP(self.policy, device_ids=[rank], find_unused_parameters=True)
        
        # Initialize optimizer and scheduler with DDP model
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
        
        # Adjust total steps for distributed training
        steps_per_epoch = len(self.train_loader) // self.world_size
        total_steps = cfg.training.epochs * steps_per_epoch
        
        from dpo.utils import WarmupCosine
        self.scheduler = WarmupCosine(
            self.optimizer, warmup_steps=cfg.scheduler.warmup_steps,
            total_steps=total_steps, min_lr_ratio=cfg.scheduler.min_lr_ratio
        )
        
        # Other trainer attributes
        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler(enabled=cfg.training.precision in ["fp16", "bf16"])
        self.global_step = 0
        self.best_metric = -1.0
        self.cfg = cfg
        
        # Only rank 0 handles saving and wandb
        if rank == 0:
            from dpo.utils import now_str
            self.save_root = os.path.join(cfg.paths.save_dir, cfg.wandb.run_name or now_str())
            os.makedirs(self.save_root, exist_ok=True)
            os.makedirs(os.path.join(self.save_root, "steps"), exist_ok=True)
        
        if cfg.training.compile and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy)
    
    def _build_distributed_dataloaders(self, cfg):
        """Build data loaders with distributed sampling."""
        # Build datasets first
        from dpo.data import DPOPairDataset
        import types
        
        train_ds = DPOPairDataset(cfg.paths.pairs.train, cfg.paths.processed_pt, cfg.featurizer, split_name="train", device=self.device)
        
        val_featurizer_cfg = types.SimpleNamespace(**vars(cfg.featurizer))
        val_featurizer_cfg.split = "test"
        val_featurizer_cfg.noise_scale = 0.0
        
        test_featurizer_cfg = types.SimpleNamespace(**vars(cfg.featurizer))  
        test_featurizer_cfg.split = "test"
        test_featurizer_cfg.noise_scale = 0.0
        
        val_ds = DPOPairDataset(cfg.paths.pairs.val, cfg.paths.processed_pt, val_featurizer_cfg, split_name="val", device=self.device, id_index=train_ds.id_index)
        test_ds = DPOPairDataset(cfg.paths.pairs.test, cfg.paths.processed_pt, test_featurizer_cfg, split_name="test", device=self.device, id_index=train_ds.id_index)
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        
        # Choose collate function
        from dpo.data import collate_batch_pairs, _collate_identity
        train_collate_fn = collate_batch_pairs if cfg.training.batch_size > 1 else _collate_identity
        val_collate_fn = collate_batch_pairs if cfg.training.batch_size > 1 else _collate_identity
        
        # Build data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.training.batch_size, 
            sampler=train_sampler,
            num_workers=cfg.training.num_workers, 
            pin_memory=cfg.training.pin_memory, 
            collate_fn=train_collate_fn, 
            drop_last=cfg.training.drop_last
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.training.batch_size if cfg.training.batch_size > 1 else 1,
            sampler=val_sampler,
            num_workers=cfg.training.num_workers, 
            pin_memory=cfg.training.pin_memory, 
            collate_fn=val_collate_fn
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=1,
            shuffle=False, 
            num_workers=cfg.training.num_workers, 
            pin_memory=cfg.training.pin_memory, 
            collate_fn=_collate_identity
        )
        
        meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)}
        return train_loader, val_loader, test_loader, meta
    
    def train(self):
        """Override train method to handle distributed training."""
        cfg = self.cfg
        self.policy.train()

        for epoch in range(cfg.training.epochs):
            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)
            
            from collections import defaultdict
            from dpo.utils import AverageMeter
            meters = defaultdict(AverageMeter)

            for batch in self.train_loader:
                self.global_step += 1

                from torch.cuda.amp import autocast
                with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], 
                             dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
                    from dpo.losses import dpo_step_losses, ce_on_sequence
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
                    meters[f"train/{k}"].update(out[k].item() if hasattr(out[k], "item") else out[k], 1)
                meters["train/lr"].update(self.optimizer.param_groups[0]["lr"], 1)

                # Only rank 0 logs and saves
                if self.rank == 0:
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
                            from dpo.ref_manager import save_checkpoint
                            save_checkpoint(self.save_root, "best", self.policy.module, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

                    if (self.global_step % cfg.training.save_every) == 0:
                        from dpo.ref_manager import save_checkpoint
                        save_checkpoint(self.save_root, f"steps/step_{self.global_step:07d}", self.policy.module, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)

        # Save final checkpoint (only rank 0)
        if self.rank == 0:
            from dpo.ref_manager import save_checkpoint
            save_checkpoint(self.save_root, "latest", self.policy.module, self.optimizer, self.scheduler, self.global_step, self.best_metric, self.cfg)


def train_worker(rank, world_size, cfg, run_name):
    """Worker function for distributed training."""
    setup_distributed(rank, world_size)
    
    # Update config for this rank
    cfg.wandb.run_name = run_name
    
    # Only rank 0 initializes wandb
    if rank == 0 and hasattr(cfg, "wandb") and getattr(cfg.wandb, "enable", False):
        wandb.init(
            project=getattr(cfg.wandb, "project", "DPO-RNA"),
            entity=getattr(cfg.wandb, "entity", None),
            name=getattr(cfg.wandb, "run_name", None),
            tags=getattr(cfg.wandb, "tags", None),
            mode=getattr(cfg.wandb, "mode", "online"),
            config=None,
        )
    
    # Create distributed trainer
    trainer = DistributedDPOTrainer(cfg, rank, world_size)
    trainer.train()
    
    if rank == 0 and wandb.run is not None:
        wandb.finish()
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    
    if args.run_name:
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "run_name"):
            cfg.wandb.run_name = args.run_name

    world_size = args.world_size
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Spawn processes for distributed training
    mp.spawn(
        train_worker,
        args=(world_size, cfg, args.run_name),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()