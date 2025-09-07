# dpo/trainer.py
from __future__ import annotations
import os
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False

from dpo.losses import dpo_sft_step
from dpo.ref_manager import RefManager
from dpo.lora import apply_lora, mark_only_lora_as_trainable
from dpo.utils import AverageMeter, set_seed
from dpo.model_factory import build_model  # <-- use our factory


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _maybe_log(step: int, payload: Dict[str, Any]):
    if _WANDB_OK and wandb.run is not None and _is_rank0():
        wandb.log(payload, step=step)


def _save_ckpt(save_dir: str, model: nn.Module, tag: str):
    if not _is_rank0():
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"policy_{tag}.pt")
    torch.save(model.state_dict(), path)
    return path


def _build_optimizer(cfg, model: nn.Module):
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"]["weight_decay"])
    betas = tuple(cfg["optim"].get("betas", [0.9, 0.98]))
    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr, weight_decay=wd, betas=betas)


def _build_scheduler(cfg, optimizer: optim.Optimizer):
    sched_name = cfg["optim"].get("scheduler", "none")
    if sched_name == "cosine_wr":
        t0 = int(cfg["optim"].get("cosine_t0_steps", 2000))
        tm = int(cfg["optim"].get("cosine_tmult", 2))
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=tm)
    return None


def _move_to_device(batch_tuple, device: torch.device):
    data_batch, y_w, y_l, w, node_mask, gids = batch_tuple
    data_batch = data_batch.to(device, non_blocking=True)
    y_w        = y_w.to(device, non_blocking=True)
    y_l        = y_l.to(device, non_blocking=True)
    w          = w.to(device, non_blocking=True)
    if node_mask is not None:
        node_mask = node_mask.to(device, non_blocking=True)
    return (data_batch, y_w, y_l, w, node_mask, gids)


def evaluate_loop(model: nn.Module,
                  ref_manager: RefManager,
                  val_loader: DataLoader,
                  device: torch.device,
                  cfg) -> Dict[str, float]:
    model.eval()
    meters = {k: AverageMeter() for k in ["loss", "loss_dpo", "loss_sft", "pref_acc"]}

    beta = float(cfg["loss"]["beta"])
    lambda_sft = float(cfg["loss"]["lambda_sft"])  # mandatory SFT
    length_norm = bool(cfg["loss"].get("length_norm", True))

    with torch.no_grad():
        for batch_tuple in val_loader:
            data_batch, y_w, y_l, w, node_mask, gids = _move_to_device(batch_tuple, device)

            out = dpo_sft_step(
                model=model,
                ref_manager=ref_manager,
                data_batch=data_batch,
                y_w=y_w,
                y_l=y_l,
                node_mask=node_mask,
                weight=w,
                beta=beta,
                lambda_sft=lambda_sft,
                length_norm=length_norm,
                train=False,
            )
            for k, meter in meters.items():
                v = out.get(k, None)
                if v is not None:
                    meter.update(float(v), n=1)

    return {k: v.avg for k, v in meters.items()}


def train_dpo(cfg: Dict[str, Any],
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: torch.device):

    set_seed(int(cfg.get("seed", 42)))

    # ---- Build policy ----
    model = build_model(cfg).to(device)

    # ---- LoRA before optimizer + reference ----
    lcfg = cfg.get("lora", {})
    wrapped = 0
    if lcfg.get("enabled", False):
        wrapped = apply_lora(model,
                             enabled=True,
                             r=int(lcfg.get("r", 8)),
                             alpha=int(lcfg.get("alpha", 16)),
                             dropout=float(lcfg.get("dropout", 0.0)),
                             target_modules=tuple(lcfg.get("target_modules", ["linear","proj","fc","out_proj"])),
                             train_bias=lcfg.get("train_bias", "none"))
        if wrapped > 0:
            mark_only_lora_as_trainable(model, train_bias=lcfg.get("train_bias","none"))
        else:
            print("[LoRA] WARNING: wrapped=0; keeping full finetune.")
        
        # final safety
        if sum(p.requires_grad for p in model.parameters()) == 0:
            for p in model.parameters(): p.requires_grad_(True)
            print("[Safety] No trainable params; enabling full finetune.")

        if _WANDB_OK and wandb.run is not None and _is_rank0():
            wandb.config.update({"lora_wrapped": wrapped}, allow_val_change=True)

    # ---- Frozen reference cloned from current policy ----
    ref_manager = RefManager(cfg, device, policy_model=model)

    # ---- Optimizer & Scheduler ----
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    rounds = int(cfg["train"].get("rounds", 1))
    epochs_per_round = int(cfg["train"].get("epochs_per_round", 1))
    val_every_steps = int(cfg["train"].get("val_every_steps", 100))
    ckpt_every_steps = int(cfg["train"].get("ckpt_every_steps", 0))
    save_dir = cfg["train"].get("save_dir", "runs/offline_dpo_full")

    beta = float(cfg["loss"]["beta"])
    lambda_sft = float(cfg["loss"]["lambda_sft"])
    length_norm = bool(cfg["loss"].get("length_norm", True))

    global_step = 0
    best_val = math.inf
    best_path: Optional[str] = None

    if _WANDB_OK and wandb.run is not None and _is_rank0():
        wandb.config.update({"global_batch_size": train_loader.batch_size * grad_accum}, allow_val_change=True)

    for r in range(rounds):
        for epoch in range(epochs_per_round):
            model.train()
            loss_meter = AverageMeter()
            dpo_meter = AverageMeter()
            sft_meter = AverageMeter()
            acc_meter = AverageMeter()

            optimizer.zero_grad(set_to_none=True)

            for step, batch_tuple in enumerate(train_loader):
                data_batch, y_w, y_l, w, node_mask, gids = _move_to_device(batch_tuple, device)

                out = dpo_sft_step(
                    model=model,
                    ref_manager=ref_manager,
                    data_batch=data_batch,
                    y_w=y_w,
                    y_l=y_l,
                    node_mask=node_mask,
                    weight=w,
                    beta=beta,
                    lambda_sft=lambda_sft,
                    length_norm=length_norm,
                    train=True,
                )

                loss = out["loss"] / grad_accum
                loss.backward()

                loss_meter.update(float(out["loss"]))
                if out.get("loss_dpo") is not None:
                    dpo_meter.update(float(out["loss_dpo"]))
                if out.get("loss_sft") is not None:
                    sft_meter.update(float(out["loss_sft"]))
                if out.get("pref_acc") is not None:
                    acc_meter.update(float(out["pref_acc"]))

                if (step + 1) % grad_accum == 0:
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                if (global_step % val_every_steps) == 0:
                    _maybe_log(global_step, {
                        "train/loss": loss_meter.avg,
                        "train/loss_dpo": dpo_meter.avg,
                        "train/loss_sft": sft_meter.avg,
                        "train/pref_acc": acc_meter.avg,
                        "lr": optimizer.param_groups[0]["lr"],
                        "round": r,
                        "epoch_in_round": epoch,
                    })

                if (global_step > 0) and (global_step % val_every_steps == 0):
                    val_metrics = evaluate_loop(model, ref_manager, val_loader, device, cfg)
                    _maybe_log(global_step, {f"val/{k}": v for k, v in val_metrics.items()})
                    if val_metrics.get("loss", math.inf) < best_val and _is_rank0():
                        best_val = val_metrics["loss"]
                        best_path = _save_ckpt(save_dir, model, f"best_step{global_step}")

                if ckpt_every_steps and (global_step > 0) and (global_step % ckpt_every_steps == 0):
                    _save_ckpt(save_dir, model, f"step{global_step}")

                global_step += 1

            val_metrics = evaluate_loop(model, ref_manager, val_loader, device, cfg)
            _maybe_log(global_step, {f"val/{k}": v for k, v in val_metrics.items()})
            if val_metrics.get("loss", math.inf) < best_val and _is_rank0():
                best_val = val_metrics["loss"]
                best_path = _save_ckpt(save_dir, model, f"best_r{r}_e{epoch}")

        # NEW REFERENCE FOR NEXT ROUND
        ref_manager.update_from_policy(model)

    final_path = _save_ckpt(save_dir, model, "final")
    if _WANDB_OK and wandb.run is not None and _is_rank0():
        if best_path:
            wandb.run.summary["best_ckpt"] = best_path
        if final_path:
            wandb.run.summary["final_ckpt"] = final_path
        wandb.run.summary["best_val_loss"] = best_val
