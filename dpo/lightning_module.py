# dpo/lightning_module.py
from __future__ import annotations
import torch
import pytorch_lightning as pl
import torch.optim as optim

from dpo.losses import dpo_sft_step
from dpo.ref_manager import RefManager
from dpo.lora import apply_lora, mark_only_lora_as_trainable
from dpo.model_factory import build_model


class DpoLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg}, logger=False)
        self.cfg = cfg

        # Optional: better Tensor Core perf
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # policy
        self.model = build_model(cfg)

        # ---- LoRA (before building reference) ----
        lcfg = cfg.get("lora", {})
        wrapped = 0
        if lcfg.get("enabled", False):
            wrapped = apply_lora(self.model,
                                 enabled=True,
                                 r=int(lcfg.get("r", 8)),
                                 alpha=int(lcfg.get("alpha", 16)),
                                 dropout=float(lcfg.get("dropout", 0.0)),
                                 target_modules=tuple(lcfg.get("target_modules", ["linear","proj","fc","out_proj"])),
                                 train_bias=lcfg.get("train_bias", "none"))
            if wrapped > 0:
                mark_only_lora_as_trainable(self.model, train_bias=lcfg.get("train_bias","none"))
            else:
                # Safety: keep full finetune if nothing got wrapped
                print("[LoRA] WARNING: wrapped=0; keeping full finetune (no freezing).")

        # Final safety: ensure we have trainable parameters
        n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        if n_trainable == 0:
            # unfreeze everything to avoid DDP crash
            for p in self.model.parameters():
                p.requires_grad_(True)
            print("[Safety] No trainable params detected; enabling full finetune.")

        # frozen reference cloned from policy
        self.ref_manager = RefManager(cfg, torch.device("cpu"), policy_model=self.model)

        self.beta = float(cfg["loss"]["beta"])
        self.lambda_sft = float(cfg["loss"]["lambda_sft"])  # mandatory SFT
        self.length_norm = bool(cfg["loss"].get("length_norm", True))

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        data_batch, y_w, y_l, w, node_mask, gids = batch
        out = dpo_sft_step(
            model=self.model,
            ref_manager=self.ref_manager,
            data_batch=data_batch,
            y_w=y_w,
            y_l=y_l,
            node_mask=node_mask,
            weight=w,
            beta=self.beta,
            lambda_sft=self.lambda_sft,
            length_norm=self.length_norm,
            train=True,
        )
        self.log_dict({
            "train/loss": out["loss"],
            "train/loss_dpo": out.get("loss_dpo", torch.nan),
            "train/loss_sft": out.get("loss_sft", torch.nan),
            "train/pref_acc": out.get("pref_acc", torch.nan),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        data_batch, y_w, y_l, w, node_mask, gids = batch
        out = dpo_sft_step(
            model=self.model,
            ref_manager=self.ref_manager,
            data_batch=data_batch,
            y_w=y_w,
            y_l=y_l,
            node_mask=node_mask,
            weight=w,
            beta=self.beta,
            lambda_sft=self.lambda_sft,
            length_norm=self.length_norm,
            train=False,
        )
        self.log_dict({
            "val/loss": out["loss"],
            "val/loss_dpo": out.get("loss_dpo", torch.nan),
            "val/loss_sft": out.get("loss_sft", torch.nan),
            "val/pref_acc": out.get("pref_acc", torch.nan),
        }, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_epoch_end(self):
        epr = int(self.cfg["train"].get("epochs_per_round", 1))
        if epr > 0 and (self.current_epoch + 1) % epr == 0:
            self.ref_manager.update_from_policy(self.model)

    def configure_optimizers(self):
        ocfg = self.cfg["optim"]
        opt = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=float(ocfg["lr"]),
                          weight_decay=float(ocfg["weight_decay"]),
                          betas=tuple(ocfg.get("betas", [0.9, 0.98])))
        if ocfg.get("scheduler", "none") == "cosine_wr":
            t0 = int(ocfg.get("cosine_t0_steps", 2000))
            tm = int(ocfg.get("cosine_tmult", 2))
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=tm)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
        return opt
