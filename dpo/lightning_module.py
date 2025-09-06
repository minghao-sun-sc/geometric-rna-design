import pytorch_lightning as pl
import torch
from dpo.losses import dpo_sft_step
from dpo.ref_manager import RefManager
from src.models import build_model

class DpoLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg}, logger=False)
        self.cfg = cfg
        self.model = build_model(cfg)
        self.ref_manager = RefManager(cfg)

        self.beta = float(cfg["loss"]["beta"])
        self.lambda_sft = float(cfg["loss"]["lambda_sft"])
        self.length_norm = bool(cfg["loss"].get("length_norm", False))

        # log hyperparams
        self.example_input_array = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data_batch, y_w, y_l, w, node_mask, gids = batch
        out = dpo_sft_step(
            model=self.model,
            ref_manager=self.ref_manager,
            data_batch=data_batch,
            y_w=y_w, y_l=y_l,
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
        }, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        data_batch, y_w, y_l, w, node_mask, gids = batch
        out = dpo_sft_step(
            model=self.model,
            ref_manager=self.ref_manager,
            data_batch=data_batch,
            y_w=y_w, y_l=y_l,
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
        # multi-round DPO: update reference after each 'round' worth of epochs
        r = int(self.cfg["train"].get("rounds", 1))
        epr = int(self.cfg["train"].get("epochs_per_round", 1))
        if epr > 0 and (self.current_epoch + 1) % epr == 0:
            self.ref_manager.update_from_policy(self.model)

    def configure_optimizers(self):
        import torch.optim as optim
        cfg = self.cfg
        opt = optim.AdamW(self.model.parameters(),
                          lr=cfg["optim"]["lr"],
                          weight_decay=cfg["optim"]["weight_decay"],
                          betas=tuple(cfg["optim"].get("betas", [0.9, 0.98])))
        if cfg["optim"].get("scheduler","none") == "cosine_wr":
            t0 = int(cfg["optim"].get("cosine_t0_steps", 2000))
            tm = int(cfg["optim"].get("cosine_tmult", 2))
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=tm)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
        return opt
