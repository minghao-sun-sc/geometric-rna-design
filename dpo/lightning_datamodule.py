# dpo/lightning_datamodule.py
from __future__ import annotations
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dpo.data import PreferencePairDataset, collate_pairs


class DpoDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = int(cfg["train"]["batch_size"])
        self.num_workers = int(cfg["train"].get("num_workers", 2))
        # pin memory only if CUDA is available
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        dc = self.cfg["data"]

        common_kwargs = dict(
            max_num_conformers=dc["max_num_conformers"],
            radius=dc["radius"],
            top_k=dc["top_k"],
            num_rbf=dc["num_rbf"],
            num_posenc=dc["num_posenc"],
            noise_scale=dc["noise_scale"],
            device="cpu",  # IMPORTANT: keep featurization on CPU
            use_seq_mask=dc.get("use_seq_mask", True),
            strict_length_check=bool(dc.get("strict_length_check", True)),
            window_align=bool(dc.get("window_align", True)),
            min_window_identity=float(dc.get("min_window_identity", 0.7)),
        )

        self.train_ds = PreferencePairDataset(
            processed_pt=dc["processed_pt"],
            split_file=dc["split_file"],
            pairs_path=dc["pairs_path_train"],
            split="train",
            **common_kwargs,
        )
        self.val_ds = PreferencePairDataset(
            processed_pt=dc["processed_pt"],
            split_file=dc["split_file"],
            pairs_path=dc["pairs_path_val"],
            split="val",
            **common_kwargs,
        )

    def train_dataloader(self):
        # persistent_workers must be False if num_workers == 0
        pw = self.num_workers > 0
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=pw,
            collate_fn=collate_pairs,
        )

    def val_dataloader(self):
        pw = self.num_workers > 0
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=pw,
            collate_fn=collate_pairs,
        )
