import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dpo.data import PreferencePairDataset, collate_pairs
import torch

class DpoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dc = self.cfg["data"]
        tr = self.cfg["train"]

        self.train_ds = PreferencePairDataset(
            processed_pt=dc["processed_pt"], split_file=dc["split_file"],
            pairs_path=dc["pairs_path_train"], split="train",
            max_num_conformers=dc["max_num_conformers"], radius=dc["radius"],
            top_k=dc["top_k"], num_rbf=dc["num_rbf"], num_posenc=dc["num_posenc"],
            noise_scale=dc["noise_scale"], device="cpu", use_seq_mask=dc.get("use_seq_mask", True)
        )
        self.val_ds = PreferencePairDataset(
            processed_pt=dc["processed_pt"], split_file=dc["split_file"],
            pairs_path=dc["pairs_path_val"], split="val",
            max_num_conformers=dc["max_num_conformers"], radius=dc["radius"],
            top_k=dc["top_k"], num_rbf=dc["num_rbf"], num_posenc=dc["num_posenc"],
            noise_scale=dc["noise_scale"], device="cpu", use_seq_mask=dc.get("use_seq_mask", True)
        )
        self.batch_size = int(tr["batch_size"])
        self.num_workers = int(tr.get("num_workers", 2))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
            collate_fn=collate_pairs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
            collate_fn=collate_pairs
        )
