import os, json, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from src.data.data_utils import get_backbone_coords
from dpo.utils import load_processed_pt, canonical_id_from_path
from src.data.featurizer import RNAGraphFeaturizer


@dataclass
class PairBatch:
    graph: Any                # torch_geometric.data.Data
    winner_seq: torch.Tensor  # [L]
    loser_seq:  torch.Tensor  # [L]
    cid: str
    split: Optional[str] = None


class DPOPairDataset(Dataset):
    def __init__(self, pairs_path: str, processed_pt_path: str, featurizer_cfg: dict, split_name: str = "train", device="cpu", id_index: Optional[Dict[str,int]] = None):
        super().__init__()
        self.device = device
        self.split_name = split_name
        # Force featurizer to use CPU to avoid device mismatch
        featurizer_device = "cpu"

        # load processed list
        self.processed = load_processed_pt(processed_pt_path)
        # index id->entry
        if id_index is None:
            self.id_index = {}
            for gi, item in enumerate(self.processed):
                for _cid in item["id_list"]:
                    self.id_index[_cid] = gi
        else:
            self.id_index = id_index

        # featurizer (IMPORTANT: same geometry choices as gRNAde.py)
        self.featurizer = RNAGraphFeaturizer(
            split = getattr(featurizer_cfg, "split", "train"),
            radius = getattr(featurizer_cfg, "radius", 0.0),
            top_k = getattr(featurizer_cfg, "top_k", 32),
            num_rbf = getattr(featurizer_cfg, "num_rbf", 32),
            num_posenc = getattr(featurizer_cfg, "num_posenc", 32),
            max_num_conformers = getattr(featurizer_cfg, "max_num_conformers", 1),
            noise_scale = getattr(featurizer_cfg, "noise_scale", 0.1),
            distance_eps = getattr(featurizer_cfg, "distance_eps", 1e-3),
            device = featurizer_device
        )

        # load pairs (jsonl or json)
        if pairs_path.endswith(".jsonl"):
            with open(pairs_path, "r") as f:
                self.pairs = [json.loads(line) for line in f]
        else:
            with open(pairs_path, "r") as f:
                self.pairs = json.load(f)

        # map to processed entry + pre-featurize or keep raw and featurize on the fly
        self.letter_to_num = self.featurizer.letter_to_num

    def __len__(self):
        return len(self.pairs)

    def _build_graph_from_entry(self, entry_idx: int):
        entry = self.processed[entry_idx]
        # convert full-atom coords to 3-bead backbone per conformer
        coords_list = []
        for coords in entry["coords_list"]:
            if isinstance(coords, torch.Tensor):
                coords_list.append(get_backbone_coords(coords.clone().detach(), entry["sequence"]).numpy())
            else:
                coords_list.append(get_backbone_coords(torch.tensor(coords), entry["sequence"]).numpy())
        raw = {
            "sequence": entry["sequence"],
            "coords_list": coords_list,
            "sec_struct_list": entry.get("sec_struct_list", ["."*len(entry["sequence"]) for _ in coords_list]),
        }
        
        try:
            graph = self.featurizer.featurize(raw)
            
            # Additional validation: ensure graph has edges
            if not hasattr(graph, 'edge_index') or graph.edge_index.size(1) == 0:
                cid = entry.get("id_list", ["unknown"])[0] if entry.get("id_list") else "unknown"
                print(f"Warning: Skipping {cid} - generated graph has no edges")
                return None
                
            return graph
            
        except Exception as e:
            cid = entry.get("id_list", ["unknown"])[0] if entry.get("id_list") else "unknown"
            print(f"Warning: Skipping {cid} due to featurization error: {e}")
            return None

    def __getitem__(self, idx: int) -> PairBatch:
        max_attempts = 10  # Avoid infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            try:
                pair = self.pairs[idx]
                cid = canonical_id_from_path(pair["pdb_file"])
                gi = self.id_index[cid]

                graph = self._build_graph_from_entry(gi)
                
                # Check if featurization failed
                if graph is None:
                    print(f"Warning: Skipping {cid} due to featurization failure")
                    # Try next index (with wraparound)
                    idx = (idx + 1) % len(self.pairs)
                    attempts += 1
                    continue

                # winner/loser sequences -> int tensors (ensure same length as graph.seq)
                def to_int_seq(seq: str):
                    if len(seq) != len(graph.seq):
                        raise ValueError(f"Sequence length mismatch for {cid}: pair={len(seq)} graph={len(graph.seq)}")
                    return torch.as_tensor([self.letter_to_num[ch] for ch in seq], dtype=torch.long, device=self.device)

                w = to_int_seq(pair["winner_seq"])
                l = to_int_seq(pair["loser_seq"])

                return PairBatch(graph=graph.to(self.device), winner_seq=w, loser_seq=l, cid=cid, split=self.split_name)
                
            except ValueError as e:
                if "Sequence length mismatch" in str(e):
                    print(f"Warning: Skipping {cid} due to length mismatch: {e}")
                    # Try next index (with wraparound)
                    idx = (idx + 1) % len(self.pairs)
                    attempts += 1
                else:
                    raise e
        
        # If we can't find a valid pair after max_attempts, raise the last error
        raise RuntimeError(f"Could not find a valid pair after {max_attempts} attempts starting from index {idx - attempts}")


def _collate_identity(x):
    # we operate one-graph-per-batch (model is naturally per-graph)
    return x[0]


def build_dataloaders(cfg, device="cpu"):
    id_index = None  # build once for all splits
    train_ds = DPOPairDataset(cfg.paths.pairs.train, cfg.paths.processed_pt, cfg.featurizer, split_name="train", device=device, id_index=id_index)
    id_index = train_ds.id_index
    
    # Create modified featurizer config for val/test
    import types
    val_featurizer_cfg = types.SimpleNamespace(**vars(cfg.featurizer))
    val_featurizer_cfg.split = "test"
    val_featurizer_cfg.noise_scale = 0.0
    
    test_featurizer_cfg = types.SimpleNamespace(**vars(cfg.featurizer))  
    test_featurizer_cfg.split = "test"
    test_featurizer_cfg.noise_scale = 0.0
    
    val_ds   = DPOPairDataset(cfg.paths.pairs.val,   cfg.paths.processed_pt, val_featurizer_cfg, split_name="val", device=device, id_index=id_index)
    test_ds  = DPOPairDataset(cfg.paths.pairs.test,  cfg.paths.processed_pt, test_featurizer_cfg, split_name="test", device=device, id_index=id_index)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,  num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=_collate_identity, drop_last=cfg.training.drop_last)
    val_loader   = DataLoader(val_ds,   batch_size=1,                         shuffle=False, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=_collate_identity)
    test_loader  = DataLoader(test_ds,  batch_size=1,                         shuffle=False, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=_collate_identity)

    meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)}
    return train_loader, val_loader, test_loader, meta
