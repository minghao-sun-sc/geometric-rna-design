# dpo/data.py
import json, os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from src.data.featurizer import RNAGraphFeaturizer

class PreferencePairDataset(Dataset):
    """
    Yields one item per preference pair:
      (pyg_data, y_w_tokens, y_l_tokens, weight, node_mask_or_none, backbone_id)

    - Reuses gRNAde featurizer to ensure identical graph construction.
    - Filters pairs to those present in the selected split (das_split.pt).
    - Supports optional seq_mask per pair (node-wise 0/1).
    """
    def __init__(
        self,
        processed_pt: str,
        split_file: str,
        pairs_path: str,
        split: str,                                # "train" | "val"
        max_num_conformers: int = 1,
        radius: float = 0.0, top_k: int = 32,
        num_rbf: int = 32, num_posenc: int = 32,
        noise_scale: float = 0.1, device: str = "cpu",
        use_seq_mask: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        # Load processed raw list (dict[str->raw]) like gRNAde does
        data_dict = torch.load(processed_pt)
        self._all_raw: List[dict] = list(data_dict.values())

        # Split indices (das)
        train_idx, val_idx, test_idx = torch.load(split_file)
        idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        self.raw_list = [self._all_raw[i] for i in idx_map]

        # backbone_id -> index inside this split's raw_list
        self._idx_by_backbone: Dict[str, int] = {
            raw["backbone_id"]: i for i, raw in enumerate(self.raw_list)
        }

        # Load pairs
        pairs = []
        if pairs_path.endswith(".jsonl"):
            with open(pairs_path) as f:
                for line in f:
                    if line.strip():
                        pairs.append(json.loads(line))
        else:
            raise ValueError("Use JSONL for pairs in prototype.")

        # Keep only pairs whose backbone is in this split
        self.pairs = [p for p in pairs if p["backbone_id"] in self._idx_by_backbone]
        self.use_seq_mask = use_seq_mask

        # Featurizer (identical to gRNAde)
        self.featurizer = RNAGraphFeaturizer(
            split = "train" if split == "train" else "test",
            radius = radius, top_k = top_k,
            num_rbf = num_rbf, num_posenc = num_posenc,
            max_num_conformers = max_num_conformers,
            noise_scale = noise_scale,
            device = self.device,
        )

        # Cache graphs by backbone_id
        self._graph_cache: Dict[str, 'torch_geometric.data.Data'] = {}

        # Token map
        self.letter_to_num = self.featurizer.letter_to_num

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_seq(self, seq: str) -> torch.Tensor:
        return torch.as_tensor(
            [self.letter_to_num[c] for c in seq],
            device=self.device, dtype=torch.long
        )

    def _get_graph(self, backbone_id: str):
        if backbone_id in self._graph_cache:
            return self._graph_cache[backbone_id]
        raw = self.raw_list[self._idx_by_backbone[backbone_id]]
        data = self.featurizer.featurize(raw).to(self.device)
        self._graph_cache[backbone_id] = data
        return data

    def __getitem__(self, i: int):
        p = self.pairs[i]
        gid = p["backbone_id"]
        data = self._get_graph(gid)

        y_w = self._encode_seq(p["winner"])
        y_l = self._encode_seq(p["loser"])
        assert y_w.numel() == data.seq.numel() == y_l.numel(), f"length mismatch for {gid}"

        weight = torch.tensor(float(p.get("weight", 1.0)), device=self.device)

        node_mask = None
        if self.use_seq_mask and ("seq_mask" in p) and (p["seq_mask"] is not None):
            # Expect list[int] len==L
            node_mask = torch.as_tensor(p["seq_mask"], device=self.device, dtype=torch.float32)
            assert node_mask.numel() == y_w.numel(), f"mask length mismatch for {gid}"

        return data, y_w, y_l, weight, node_mask, gid

def collate_pairs(batch):
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    data_batch = Batch.from_data_list(datas)
    y_w = torch.cat(ys_w, dim=0)
    y_l = torch.cat(ys_l, dim=0)
    w   = torch.stack(ws)  # per-graph weights, len == num_graphs
    # Concatenate node masks (if present) in node-order (PyG concatenation)
    node_mask = None
    if masks[0] is not None:
        node_mask = torch.cat(masks, dim=0)
    return data_batch, y_w, y_l, w, node_mask, gids
