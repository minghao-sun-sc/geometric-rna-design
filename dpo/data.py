# dpo/data.py
import os
import json
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.data.featurizer import RNAGraphFeaturizer
from dpo.common_id import (
    canonicalize_id,
    canonical_from_id_list,
    extract_index_from_pair,
    extract_backbone_id_from_pair,
)

##############################
# Helpers
##############################

def _load_pairs_any(pairs_path: str) -> List[dict]:
    """Load pairs from JSONL or JSON (list or {'pairs': [...]})."""
    pairs: List[dict] = []
    if pairs_path.endswith(".jsonl"):
        with open(pairs_path) as f:
            for line in f:
                s = line.strip()
                if s:
                    pairs.append(json.loads(s))
    elif pairs_path.endswith(".json"):
        with open(pairs_path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            pairs = obj
        elif isinstance(obj, dict) and "pairs" in obj:
            assert isinstance(obj["pairs"], list)
            pairs = obj["pairs"]
        else:
            raise ValueError("JSON must be a list or a dict with a 'pairs' key.")
    else:
        raise ValueError(f"Unsupported pairs file format: {pairs_path}")
    return pairs


def _maybe_fix_rna_char(c: str) -> str:
    """Uppercase and map DNA 'T' to RNA 'U'; fall back to '_' for unexpected."""
    c = c.upper()
    if c == "T":
        return "U"
    if c in {"A", "C", "G", "U", "_"}:
        return c
    # unk/base-modded char -> '_'
    return "_"


##############################
# Dataset
##############################

class PreferencePairDataset(Dataset):
    """
    One item per preference pair:
      returns (pyg_data, y_w_tokens, y_l_tokens, weight, node_mask_or_none, gid_str)

    Key behaviors:
    - Uses DAS split via split_file, and only pairs whose backbone falls in the chosen split.
    - Resolves pairs by:
        (1) explicit global index if present (index/idx/graph_index/...)
        (2) otherwise, canonicalized IDs (including 'pdb_file')
      mapping those to the local split index.
    - Reuses gRNAde RNAGraphFeaturizer; caches graphs per (local) index.
    - Winner/loser sequences are tokenized using the featurizer's letter_to_num.
    """
    def __init__(
        self,
        processed_pt: str,
        split_file: str,
        pairs_path: str,
        split: str,  # "train" | "val" | "test"
        max_num_conformers: int = 1,
        radius: float = 0.0,
        top_k: int = 32,
        num_rbf: int = 32,
        num_posenc: int = 32,
        noise_scale: float = 0.1,
        device: str = "cpu",
        use_seq_mask: bool = True,
        strict_length_check: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        split = split.lower().strip()
        assert split in {"train", "val", "test"}

        # ---- Load the full processed store and DAS split indices
        data_dict = torch.load(processed_pt)
        all_raws: List[dict] = list(data_dict.values())
        train_idx, val_idx, test_idx = torch.load(split_file)

        if split == "train":
            idx_map = list(map(int, train_idx))
        elif split == "val":
            idx_map = list(map(int, val_idx))
        else:
            idx_map = list(map(int, test_idx))

        # Keep only raw entries belonging to this split
        self.raw_list: List[dict] = [all_raws[i] for i in idx_map]

        # Build global->local index mapping for this split
        self._global_to_local: Dict[int, int] = {g: li for li, g in enumerate(idx_map)}

        # ---- Build canonical id -> local index map using *all* IDs from id_list
        self._id2local: Dict[str, int] = {}
        for li, raw in enumerate(self.raw_list):
            for it in raw.get("id_list", []):
                cid = canonicalize_id(it)
                if cid and (cid not in self._id2local):
                    self._id2local[cid] = li

        # ---- RNAGraphFeaturizer (identical to gRNAde)
        self.featurizer = RNAGraphFeaturizer(
            split="train" if split == "train" else "test",
            radius=radius,
            top_k=top_k,
            num_rbf=num_rbf,
            num_posenc=num_posenc,
            max_num_conformers=max_num_conformers,
            noise_scale=noise_scale,
            device=torch.device("cpu"),
        )
        self.letter_to_num = self.featurizer.letter_to_num  # {'A','G','C','U','_'}
        self._graph_cache: Dict[int, "torch_geometric.data.Data"] = {}

        # ---- Load pairs and resolve to local indices
        raw_pairs = _load_pairs_any(pairs_path)
        self.pairs: List[dict] = []
        self._dropped_counters = {"unresolved": 0, "len_mismatch": 0}

        for p in raw_pairs:
            # 1) try explicit global index
            global_idx = extract_index_from_pair(p)
            local_idx: Optional[int] = None
            if global_idx is not None:
                local_idx = self._global_to_local.get(global_idx, None)

            # 2) otherwise resolve via canonical id (supports 'pdb_file')
            gid: str = ""
            if local_idx is None:
                gid = extract_backbone_id_from_pair(p)
                local_idx = self._id2local.get(gid, None)

            if local_idx is None:
                self._dropped_counters["unresolved"] += 1
                continue

            # optional strict length check before featurization
            if strict_length_check:
                seq_len = len(self.raw_list[local_idx].get("sequence", ""))
                wseq = p.get("winner_seq") or p.get("winner") or ""
                lseq = p.get("loser_seq") or p.get("loser") or ""
                if (not wseq) or (not lseq) or (len(wseq) != len(lseq)) or (seq_len and len(wseq) != seq_len):
                    self._dropped_counters["len_mismatch"] += 1
                    continue

            # store the resolved/localized pair
            self.pairs.append({
                "_local_index": local_idx,
                "_gid": gid if gid else canonical_from_id_list(self.raw_list[local_idx].get("id_list", [])),
                "winner_seq": p.get("winner_seq") or p.get("winner") or "",
                "loser_seq":  p.get("loser_seq")  or p.get("loser")  or "",
                "weight": float(p.get("weight", 1.0)),
                "seq_mask": p.get("seq_mask", None),  # optional [0/1] list
            })

        kept = len(self.pairs)
        if self._dropped_counters["unresolved"] or self._dropped_counters["len_mismatch"]:
            print(
                f"[PreferencePairDataset:{split}] kept={kept}, "
                f"dropped_unresolved={self._dropped_counters['unresolved']}, "
                f"dropped_len_mismatch={self._dropped_counters['len_mismatch']}"
            )

        self.use_seq_mask = bool(use_seq_mask)

    def __len__(self) -> int:
        return len(self.pairs)

    # --------------------------
    # Featurization cache
    # --------------------------
    def _get_graph_by_local(self, li: int):
        if li in self._graph_cache:
            return self._graph_cache[li]
        raw = self.raw_list[li]
        data = self.featurizer.featurize(raw)
        self._graph_cache[li] = data
        return data

    # --------------------------
    # Tokenization
    # --------------------------
    def _encode_seq(self, seq: str) -> torch.Tensor:
        # Ensure RNA alphabet and map unknowns to '_'
        seq = "".join(_maybe_fix_rna_char(c) for c in (seq or ""))
        return torch.as_tensor(
            [self.letter_to_num[c] for c in seq],
            device=self.device,
            dtype=torch.long
        )

    def __getitem__(self, i: int):
        entry = self.pairs[i]
        li = entry["_local_index"]
        data = self._get_graph_by_local(li)

        y_w = self._encode_seq(entry["winner_seq"])
        y_l = self._encode_seq(entry["loser_seq"])

        # Length consistency — keep this assert; earlier we pre-checked against 'sequence' length
        assert y_w.numel() == y_l.numel() == data.seq.numel(), (
            f"length mismatch at i={i}, local={li}, gid={entry['_gid']}: "
            f"Lw={y_w.numel()} Ll={y_l.numel()} Lg={int(data.seq.numel())}"
        )

        weight = torch.tensor(float(entry["weight"]), device=self.device)
        node_mask = None
        if self.use_seq_mask and (entry.get("seq_mask", None) is not None):
            node_mask = torch.as_tensor(entry["seq_mask"], device=self.device, dtype=torch.float32)
            assert node_mask.numel() == y_w.numel(), "seq_mask length mismatch"

        gid = entry["_gid"]
        return data, y_w, y_l, weight, node_mask, gid


def collate_pairs(batch: List[Tuple[Any, ...]]):
    """
    Collate for PreferencePairDataset.
    Returns: (Batch, y_w, y_l, weights, node_mask|None, gid_list)
    """
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    data_batch = Batch.from_data_list(datas)

    y_w = torch.cat(ys_w, dim=0)
    y_l = torch.cat(ys_l, dim=0)
    w   = torch.stack(ws)  # per-graph weights, #graphs == num examples in this batch

    node_mask = None
    if masks[0] is not None:
        node_mask = torch.cat(masks, dim=0)

    return data_batch, y_w, y_l, w, node_mask, list(gids)
