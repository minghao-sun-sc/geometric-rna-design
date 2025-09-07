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
            pairs = obj["pairs"]
        else:
            raise ValueError("JSON must be a list or {'pairs': [...]} format.")
    else:
        raise ValueError(f"Unsupported pairs file format: {pairs_path}")
    return pairs


def _maybe_fix_rna_char(c: str) -> str:
    c = c.upper()
    if c == "T":
        return "U"
    if c in {"A", "C", "G", "U", "_"}:
        return c
    return "_"

def _normalize_seq(s: str) -> str:
    return "".join(_maybe_fix_rna_char(c) for c in (s or ""))


def _best_window_mask(graph_seq: str, pair_seq: str, min_identity: float = 0.7):
    """
    Find the best contiguous window in graph_seq for pair_seq.
    Returns (start, identity, mask_list) or None if no window meets min_identity.
    """
    g = _normalize_seq(graph_seq)
    q = _normalize_seq(pair_seq)
    Lg, Lq = len(g), len(q)
    if Lq > Lg or Lq == 0:
        return None
    best_s, best_hit = -1, -1
    for s in range(Lg - Lq + 1):
        hit = 0
        for i in range(Lq):
            a, b = q[i], g[s + i]
            # treat '_' as wildcard (unknown) → not counted as a match
            if a != "_" and b != "_" and a == b:
                hit += 1
        if hit > best_hit:
            best_hit, best_s = hit, s
    if best_s < 0:
        return None
    identity = best_hit / max(1, Lq)
    if identity < float(min_identity):
        return None
    mask = [0] * Lg
    for i in range(Lq):
        mask[best_s + i] = 1
    return best_s, identity, mask


##############################
# Dataset
##############################

class PreferencePairDataset(Dataset):
    """
    Returns (pyg_data, y_w_tokens, y_l_tokens, weight, node_mask_or_none, gid_str).

    New behavior:
    - If len(pair) < len(graph), attempt window alignment and supervise only that window
      via a 0/1 node_mask; y_w/y_l are padded to graph length (outside window not used).
    - If exact length match, mask is None (supervise entire sequence).
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
        window_align: bool = True,
        min_window_identity: float = 0.7,
    ):
        super().__init__()
        self.device = torch.device(device)
        split = split.lower().strip()
        assert split in {"train", "val", "test"}

        # Load processed store and split
        data_dict = torch.load(processed_pt)
        all_raws: List[dict] = list(data_dict.values())
        train_idx, val_idx, test_idx = torch.load(split_file)

        if split == "train":
            idx_map = list(map(int, train_idx))
        elif split == "val":
            idx_map = list(map(int, val_idx))
        else:
            idx_map = list(map(int, test_idx))

        self.raw_list: List[dict] = [all_raws[i] for i in idx_map]
        self._global_to_local: Dict[int, int] = {g: li for li, g in enumerate(idx_map)}

        # Map every canonical id in id_list -> local index
        self._id2local: Dict[str, int] = {}
        for li, raw in enumerate(self.raw_list):
            for it in raw.get("id_list", []):
                cid = canonicalize_id(it)
                if cid and (cid not in self._id2local):
                    self._id2local[cid] = li

        # Featurizer on CPU (avoid CUDA in DataLoader workers)
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
        self.letter_to_num = self.featurizer.letter_to_num
        self._graph_cache: Dict[int, "torch_geometric.data.Data"] = {}

        # Config flags
        self.use_seq_mask = bool(use_seq_mask)
        self.strict_length_check = bool(strict_length_check)
        self.window_align = bool(window_align)
        self.min_window_identity = float(min_window_identity)

        # Load pairs and resolve
        raw_pairs = _load_pairs_any(pairs_path)
        self.pairs: List[dict] = []
        self._dropped_counters = {"unresolved": 0, "len_mismatch": 0, "no_window": 0}

        for p in raw_pairs:
            # Resolve local index
            global_idx = extract_index_from_pair(p)
            local_idx: Optional[int] = None
            if global_idx is not None:
                local_idx = self._global_to_local.get(global_idx, None)
            gid: str = ""
            if local_idx is None:
                gid = extract_backbone_id_from_pair(p)
                local_idx = self._id2local.get(gid, None)
            if local_idx is None:
                self._dropped_counters["unresolved"] += 1
                continue

            raw = self.raw_list[local_idx]
            gL = len(raw.get("sequence", ""))

            wseq = p.get("winner_seq") or p.get("winner") or ""
            lseq = p.get("loser_seq")  or p.get("loser")  or ""
            wL, lL = len(wseq), len(lseq)

            # must be equal-length pair
            if (wL == 0) or (lL == 0) or (wL != lL):
                self._dropped_counters["len_mismatch"] += 1
                continue

            # exact match → keep, no mask
            if wL == gL:
                self.pairs.append({
                    "_local_index": local_idx,
                    "_gid": gid if gid else canonical_from_id_list(raw.get("id_list", [])),
                    "winner_seq": wseq,
                    "loser_seq":  lseq,
                    "weight": float(p.get("weight", 1.0)),
                    "seq_mask": None,
                    "_window": None,
                })
                continue

            # shorter-than-graph → try windowing
            if (wL < gL) and self.window_align:
                # try alignment against the backbone sequence
                gseq = raw.get("sequence", "")
                found = _best_window_mask(gseq, wseq, self.min_window_identity)
                if found is None:
                    self._dropped_counters["no_window"] += 1
                    continue
                start, ident, mask = found
                self.pairs.append({
                    "_local_index": local_idx,
                    "_gid": gid if gid else canonical_from_id_list(raw.get("id_list", [])),
                    "winner_seq": wseq,
                    "loser_seq":  lseq,
                    "weight": float(p.get("weight", 1.0)),
                    "seq_mask": mask,         # 0/1 per node
                    "_window": (start, wL),   # for padding
                })
                continue

            # longer-than-graph or no window allowed → drop
            self._dropped_counters["len_mismatch"] += 1

        kept = len(self.pairs)
        if any(self._dropped_counters.values()):
            print(
                f"[PreferencePairDataset:{split}] kept={kept}, "
                f"dropped_unresolved={self._dropped_counters['unresolved']}, "
                f"dropped_len_mismatch={self._dropped_counters['len_mismatch']}, "
                f"dropped_no_window={self._dropped_counters['no_window']}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    # --------------------------
    # Featurization cache
    # --------------------------
    def _get_graph_by_local(self, li: int):
        if li in self._graph_cache:
            return self._graph_cache[li]
        raw = self.raw_list[li]
        data = self.featurizer.featurize(raw)  # stay on CPU; trainer moves to GPU
        self._graph_cache[li] = data
        return data

    # --------------------------
    # Tokenization
    # --------------------------
    def _encode_seq(self, seq: str) -> torch.Tensor:
        seq = _normalize_seq(seq)
        try:
            tokens = [self.letter_to_num[c] for c in seq]
            # Sanity check: all tokens should be in valid range [0, vocab_size)
            vocab_size = len(self.letter_to_num)
            for i, tok in enumerate(tokens):
                if not (0 <= tok < vocab_size):
                    raise ValueError(f"Invalid token {tok} at position {i} for char '{seq[i]}' (vocab_size={vocab_size})")
            
            return torch.as_tensor(tokens, device=torch.device("cpu"), dtype=torch.long)
        except KeyError as e:
            raise ValueError(f"Character {e} not found in vocabulary {self.letter_to_num}. Sequence: '{seq}'")

    def __getitem__(self, i: int):
        entry = self.pairs[i]
        li = entry["_local_index"]
        data = self._get_graph_by_local(li)

        gL = int(data.seq.numel())  # graph length (tokens in featurizer)
        wseq = entry["winner_seq"]
        lseq = entry["loser_seq"]

        if entry["_window"] is None:
            # exact-length case
            y_w = self._encode_seq(wseq)
            y_l = self._encode_seq(lseq)
            assert y_w.numel() == gL == y_l.numel(), "exact-match length mismatch"
            node_mask = None
        else:
            # windowed case: pad to graph length; mask supervises only the window
            start, Lq = entry["_window"]

            # initialize with '_' tokens (unknown); they won't be used if mask=0
            pad_tok = self.letter_to_num["_"]
            y_w = torch.full((gL,), pad_tok, dtype=torch.long)
            y_l = torch.full((gL,), pad_tok, dtype=torch.long)

            w_tokens = self._encode_seq(wseq)
            l_tokens = self._encode_seq(lseq)
            
            # Use actual sequence lengths instead of stored window length
            # to handle any discrepancies in window calculation
            actual_len_w = w_tokens.numel()
            actual_len_l = l_tokens.numel()
            
            # Ensure we don't exceed the available space
            if start < 0 or start >= gL:
                raise ValueError(f"Invalid window start {start} for graph length {gL}")
            
            end_pos = min(start + actual_len_w, gL)
            copy_len = end_pos - start
            if copy_len > 0:
                y_w[start:end_pos] = w_tokens[:copy_len]
            
            end_pos = min(start + actual_len_l, gL)  
            copy_len = end_pos - start
            if copy_len > 0:
                y_l[start:end_pos] = l_tokens[:copy_len]

            mask_list = entry["seq_mask"]  # list of 0/1
            node_mask = torch.as_tensor(mask_list, dtype=torch.float32)

        weight = torch.tensor(float(entry["weight"]), dtype=torch.float32)
        gid = entry["_gid"]
        return data, y_w, y_l, weight, node_mask, gid


def collate_pairs(batch: List[Tuple[Any, ...]]):
    """Collate function that attaches all data to the PyG batch for DPO training."""
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    
    # Create base PyG batch
    data_batch = Batch.from_data_list(datas)
    
    # Attach DPO-specific data as batch attributes
    data_batch.y_w = torch.cat(ys_w, dim=0)
    data_batch.y_l = torch.cat(ys_l, dim=0) 
    data_batch.weight = torch.stack(ws)
    
    # Handle mixed mask cases: some None (exact match), some tensors (windowed)
    all_masks = []
    for i, mask in enumerate(masks):
        if mask is not None:
            all_masks.append(mask)
        else:
            # For exact match cases, create an all-ones mask of appropriate length
            seq_len = ys_w[i].shape[0]
            all_masks.append(torch.ones(seq_len, dtype=torch.float32))
    
    data_batch.node_mask = torch.cat(all_masks, dim=0)
    
    data_batch.gids = list(gids)
    
    return data_batch  # Single object with all attributes
