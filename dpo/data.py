import os, json, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as GeometricBatch

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
        self.device = "cpu"  # Always use CPU to avoid CUDA multiprocessing issues
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
        max_attempts = 100  # Significantly increased attempts for robustness
        attempts = 0
        original_idx = idx
        failed_indices = []
        
        # Use different skip strategies to avoid clustered bad data
        skip_patterns = [1, 7, 23, 101, 503, 1009, 2003]  # More prime numbers to avoid patterns
        current_pattern = 0
        
        while attempts < max_attempts:
            try:
                pair = self.pairs[idx]
                cid = canonical_id_from_path(pair["pdb_file"])
                gi = self.id_index[cid]

                graph = self._build_graph_from_entry(gi)
                
                # Check if featurization failed
                if graph is None:
                    print(f"Warning: Skipping index {idx} ({cid}) due to featurization failure (attempt {attempts+1}/{max_attempts})")
                    failed_indices.append((idx, cid, "featurization_failure"))
                    # Use variable skip pattern to avoid clusters
                    skip = skip_patterns[current_pattern % len(skip_patterns)]
                    idx = (idx + skip) % len(self.pairs)
                    attempts += 1
                    current_pattern += 1
                    continue

                # winner/loser sequences -> int tensors (ensure same length as graph.seq)
                def to_int_seq(seq: str):
                    if len(seq) != len(graph.seq):
                        raise ValueError(f"Sequence length mismatch for {cid}: pair={len(seq)} graph={len(graph.seq)}")
                    # Always create on CPU first to avoid CUDA multiprocessing issues
                    return torch.as_tensor([self.letter_to_num[ch] for ch in seq], dtype=torch.long, device="cpu")

                w = to_int_seq(pair["winner_seq"])
                l = to_int_seq(pair["loser_seq"])

                # Move to target device after creation
                return PairBatch(graph=graph.to(self.device), winner_seq=w.to(self.device), loser_seq=l.to(self.device), cid=cid, split=self.split_name)
                
            except ValueError as e:
                if "Sequence length mismatch" in str(e):
                    print(f"Warning: Skipping index {idx} ({cid}) due to length mismatch: {e} (attempt {attempts+1}/{max_attempts})")
                    failed_indices.append((idx, cid, f"length_mismatch: {e}"))
                    # Use variable skip pattern to avoid clusters of same problematic structure
                    skip = skip_patterns[current_pattern % len(skip_patterns)]
                    idx = (idx + skip) % len(self.pairs)
                    attempts += 1
                    current_pattern += 1
                else:
                    print(f"Error: Unexpected ValueError at index {idx} ({cid}): {e}")
                    raise e
            except Exception as e:
                print(f"Error: Unexpected exception at index {idx}: {e}")
                failed_indices.append((idx, "unknown", f"unexpected: {e}"))
                # Use variable skip pattern to avoid clusters
                skip = skip_patterns[current_pattern % len(skip_patterns)]
                idx = (idx + skip) % len(self.pairs)
                attempts += 1
                current_pattern += 1
        
        # If we can't find a valid pair after max_attempts, try a fallback strategy
        print(f"\nWARNING: Could not find a valid pair after {max_attempts} attempts starting from index {original_idx}")
        print(f"Failed indices and reasons:")
        for fidx, fcid, reason in failed_indices[-10:]:  # Show last 10 failures
            print(f"  - Index {fidx} ({fcid}): {reason}")
        
        # Fallback strategy: try to find any valid pair by sampling randomly
        print(f"Attempting fallback strategy: random sampling...")
        import random
        fallback_attempts = 50
        for _ in range(fallback_attempts):
            random_idx = random.randint(0, len(self.pairs) - 1)
            try:
                pair = self.pairs[random_idx]
                cid = canonical_id_from_path(pair["pdb_file"])
                gi = self.id_index[cid]
                graph = self._build_graph_from_entry(gi)
                
                if graph is None:
                    continue
                
                # Check sequence length compatibility
                if len(pair["winner_seq"]) != len(graph.seq) or len(pair["loser_seq"]) != len(graph.seq):
                    continue
                
                def to_int_seq(seq: str):
                    return torch.as_tensor([self.letter_to_num[ch] for ch in seq], dtype=torch.long, device="cpu")
                
                w = to_int_seq(pair["winner_seq"])
                l = to_int_seq(pair["loser_seq"])
                
                print(f"SUCCESS: Fallback found valid pair at index {random_idx} ({cid})")
                return PairBatch(graph=graph.to(self.device), winner_seq=w.to(self.device), loser_seq=l.to(self.device), cid=cid, split=self.split_name)
                
            except Exception as e:
                continue
        
        # If even fallback fails, raise error with more context
        print(f"CRITICAL: Even fallback strategy failed after {fallback_attempts} random attempts")
        print(f"Dataset may be severely corrupted. Total pairs: {len(self.pairs)}")
        raise RuntimeError(f"Could not find a valid pair after {max_attempts} systematic attempts and {fallback_attempts} fallback attempts starting from index {original_idx}. Dataset may be corrupted.")


def _collate_identity(x):
    # we operate one-graph-per-batch (model is naturally per-graph)
    # Keep on CPU - device transfer happens in trainer
    return x[0]


def collate_batch_pairs(batch_list):
    """
    Collate function that properly batches multiple PairBatch objects.
    Uses torch_geometric.data.Batch to combine multiple graphs.
    Keeps data on CPU - device transfer happens in trainer to avoid CUDA multiprocessing issues.
    """
    if len(batch_list) == 1:
        # Single item batch - keep on CPU
        return batch_list[0]
    
    # Extract graphs and sequences from each PairBatch
    graphs = [item.graph for item in batch_list]
    winner_seqs = [item.winner_seq for item in batch_list]
    loser_seqs = [item.loser_seq for item in batch_list]
    cids = [item.cid for item in batch_list]
    splits = [item.split for item in batch_list]
    
    # Batch the graphs using PyTorch Geometric's Batch
    batched_graph = GeometricBatch.from_data_list(graphs)
    
    # Stack sequences - they should all be the same length within a batch
    # If sequences have different lengths, we need to pad them
    max_len = max(seq.size(0) for seq in winner_seqs)
    
    # Pad sequences to max length
    padded_winner = []
    padded_loser = []
    for w, l in zip(winner_seqs, loser_seqs):
        pad_len = max_len - w.size(0)
        if pad_len > 0:
            # Pad with -1 (will be masked in loss calculation)
            w_padded = torch.cat([w, torch.full((pad_len,), -1, dtype=w.dtype, device=w.device)])
            l_padded = torch.cat([l, torch.full((pad_len,), -1, dtype=l.dtype, device=l.device)])
        else:
            w_padded = w
            l_padded = l
        padded_winner.append(w_padded)
        padded_loser.append(l_padded)
    
    # Stack into batch dimension
    winner_seq_batch = torch.stack(padded_winner)  # [B, L]
    loser_seq_batch = torch.stack(padded_loser)    # [B, L]
    
    # Return a batched PairBatch (keep on CPU)
    # Note: We're modifying the PairBatch to hold batched data
    # The graph is now a Batch object, and sequences are [B, L] tensors
    return PairBatch(
        graph=batched_graph,
        winner_seq=winner_seq_batch,
        loser_seq=loser_seq_batch,
        cid=cids,  # List of cids
        split=splits[0] if splits[0] is not None else None
    )


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

    # Use batch collation if batch_size > 1, otherwise use identity collation
    # Keep data on CPU - device transfer happens in trainer
    train_collate_fn = collate_batch_pairs if cfg.training.batch_size > 1 else _collate_identity
    val_collate_fn = collate_batch_pairs if cfg.training.batch_size > 1 else _collate_identity
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,  num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=train_collate_fn, drop_last=cfg.training.drop_last)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size if cfg.training.batch_size > 1 else 1, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=val_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,                         shuffle=False, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, collate_fn=_collate_identity)

    meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)}
    return train_loader, val_loader, test_loader, meta
