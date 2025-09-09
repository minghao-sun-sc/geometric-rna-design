# dpo/data_simple.py
"""
Simplified dataset for exact-match pairs only.
No windowing, no masking, no length mismatches.
"""

import json
import torch
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.data.featurizer import RNAGraphFeaturizer


def _load_pairs_jsonl(pairs_path: str) -> List[dict]:
    """Load pairs from JSONL file."""
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            s = line.strip()
            if s:
                pairs.append(json.loads(s))
    return pairs


def _normalize_seq(s: str) -> str:
    """Normalize RNA sequence to uppercase and T->U."""
    s = s.upper().replace('T', 'U')
    # Ensure only valid RNA bases
    valid = set('ACGU')
    return ''.join(c if c in valid else 'A' for c in s)


class SimplePreferencePairDataset(Dataset):
    """
    Simplified dataset for exact-match pairs only.
    
    Returns: (pyg_data, y_w_tokens, y_l_tokens, weight)
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
        max_pairs: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        split = split.lower().strip()
        assert split in {"train", "val", "test"}
        
        # Load processed store and split
        print(f"[SimpleDataset] Loading processed data...")
        data_dict = torch.load(processed_pt)
        all_raws: List[dict] = list(data_dict.values())
        train_idx, val_idx, test_idx = torch.load(split_file)
        
        # Convert to lists if needed
        if not isinstance(train_idx, list):
            train_idx = train_idx.tolist()
        if not isinstance(val_idx, list):
            val_idx = val_idx.tolist()
        if not isinstance(test_idx, list):
            test_idx = test_idx.tolist()
        
        if split == "train":
            idx_map = train_idx
        elif split == "val":
            idx_map = val_idx
        else:
            idx_map = test_idx
        
        self.raw_list: List[dict] = [all_raws[i] for i in idx_map]
        self._global_to_local: Dict[int, int] = {g: li for li, g in enumerate(idx_map)}
        
        # Build PDB name to local index map
        self._pdb_to_local: Dict[str, int] = {}
        for li, raw in enumerate(self.raw_list):
            for id_item in raw.get("id_list", []):
                if isinstance(id_item, str):
                    # Extract PDB code
                    if '/' in id_item:
                        pdb_name = id_item.split('/')[-1].replace('.pdb', '')
                    else:
                        pdb_name = id_item.split('.')[0]
                    
                    if pdb_name and pdb_name not in self._pdb_to_local:
                        self._pdb_to_local[pdb_name] = li
        
        # Featurizer on CPU
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
        self._graph_cache: Dict[int, Any] = {}
        
        # Load pairs
        print(f"[SimpleDataset] Loading pairs from {pairs_path}...")
        raw_pairs = _load_pairs_jsonl(pairs_path)
        
        # Filter for exact matches only
        self.pairs: List[dict] = []
        skipped_unresolved = 0
        skipped_mismatch = 0
        
        for p in raw_pairs:
            # Extract PDB name
            pdb_file = p.get("pdb_file", "")
            if '/' in pdb_file:
                pdb_name = pdb_file.split('/')[-1].replace('.pdb', '')
            else:
                pdb_name = pdb_file.replace('.pdb', '')
            
            # Find local index
            local_idx = self._pdb_to_local.get(pdb_name)
            if local_idx is None:
                skipped_unresolved += 1
                continue
            
            # Check length match
            raw = self.raw_list[local_idx]
            graph_len = len(raw.get("sequence", ""))
            winner_seq = p.get("winner_seq", "")
            loser_seq = p.get("loser_seq", "")
            
            if len(winner_seq) != graph_len or len(loser_seq) != graph_len:
                skipped_mismatch += 1
                continue
            
            # Add to pairs
            self.pairs.append({
                "_local_index": local_idx,
                "_pdb_name": pdb_name,
                "winner_seq": winner_seq,
                "loser_seq": loser_seq,
                "weight": float(p.get("weight", 1.0)),
            })
            
            if max_pairs and len(self.pairs) >= max_pairs:
                break
        
        print(f"[SimpleDataset:{split}] Loaded {len(self.pairs)} exact-match pairs")
        if skipped_unresolved > 0:
            print(f"  Skipped {skipped_unresolved} unresolved PDBs")
        if skipped_mismatch > 0:
            print(f"  Skipped {skipped_mismatch} length mismatches")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _get_graph_by_local(self, li: int):
        """Get featurized graph, with caching."""
        if li in self._graph_cache:
            return self._graph_cache[li]
        
        raw = self.raw_list[li]
        try:
            data = self.featurizer.featurize(raw)
            
            # Validate the graph
            if hasattr(data, 'edge_index') and data.edge_index.numel() == 0:
                print(f"[SimpleDataset] Warning: structure {li} has no edges")
                return None
            
            self._graph_cache[li] = data
            return data
            
        except Exception as e:
            print(f"[SimpleDataset] Error featurizing structure {li}: {e}")
            return None
    
    def _encode_seq(self, seq: str) -> torch.Tensor:
        """Encode sequence to token indices."""
        seq = _normalize_seq(seq)
        
        # Map to vocabulary (A=0, C=1, G=2, U=3)
        vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        tokens = [vocab.get(c, 0) for c in seq]  # Default to A if unknown
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __getitem__(self, i: int):
        """Get a single training example."""
        entry = self.pairs[i]
        li = entry["_local_index"]
        
        # Get graph
        data = self._get_graph_by_local(li)
        if data is None:
            # Skip to next valid sample
            return self.__getitem__((i + 1) % len(self.pairs))
        
        # Encode sequences
        y_w = self._encode_seq(entry["winner_seq"])
        y_l = self._encode_seq(entry["loser_seq"])
        
        # Verify lengths match
        graph_len = data.seq.numel()
        assert y_w.numel() == graph_len, f"Winner length mismatch: {y_w.numel()} vs {graph_len}"
        assert y_l.numel() == graph_len, f"Loser length mismatch: {y_l.numel()} vs {graph_len}"
        
        weight = torch.tensor(entry["weight"], dtype=torch.float32)
        
        return data, y_w, y_l, weight


def collate_simple_pairs(batch: List[Tuple[Any, ...]]):
    """Simple collate function for exact-match pairs."""
    datas, ys_w, ys_l, ws = zip(*batch)
    
    # Create PyG batch
    data_batch = Batch.from_data_list(datas)
    
    # Attach DPO data
    data_batch.y_w = torch.cat(ys_w, dim=0)
    data_batch.y_l = torch.cat(ys_l, dim=0)
    data_batch.weight = torch.stack(ws)
    
    # No masks needed - all sequences match exactly
    data_batch.node_mask = None
    
    return data_batch