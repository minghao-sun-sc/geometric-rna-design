# dpo/patches.py
from typing import List, Optional, Sequence, Union
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]

def _find_index(names_list: List[str], target_names: List[str]) -> Optional[int]:
    """Find first index in names_list matching any of target_names (case & star/apostrophe tolerant)."""
    canon = {i: n.upper().replace('*', "'") for i, n in enumerate(names_list)}
    targets = [t.upper().replace('*', "'") for t in target_names]
    for i, n in canon.items():
        if n in targets:
            return i
    return None

def _to_4d_tensor(coords_list: ArrayLike) -> torch.Tensor:
    """
    Coerce coords_list into shape [n_conf, n_res, n_atoms, 3] as a float32 torch.Tensor.
    Accepts:
      - numpy arrays of shape [..]
      - torch tensors of shape [..]
      - nested lists mixing floats/ndarrays/tensors
    Also adds a conformer dim if missing (i.e., [n_res, n_atoms, 3] -> [1, n_res, n_atoms, 3]).
    """
    if isinstance(coords_list, torch.Tensor):
        X = coords_list.detach()
        if X.dtype != torch.float32:
            X = X.float()
    elif isinstance(coords_list, np.ndarray):
        if coords_list.dtype != np.float32:
            coords_list = coords_list.astype(np.float32, copy=False)
        X = torch.from_numpy(coords_list)
    else:
        # Python list/tuple; normalize by stacking
        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.asarray(x, dtype=np.float32)
        X = _to_numpy(coords_list)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        X = torch.from_numpy(X)

    # Ensure rank is 4: [n_conf, n_res, n_atoms, 3]
    if X.ndim == 3 and X.shape[-1] == 3:
        X = X.unsqueeze(0)  # add conformer dim
    if X.ndim != 4 or X.shape[-1] != 3:
        raise ValueError(f"[three-bead patch] coords_list must be [..., 3] with 3D last axis. Got shape {tuple(X.shape)}")
    return X

def patch_featurizer_three_bead():
    """
    Force RNAGraphFeaturizer to down-project heavy-atom coords to 3 beads (P, C4', N1/N9).
    Keeps edge_s dim at 131 (num_rbf=32, num_posenc=32) to match ARv1 checkpoints.
    """
    import src.data.featurizer as feat_mod
    from src.constants import RNA_ATOMS

    # Resolve canonical atom indices from RNA_ATOMS (robust to C4* vs C4')
    idx_P   = _find_index(RNA_ATOMS, ["P"])
    idx_C4p = _find_index(RNA_ATOMS, ["C4'", "C4*"])
    idx_N1  = _find_index(RNA_ATOMS, ["N1"])
    idx_N9  = _find_index(RNA_ATOMS, ["N9"])
    if None in (idx_P, idx_C4p, idx_N1, idx_N9):
        raise RuntimeError(f"[three-bead patch] Could not resolve indices in RNA_ATOMS: "
                           f"P={idx_P}, C4'={idx_C4p}, N1={idx_N1}, N9={idx_N9}")

    orig_call = feat_mod.RNAGraphFeaturizer.__call__

    def _compress_to_three_beads(coords_list: torch.Tensor, seq: str) -> torch.Tensor:
        """
        coords_list: [n_conf, n_res, n_atoms, 3]  (heavy atoms)
        returns:     [n_conf, n_res, 3,      3]  (P, C4', N1/N9)
        """
        if coords_list.ndim != 4 or coords_list.shape[-1] != 3:
            raise ValueError(f"[three-bead patch] expected 4D coords [..., 3], got {tuple(coords_list.shape)}")

        n_conf, n_res, n_atoms, _ = coords_list.shape

        # Per-residue base-anchor: A/G -> N9, C/U -> N1; default to N1 for others
        sel_N = []
        for ch in seq:
            u = (ch or "").upper()
            sel_N.append(idx_N9 if u in ("A", "G") else idx_N1)

        idx_res = torch.tensor(sel_N, dtype=torch.long, device=coords_list.device)  # [n_res]
        idx_P_all   = torch.full((n_res,), idx_P,   dtype=torch.long, device=coords_list.device)
        idx_C4p_all = torch.full((n_res,), idx_C4p, dtype=torch.long, device=coords_list.device)

        # Build gather index for atom dim (dim=2)
        # Gather expects index same shape as output: [n_conf, n_res, 3, 3]
        idx_stack = torch.stack([idx_P_all, idx_C4p_all, idx_res], dim=1)   # [n_res, 3]
        idx_stack = idx_stack.view(1, n_res, 3, 1).expand(n_conf, n_res, 3, 3)

        X3 = torch.gather(coords_list, dim=2, index=idx_stack)
        return X3

    def patched_call(self, rna_dict):
        # Shallow copy; we only change coords_list
        rna2 = dict(rna_dict)

        X = _to_4d_tensor(rna2["coords_list"])
        if X.shape[2] != 3:
            # compress full heavy-atom set to (P, C4', N1/N9)
            X3 = _compress_to_three_beads(X, rna2["sequence"])
            rna2["coords_list"] = X3.cpu().numpy()  # keep numpy to match original pipeline
        else:
            # already 3-bead, just normalize to numpy float32
            rna2["coords_list"] = X.cpu().numpy().astype(np.float32, copy=False)

        return orig_call(self, rna2)

    feat_mod.RNAGraphFeaturizer.__call__ = patched_call
    print("[patches] RNAGraphFeaturizer.__call__ patched to 3-bead mode (P, C4', N1/N9).")
