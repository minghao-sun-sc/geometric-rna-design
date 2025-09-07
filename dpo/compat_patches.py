# dpo/compat_patches.py
from __future__ import annotations

def patch_featurizer_internal_coords():
    """
    Make src.data.featurizer.internal_coords robust to coords with >3 bb atoms
    by trimming to the first three (P, C4', N) before delegating to the
    original function. Avoids editing gRNAde sources.
    """
    import torch
    from src.data import featurizer as _fz
    from src.constants import DISTANCE_EPS as _DISTANCE_EPS

    _orig = _fz.internal_coords

    def _internal_coords_trim3(X: torch.Tensor,
                               C=None,
                               return_masks: bool = False,
                               distance_eps: float = _DISTANCE_EPS):
        # If more than 3 backbone atoms are present, keep the first three.
        if X.dim() != 4:
            return _orig(X, C, return_masks, distance_eps)
        if X.size(2) > 3:
            X = X[:, :, :3, :].contiguous()
        return _orig(X, C, return_masks, distance_eps)

    # swap in our robust version
    _fz.internal_coords = _internal_coords_trim3
