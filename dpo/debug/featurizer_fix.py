#!/usr/bin/env python
"""
Analysis and fix for the featurizer patch issue.

Problem: The current patch only compresses coordinates AFTER the featurizer
has already called internal_coords() with the full heavy-atom set.
This causes a dimension mismatch in internal_coords() which expects exactly 3 atoms.

Solution: Move the coordinate compression BEFORE internal_coords is called.
"""

def create_fixed_patch():
    """
    This shows the corrected patch that should be applied to fix the featurizer.
    The key change is to compress coordinates BEFORE the featurizer processes them.
    """
    
    FIXED_PATCH = '''
def patch_featurizer_three_bead():
    """
    Force RNAGraphFeaturizer to down-project heavy-atom coords to 3 beads (P, C4', N1/N9).
    Keeps edge_s dim at 131 (num_rbf=32, num_posenc=32) to match ARv1 checkpoints.
    
    FIXED VERSION: Compresses coordinates BEFORE calling internal_coords.
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

    # Store original methods
    orig_call = feat_mod.RNAGraphFeaturizer.__call__
    orig_featurize = feat_mod.RNAGraphFeaturizer.featurize

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
        idx_stack = torch.stack([idx_P_all, idx_C4p_all, idx_res], dim=1)   # [n_res, 3]
        idx_stack = idx_stack.view(1, n_res, 3, 1).expand(n_conf, n_res, 3, 3)

        X3 = torch.gather(coords_list, dim=2, index=idx_stack)
        return X3

    def patched_featurize(self, rna_dict):
        """
        Patched featurize method that compresses coordinates BEFORE processing.
        This ensures internal_coords gets exactly 3 atoms as expected.
        """
        # Make a copy and compress coordinates if needed
        rna2 = dict(rna_dict)
        
        X = _to_4d_tensor(rna2["coords_list"])
        if X.shape[2] != 3:
            # compress full heavy-atom set to (P, C4', N1/N9) BEFORE featurization
            X3 = _compress_to_three_beads(X, rna2["sequence"])
            rna2["coords_list"] = X3.cpu().numpy()
        else:
            # already 3-bead, just normalize to numpy float32
            rna2["coords_list"] = X.cpu().numpy().astype(np.float32, copy=False)

        # Now call the original featurize with the compressed coordinates
        return orig_featurize(self, rna2)

    # Apply the patch to featurize (not __call__)
    feat_mod.RNAGraphFeaturizer.featurize = patched_featurize
    print("[patches] RNAGraphFeaturizer.featurize patched to 3-bead mode (P, C4', N1/N9).")
    '''
    
    return FIXED_PATCH

if __name__ == "__main__":
    print("ISSUE ANALYSIS:")
    print("=" * 60)
    print("The current patch compresses coordinates in __call__ AFTER featurize")
    print("is called, but featurize calls internal_coords() which expects 3 atoms.")
    print("This causes 'too many values to unpack (expected 3)' error.")
    print()
    print("SOLUTION:")
    print("=" * 60)
    print("Patch the featurize() method instead to compress coordinates")
    print("BEFORE internal_coords() is called.")
    print()
    print("PROPOSED FIX:")
    print("=" * 60)
    print(create_fixed_patch())