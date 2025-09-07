# dpo/debug/check_edge_dim_once.py
import argparse
import yaml
from torch.utils.data import DataLoader

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from dpo.patches import patch_featurizer_three_bead
from dpo.data import PreferencePairDataset, collate_pairs

def _find_graph_container(x):
    """Return the object that carries edge_s/edge_v, wherever it lives in the batch."""
    try:
        if hasattr(x, "edge_s") and hasattr(x, "edge_v"):
            return x
    except Exception:
        pass
    if isinstance(x, dict):
        for v in x.values():
            g = _find_graph_container(v)
            if g is not None:
                return g
    if isinstance(x, (list, tuple)):
        for v in x:
            g = _find_graph_container(v)
            if g is not None:
                return g
    return None

def main(cfg_path: str):
    # Ensure 3-bead compression (P, C4', N1/N9) BEFORE dataset/featurization runs
    patch_featurizer_three_bead()

    cfg = yaml.safe_load(open(cfg_path))
    dc = cfg["data"]

    ds = PreferencePairDataset(
        processed_pt=dc["processed_pt"],
        split_file=dc["split_file"],
        pairs_path=dc["pairs_path_val"],
        split="val",
        max_num_conformers=dc["max_num_conformers"],
        radius=dc["radius"],
        top_k=dc["top_k"],
        num_rbf=dc["num_rbf"],
        num_posenc=dc["num_posenc"],
        noise_scale=dc["noise_scale"],
        use_seq_mask=dc.get("use_seq_mask", True),
        strict_length_check=dc.get("strict_length_check", True),
        window_align=dc.get("window_align", True),
        min_window_identity=float(dc.get("min_window_identity", 0.75)),
    )

    # Use workers=0 for a deterministic single-batch probe
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_pairs)
    batch = next(iter(dl))

    g = _find_graph_container(batch)
    if g is None:
        print("[WARN] Could not find a graph container with edge_s/edge_v in the batch structure.")
        print("       Batch type/summary:", type(batch))
        if isinstance(batch, (list, tuple)):
            print("       Tuple element types:", [type(x) for x in batch])
        return

    print("edge_s shape:", tuple(g.edge_s.shape), " (expect last dim 131)")
    print("edge_v shape:", tuple(g.edge_v.shape), " (expect last dims (3, 3))")
    ok = (g.edge_s.shape[-1] == 131) and (g.edge_v.shape[-2:] == (3, 3))
    print("[OK]" if ok else "[WARN]", "Edge feature dims",
          "match ARv1 expectation." if ok else "mismatch ARv1 (check patch order).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dpo/configs/default.yaml")
    args = ap.parse_args()
    main(args.config)
