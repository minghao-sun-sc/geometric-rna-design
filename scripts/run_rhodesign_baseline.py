"""Generate RhoDesign baseline sequences for the DAS test set.

For each PDB in `data/das_split_raw_data/das_split_raw_pdb/<gid>.pdb` whose
gid appears in the test set (read from FullEvalDataset), runs RhoDesign 8 times
at temperature 0.1 and writes:

    runs/baselines/rhodesign/designs/<gid>/sample{0..7}.fasta

Then `python -m dpo.bench.eval_full --config <cfg> --from_fasta_dir <DIR>` consumes
this directory to produce SSTT metrics matching the rest of the panel.

Run from inside the `grnade` env on a GPU node (uses the standard sample API of
`RhoDesignModel`). RhoDesign's source uses biotite for PDB parsing; biotite is
already installed in `grnade`, so no extra env work is needed.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RHO_SRC = ROOT / "external" / "RhoDesign" / "src"
sys.path.insert(0, str(RHO_SRC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", default=str(ROOT / "data/das_split_raw_data/das_split_raw_pdb"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs/baselines/rhodesign/designs"))
    parser.add_argument("--ckpt", default=str(RHO_SRC.parent / "checkpoint/no_ss_apexp_best.pth"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many structures (smoke test).")
    parser.add_argument("--gids-from", default=None,
                        help="Optional file with one gid per line; default: enumerate the test split via FullEvalDataset")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Get the canonical test gid list — same the SSTT eval iterates over.
    if args.gids_from:
        with open(args.gids_from) as f:
            gids = [g.strip() for g in f if g.strip()]
    else:
        sys.path.insert(0, str(ROOT))
        # Local config to instantiate FullEvalDataset
        from types import SimpleNamespace as SN
        from dpo.bench.eval_full import FullEvalDataset
        # Reuse the same featurizer settings used in bench_full.yaml
        feat = SN(
            top_k=32, num_rbf=32, num_posenc=32,
            num_conformers=1, noise_scale=0.0,
        )
        ds = FullEvalDataset(
            str(ROOT / "data/processed.pt"),
            str(ROOT / "data/das_split.pt"),
            "test",
            feat,
            device="cpu",
            small_dataset=False,
        )
        gids = [item.gid for item in ds.items]
    if args.limit:
        gids = gids[: args.limit]
    print(f"[runner] processing {len(gids)} structures")

    # Import RhoDesign locally (after sys.path insert)
    from RhoDesign_without2d import RhoDesignModel
    from alphabet import Alphabet
    from util import load_structure, extract_coords_from_structure

    class args_class:  # mirrors RhoDesign's inference_without2d.py
        def __init__(self):
            self.local_rank = -1
            self.device_id = [0]
            self.epochs = 100
            self.lr = 1e-5
            self.batch_size = 1
            self.encoder_embed_dim = 512
            self.decoder_embed_dim = 512
            self.dropout = 0.1
            self.gvp_top_k_neighbors = 15
            self.gvp_node_hidden_dim_vector = 256
            self.gvp_node_hidden_dim_scalar = 512
            self.gvp_edge_hidden_dim_scalar = 32
            self.gvp_edge_hidden_dim_vector = 1
            self.gvp_num_encoder_layers = 3
            self.gvp_dropout = 0.1
            self.encoder_layers = 3
            self.encoder_attention_heads = 4
            self.attention_dropout = 0.1
            self.encoder_ffn_embed_dim = 512
            self.decoder_layers = 3
            self.decoder_attention_heads = 4
            self.decoder_ffn_embed_dim = 512

    device = torch.device(args.device)
    dictionary = Alphabet(["A", "G", "C", "U", "X"])
    model = RhoDesignModel(args_class(), dictionary).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[runner] loaded RhoDesign checkpoint: {args.ckpt}")

    pdb_dir = Path(args.pdb_dir)
    n_done = 0
    n_skip = 0
    n_fail = 0
    for gi, gid in enumerate(gids):
        pdb_path = pdb_dir / f"{gid}.pdb"
        if not pdb_path.exists():
            print(f"[runner] missing PDB for {gid} ({pdb_path}); skip")
            n_skip += 1
            continue
        out_dir = out_root / gid
        out_dir.mkdir(parents=True, exist_ok=True)
        # Skip if all 8 samples already exist
        if all((out_dir / f"sample{s}.fasta").exists() for s in range(args.n_samples)):
            n_done += 1
            continue
        try:
            with torch.no_grad():
                pdb = load_structure(str(pdb_path))
                coords, seq = extract_coords_from_structure(pdb)
                for s in range(args.n_samples):
                    # Different seed per sample → diverse stochastic sampling
                    torch.manual_seed(42 + s)
                    np.random.seed(42 + s)
                    random.seed(42 + s)
                    pred = model.sample(coords, device, temperature=args.temperature)
                    out_fp = out_dir / f"sample{s}.fasta"
                    with open(out_fp, "w") as f:
                        f.write(f">{gid}_sample{s}\n{pred}\n")
            n_done += 1
            if (gi + 1) % 10 == 0 or gi < 3:
                print(f"[runner] {gi + 1}/{len(gids)} done={n_done} fail={n_fail} skip={n_skip}")
        except Exception as e:
            print(f"[runner] FAIL on {gid}: {e}")
            n_fail += 1

    print(f"[runner] DONE: {n_done} structures with {args.n_samples} samples each, "
          f"{n_skip} skipped (no PDB), {n_fail} failed")
    print(f"[runner] sequences at: {out_root}")


if __name__ == "__main__":
    main()
