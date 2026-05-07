"""Generate RIdiffusion (discrete-diffusion + Hyperbolic-GNN) baseline sequences
for the DAS test set.

For each PDB, generates `--n-samples` (default 8) sequences via DDIM sampling
and writes them to:

    runs/baselines/ridiffusion/designs/<gid>/sample{0..7}.fasta

Then `python -m dpo.bench.eval_full --config <cfg> --from_fasta_dir <DIR>`
consumes the directory.

Run from the `grnade` conda env on a GPU node. RIdiffusion's deps are all
present in grnade (torch_geometric, torch_scatter, biotite, ema_pytorch).
The submodule's `seq_generator.py` references `./mean_attr.pt`, but that file
is only used when `if_transform=True`, which is NOT the default — so we don't
need it. Set `--gids-from runs/baselines/test_gids.txt` to enumerate the
canonical 98 test structures.
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
RID_SRC = ROOT / "external" / "RIdiffusion"
sys.path.insert(0, str(RID_SRC))

NT_TYPES = ["A", "U", "G", "C"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", default=str(ROOT / "data/das_split_raw_data/das_split_raw_pdb"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs/baselines/ridiffusion/designs"))
    parser.add_argument("--ckpt", default=str(RID_SRC / "weight/weight3.pt"))
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--ddim-step", type=int, default=100,
                        help="DDIM step size; smaller = more steps = slower but possibly higher quality")
    parser.add_argument("--diverse", action="store_true",
                        help="Use multinomial sampling instead of argmax (default off — argmax-greedy)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gids-from", default=str(ROOT / "runs/baselines/test_gids.txt"))
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.gids_from) as f:
        gids = [g.strip() for g in f if g.strip()]
    if args.limit:
        gids = gids[: args.limit]
    print(f"[runner] processing {len(gids)} structures")

    # Imports must come after sys.path insert
    from RIdiffusion import HEGNN, RIdiffusion
    from generate_graph_ss import pdb2graph
    from torch_geometric.data import Batch, Data
    from ema_pytorch import EMA

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    config["noise_type"] = "uniform"
    print(f"[runner] config: timesteps={config['timesteps']} hidden_dim={config['hidden_dim']}")

    gnn = HEGNN(
        config,
        input_feat_dim=config["input_feat_dim"],
        hidden_channels=config["hidden_dim"],
        edge_attr_dim=config["edge_attr_dim"],
        dropout=config["drop_out"],
        n_layers=config["depth"],
        update_edge=config["update_edge"],
        embedding=config["embedding"],
        embedding_dim=config["embedding_dim"],
        embed_ss=config["embed_ss"],
        norm_feat=config["norm_feat"],
    )
    diffusion = RIdiffusion(model=gnn, config=config, timesteps=config["timesteps"]).to(device)
    diffusion = EMA(diffusion)
    diffusion.load_state_dict(ckpt["ema"])
    diffusion = diffusion.to(device)
    diffusion.ema_model.eval()
    print(f"[runner] loaded RIdiffusion checkpoint: {args.ckpt}")

    def _prepare_graph(data):
        del data["distances"]
        del data["edge_dist"]
        mu_r_norm = data.mu_r_norm
        extra_x = torch.cat([data.x[:, 4:], mu_r_norm], dim=1)
        return Data(
            x=data.x[:, :4],
            extra_x=extra_x,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss=data.ss[: data.x.shape[0], :],
            sasa=data.x[:, 4],
        )

    pdb_dir = Path(args.pdb_dir)
    n_done = n_skip = n_fail = 0
    for gi, gid in enumerate(gids):
        pdb_path = pdb_dir / f"{gid}.pdb"
        if not pdb_path.exists():
            print(f"[runner] missing PDB for {gid}; skip")
            n_skip += 1
            continue
        out_dir = out_root / gid
        out_dir.mkdir(parents=True, exist_ok=True)
        if all((out_dir / f"sample{s}.fasta").exists() for s in range(args.n_samples)):
            n_done += 1
            continue
        try:
            with torch.no_grad():
                graph = pdb2graph(str(pdb_path))
                if graph is None:
                    print(f"[runner] pdb2graph returned None for {gid}; skip")
                    n_fail += 1
                    continue
                input_graph = Batch.from_data_list([_prepare_graph(graph)]).to(device)
                for s in range(args.n_samples):
                    torch.manual_seed(42 + s)
                    np.random.seed(42 + s)
                    random.seed(42 + s)
                    prob, sample_graph = diffusion.ema_model.ddim_sample(
                        input_graph, diverse=args.diverse, step=args.ddim_step
                    )
                    seq = "".join(NT_TYPES[int(idx.item())] for idx in sample_graph.argmax(dim=1))
                    with open(out_dir / f"sample{s}.fasta", "w") as f:
                        f.write(f">{gid}_sample{s}\n{seq}\n")
            n_done += 1
            if (gi + 1) % 10 == 0 or gi < 3:
                print(f"[runner] {gi + 1}/{len(gids)} done={n_done} fail={n_fail} skip={n_skip}")
        except Exception as e:
            print(f"[runner] FAIL on {gid}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"[runner] DONE: {n_done} done, {n_skip} skipped, {n_fail} failed")
    print(f"[runner] sequences at: {out_root}")


if __name__ == "__main__":
    main()
