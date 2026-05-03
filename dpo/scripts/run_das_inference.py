#!/usr/bin/env python3
"""Generate DAS test-set designs from a checkpoint."""

import argparse
import sys
from pathlib import Path

import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dpo.bench.eval_full import load_cfg, FullEvalDataset, _load_weights
from dpo.ref_manager import build_model_from_cfg
from dpo.utils import set_seed
from src.constants import NUM_TO_LETTER


def tokens_to_sequence(token_row: torch.Tensor) -> str:
    """Convert a tensor of nucleotide tokens into a string."""
    return "".join(NUM_TO_LETTER[int(tok)] for tok in token_row)


def build_model(model_cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Initialise the Autoregressive gRNAde model and load weights."""
    model = build_model_from_cfg(model_cfg).to(device)
    _load_weights(model, ckpt_path, device)
    model.eval()
    return model


def save_sequences(structure_id: str, native_seq: str, samples: torch.Tensor,
                   ckpt_label: str, temperature: float, out_dir: Path) -> None:
    """Persist native + designed sequences for one structure to FASTA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{structure_id}.fasta"

    records = [SeqRecord(Seq(native_seq), id=f"{structure_id}_native",
                         description="source=native")]

    for idx, seq_tensor in enumerate(samples):
        seq_str = tokens_to_sequence(seq_tensor)
        records.append(SeqRecord(
            Seq(seq_str),
            id=f"{structure_id}_sample{idx}",
            description=f"ckpt={ckpt_label} temperature={temperature:.3f}"
        ))

    SeqIO.write(records, output_path, "fasta")


def main():
    parser = argparse.ArgumentParser(description="Sample DAS designs from a checkpoint")
    parser.add_argument("--config", default="dpo/configs/bench_full.yaml",
                        help="Path to evaluation config with dataset + featurizer params")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint to sample from")
    parser.add_argument("--ckpt-name", required=True, help="Short name for checkpoint (used in FASTA headers)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to place per-structure FASTA files")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of sequences to sample per structure")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split-name", default=None, help="Override split name (train/val/test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of structures for quick tests")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(getattr(cfg, 'device', 'cuda') if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    split_name = args.split_name or getattr(cfg.paths, 'split_name', 'test')
    small_dataset = getattr(cfg.paths, 'small_dataset', False)

    dataset = FullEvalDataset(
        cfg.paths.processed_pt,
        cfg.paths.split_pt,
        split_name,
        cfg.featurizer,
        device="cpu",
        small_dataset=small_dataset
    )

    if args.limit is not None:
        items = dataset[:args.limit]
    else:
        items = dataset

    model = build_model(cfg.model, args.ckpt_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {len(items)} DAS structures from {args.ckpt_name}")
    with torch.no_grad():
        for item in tqdm(items, desc=f"{args.ckpt_name}"):
            graph = item.graph.clone().to(device)
            graph.seq = item.seq.to(device)
            samples = model.sample(graph, args.n_samples, temperature=args.temperature)
            samples = samples.cpu()
            save_sequences(
                structure_id=item.gid,
                native_seq=item.raw_data['sequence'],
                samples=samples,
                ckpt_label=args.ckpt_name,
                temperature=args.temperature,
                out_dir=output_dir
            )

    print(f"Designs written to {output_dir}")


if __name__ == "__main__":
    main()
