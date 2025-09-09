"""
Full evaluation pipeline for DPO-RNA models including all gRNAde metrics:
- Sequence Recovery (teacher-forced)
- Perplexity
- 2D Self-consistency (EternaFold)
- 3D Self-consistency (RhoFold: RMSD, TM-score, GDT)
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import math
import csv
import time
import yaml
import copy
import shutil
from types import SimpleNamespace as SN
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.classification import binary_matthews_corrcoef

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import wandb

from src.data.featurizer import RNAGraphFeaturizer
from src.data.data_utils import get_backbone_coords
from src.constants import NUM_TO_LETTER, RMSD_THRESHOLD, TM_THRESHOLD, GDT_THRESHOLD

from dpo.ref_manager import build_model_from_cfg
from dpo.utils import set_seed, load_processed_pt

# Import existing self-consistency functions directly from evaluator
from src.evaluator import (
    self_consistency_score_eternafold,
    self_consistency_score_rhofold
)


def _to_sn(o):
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(x) for x in o]
    return o


def load_cfg(path: str) -> SN:
    with open(path, "r") as f:
        return _to_sn(yaml.safe_load(f))


def _load_split(path: str):
    tr, va, te = torch.load(path, map_location="cpu")
    return list(map(int, tr)), list(map(int, va)), list(map(int, te))


@dataclass
class GraphItem:
    gid: str
    graph: Any    # torch_geometric.data.Data
    seq: torch.Tensor  # [L] long 0..3
    raw_data: dict  # Original raw data for self-consistency evaluation


class FullEvalDataset(Dataset):
    """Dataset for full evaluation including self-consistency metrics."""
    
    def __init__(self, processed_pt: str, split_pt: str, split_name: str, feat_cfg: SN, device="cpu"):
        super().__init__()
        self.device = device
        all_items = load_processed_pt(processed_pt)
        tr, va, te = _load_split(split_pt)
        
        if split_name == "train":
            idxs = tr
        elif split_name == "val":
            idxs = va
        else:
            idxs = te

        self.items: List[GraphItem] = []
        
        # Force featurizer to use CPU to avoid device mismatch issues
        featurizer_device = "cpu"
        self.featurizer = RNAGraphFeaturizer(
            split=feat_cfg.split,
            radius=feat_cfg.radius, 
            top_k=feat_cfg.top_k,
            num_rbf=feat_cfg.num_rbf, 
            num_posenc=feat_cfg.num_posenc,
            max_num_conformers=feat_cfg.max_num_conformers,
            noise_scale=feat_cfg.noise_scale,
            distance_eps=getattr(feat_cfg, "distance_eps", 1e-3),
            device=featurizer_device
        )
        
        for gi in idxs:
            entry = all_items[gi]
            # Build coords_list (3-bead backbone) for each conformer
            coords_list = []
            for coords in entry["coords_list"]:
                if isinstance(coords, torch.Tensor):
                    coords_list.append(get_backbone_coords(coords.clone().detach(), entry["sequence"]))
                else:
                    coords_list.append(get_backbone_coords(torch.tensor(coords), entry["sequence"]))
            
            # Create raw data for featurization (numpy arrays for featurizer)
            raw_for_featurizer = {
                "sequence": entry["sequence"],
                "coords_list": [coords.numpy() for coords in coords_list],
                "sec_struct_list": entry.get("sec_struct_list", ["."*len(entry["sequence"]) for _ in coords_list])
            }
            graph = self.featurizer.featurize(raw_for_featurizer).to(device)
            
            # Create raw data for evaluator functions (torch tensors)
            raw_data = {
                "sequence": entry["sequence"],
                "coords_list": coords_list,  # Keep as torch tensors
                "sec_struct_list": entry.get("sec_struct_list", ["."*len(entry["sequence"]) for _ in coords_list]),
                # Include other fields that might be needed
                "id_list": entry.get("id_list", [f"idx_{gi}"]),
                "rfam_list": entry.get("rfam_list", ["unknown"]),
                "eq_class_list": entry.get("eq_class_list", ["unknown"]),
                "cluster_structsim0.45": entry.get("cluster_structsim0.45", "unknown")
            }
            
            # Use the graph's sequence to ensure length consistency
            seq = graph.seq.to(device)
            gid = entry["id_list"][0] if entry.get("id_list") else f"idx_{gi}"
            
            self.items.append(GraphItem(gid=gid, graph=graph, seq=seq, raw_data=raw_data))

    def __len__(self): 
        return len(self.items)
    
    def __getitem__(self, i): 
        return self.items[i]


def _load_weights(model, ckpt_path, device):
    """Load model weights from checkpoint."""
    sd = torch.load(ckpt_path, map_location=device)
    # Two formats: (A) state_dict directly; (B) {"model": state_dict, ...}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return model


@torch.no_grad()
def eval_full_metrics(
    cfg: SN, 
    ds: FullEvalDataset, 
    ckpt_name: str, 
    ckpt_path: str, 
    device,
    n_samples: int = 8,
    temperature: float = 1.0,
    metrics: List[str] = ['recovery', 'perplexity', 'sc_eternafold', 'sc_rhofold'],
    save_designs: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Full evaluation with all gRNAde metrics including sampling and self-consistency.
    
    Args:
        cfg: Configuration
        ds: Dataset
        ckpt_name: Checkpoint name for logging
        ckpt_path: Path to checkpoint
        device: Device to run on
        n_samples: Number of sequences to sample per structure
        temperature: Sampling temperature
        metrics: List of metrics to compute
        save_designs: Whether to save designed sequences
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of metric values
    """
    
    model = build_model_from_cfg(cfg.model).to(device)
    _load_weights(model, ckpt_path, device)
    model.eval()
    
    # Initialize RhoFold if needed (following src/evaluator.py pattern)
    rhofold = None
    current_datetime = None
    if 'sc_rhofold' in metrics:
        try:
            from tools.rhofold.rf import RhoFold
            from tools.rhofold.config import rhofold_config
            from src.constants import PROJECT_PATH
            from datetime import datetime
            
            # Initialize RhoFold for 3D self-consistency score
            rhofold = RhoFold(rhofold_config, device)
            rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
            print(f"Loading RhoFold checkpoint: {rhofold_path}")
            rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
            # Transfer model to device in eval mode
            rhofold = rhofold.to(device)
            rhofold.eval()
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        except Exception as e:
            print(f"Failed to initialize RhoFold: {e}")
            print("Skipping 3D self-consistency metrics")
            rhofold = None
    
    # Metrics storage
    recovery_list = []
    perplexity_list = []
    sc_eternafold_list = []
    sc_rmsd_list = []
    sc_tm_list = []
    sc_gdt_list = []
    rmsd_within_thresh_list = []
    tm_within_thresh_list = []
    gdt_within_thresh_list = []
    
    # Process each structure
    for idx, item in enumerate(tqdm(ds, desc=f"Evaluating {ckpt_name}")):
        # Skip if graph sequence is empty
        if len(item.seq) == 0:
            continue
            
        # Sample n_samples sequences
        graph = item.graph.clone()
        graph.seq = item.seq
        
        # Sample sequences from the model
        samples, logits = model.sample(graph, n_samples, temperature, return_logits=True)
        
        # Compute sequence recovery (teacher-forced)
        recovery = samples.eq(item.seq).float().cpu().numpy()
        recovery_list.append(recovery.mean())
        
        # Compute perplexity
        n_nodes = logits.shape[1]
        perplexity = torch.exp(F.cross_entropy(
            logits.view(n_samples * n_nodes, model.out_dim),
            samples.view(n_samples * n_nodes).long(),
            reduction="none"
        ).view(n_samples, n_nodes).mean(dim=1))
        perplexity_list.extend(perplexity.cpu().numpy().tolist())
        
        # Get mask for valid coordinates from the featurized graph
        if hasattr(item.graph, 'mask_coords'):
            mask_coords = item.graph.mask_coords.cpu().numpy()
        else:
            # Fallback: create mask based on sequence length
            mask_coords = torch.ones(len(item.seq), dtype=torch.bool).numpy()
        
        # 2D Self-consistency with EternaFold
        if 'sc_eternafold' in metrics:
            try:
                sc_scores = self_consistency_score_eternafold(
                    samples.cpu().numpy(),
                    item.raw_data['sec_struct_list'],
                    mask_coords,
                    return_sec_structs=False
                )
                sc_eternafold_list.extend(sc_scores.tolist())
            except Exception as e:
                print(f"EternaFold failed for {item.gid}: {e}")
                sc_eternafold_list.extend([0.0] * n_samples)
        
        # 3D Self-consistency with RhoFold
        if 'sc_rhofold' in metrics and rhofold is not None:
            try:
                # Create output directory following evaluator.py pattern
                if current_datetime:
                    sample_output_dir = os.path.join(
                        output_dir or "runs/eval_full",
                        f"designs_{ckpt_name}",
                        current_datetime,
                        f"sample{idx}"
                    )
                else:
                    sample_output_dir = os.path.join(
                        output_dir or "runs/eval_full",
                        f"{ckpt_name}",
                        f"sample_{idx}"
                    )
                
                sc_rmsd, sc_tm, sc_gdt = self_consistency_score_rhofold(
                    samples.cpu().numpy(),
                    item.raw_data,
                    mask_coords,
                    rhofold,
                    sample_output_dir,
                    save_designs=save_designs,
                    save_pdbs=False,
                    use_relax=False
                )
                
                sc_rmsd_list.extend(sc_rmsd.tolist())
                sc_tm_list.extend(sc_tm.tolist())
                sc_gdt_list.extend(sc_gdt.tolist())
                
                # Compute threshold metrics
                rmsd_within_thresh_list.append((sc_rmsd <= RMSD_THRESHOLD).sum() / n_samples)
                tm_within_thresh_list.append((sc_tm >= TM_THRESHOLD).sum() / n_samples)
                gdt_within_thresh_list.append((sc_gdt >= GDT_THRESHOLD).sum() / n_samples)
                
            except Exception as e:
                print(f"RhoFold failed for {item.gid}: {e}")
                sc_rmsd_list.extend([100.0] * n_samples)  # Large RMSD for failed
                sc_tm_list.extend([0.0] * n_samples)
                sc_gdt_list.extend([0.0] * n_samples)
                rmsd_within_thresh_list.append(0.0)
                tm_within_thresh_list.append(0.0)
                gdt_within_thresh_list.append(0.0)
    
    # Aggregate metrics
    results = {
        "n_structures": len(ds),
        "n_samples_per_structure": n_samples,
        "temperature": temperature,
        "recovery": np.mean(recovery_list) if recovery_list else 0.0,
        "perplexity": np.mean(perplexity_list) if perplexity_list else 0.0,
    }
    
    if 'sc_eternafold' in metrics:
        results["sc_eternafold"] = np.mean(sc_eternafold_list) if sc_eternafold_list else 0.0
    
    if 'sc_rhofold' in metrics:
        results.update({
            "sc_rmsd": np.mean(sc_rmsd_list) if sc_rmsd_list else 100.0,
            "sc_tm": np.mean(sc_tm_list) if sc_tm_list else 0.0,
            "sc_gdt": np.mean(sc_gdt_list) if sc_gdt_list else 0.0,
            "rmsd_within_2A": np.mean(rmsd_within_thresh_list) if rmsd_within_thresh_list else 0.0,
            "tm_above_045": np.mean(tm_within_thresh_list) if tm_within_thresh_list else 0.0,
            "gdt_above_050": np.mean(gdt_within_thresh_list) if gdt_within_thresh_list else 0.0,
        })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--n_samples", default=8, type=int, help="Number of samples per structure")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature")
    parser.add_argument("--metrics", nargs="+", 
                       default=['recovery', 'perplexity', 'sc_eternafold', 'sc_rhofold'],
                       help="Metrics to compute")
    parser.add_argument("--save_designs", action="store_true", help="Save designed sequences")
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print(f"Loading {cfg.paths.split_name} split...")
    ds = FullEvalDataset(
        cfg.paths.processed_pt, 
        cfg.paths.split_pt, 
        cfg.paths.split_name, 
        cfg.featurizer, 
        device=device
    )
    print(f"Loaded {len(ds)} structures")
    
    # Initialize wandb if enabled
    wb = cfg.eval.wandb
    use_wandb = getattr(wb, "enable", False)
    if use_wandb:
        run_name = wb.run_name or f"full_eval_{cfg.paths.split_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=wb.project, entity=wb.entity, name=run_name, tags=wb.tags)
    
    # Evaluate each checkpoint
    os.makedirs(cfg.eval.out_dir, exist_ok=True)
    rows = []
    
    for ck in cfg.paths.checkpoints:
        name, path = ck.name, ck.path
        print(f"\n[Evaluating] {name} <- {path}")
        
        stats = eval_full_metrics(
            cfg, ds, name, path, device,
            n_samples=args.n_samples,
            temperature=args.temperature,
            metrics=args.metrics,
            save_designs=args.save_designs,
            output_dir=cfg.eval.out_dir
        )
        
        row = {
            "ckpt_name": name,
            "ckpt_path": path,
            "split": cfg.paths.split_name,
            **{k: float(v) for k, v in stats.items()}
        }
        rows.append(row)
        
        # Print summary
        print(f"  Recovery: {row['recovery']:.4f}")
        print(f"  Perplexity: {row['perplexity']:.2f}")
        if 'sc_eternafold' in args.metrics:
            print(f"  2D Self-consistency (EternaFold): {row.get('sc_eternafold', 0):.4f}")
        if 'sc_rhofold' in args.metrics:
            print(f"  3D Self-consistency:")
            print(f"    RMSD: {row.get('sc_rmsd', 100):.2f} Å")
            print(f"    TM-score: {row.get('sc_tm', 0):.4f}")
            print(f"    GDT: {row.get('sc_gdt', 0):.4f}")
            print(f"    % RMSD ≤ 2Å: {row.get('rmsd_within_2A', 0):.2%}")
            print(f"    % TM ≥ 0.45: {row.get('tm_above_045', 0):.2%}")
            print(f"    % GDT ≥ 0.50: {row.get('gdt_above_050', 0):.2%}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({f"{name}/{k}": v for k, v in row.items() if k not in ["ckpt_name", "ckpt_path", "split"]})
    
    # Write CSV with all results
    out_csv = os.path.join(cfg.eval.out_dir, f"full_eval_{cfg.paths.split_name}.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[Saved results to] {out_csv}")
    
    if use_wandb:
        # Create comparison table
        table = wandb.Table(columns=list(rows[0].keys()) if rows else [])
        for r in rows:
            table.add_data(*[r[k] for k in r.keys()])
        wandb.log({"full_eval/comparison_table": table})
        wandb.finish()


if __name__ == "__main__":
    main()