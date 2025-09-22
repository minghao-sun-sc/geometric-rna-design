# backup file before adding pass@k


"""
Full evaluation pipeline for DPO-RNA models including all gRNAde metrics:
- Sequence Recovery (teacher-forced)
- Perplexity
- 2D Self-consistency (EternaFold)
- 3D Self-consistency (RhoFold: RMSD, TM-score, GDT)
"""

import os
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()
import sys 

import math
import csv
import json
import time
import yaml
import copy
import shutil
from types import SimpleNamespace as SN
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# Import these inside functions to avoid NetworkX conflicts
# from src.data.featurizer import RNAGraphFeaturizer  
# from src.data.data_utils import get_backbone_coords
from src.constants import NUM_TO_LETTER, RMSD_THRESHOLD, RMSD_THRESHOLD_2, TM_THRESHOLD, GDT_THRESHOLD, PLDDT_THRESHOLD

from dpo.ref_manager import build_model_from_cfg
from dpo.utils import set_seed, load_processed_pt

# Import existing self-consistency functions directly from evaluator
from src.evaluator import (
    self_consistency_score_eternafold,
    self_consistency_score_rhofold_extended,
    get_trimer_profile_novelty
)
from src.constants import (
    RMSD_THRESHOLD, RMSD_THRESHOLD_2, TM_THRESHOLD, 
    GDT_THRESHOLD, PLDDT_THRESHOLD
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
        
        # Import here to avoid NetworkX conflicts at module level
        from src.data.featurizer import RNAGraphFeaturizer
        
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
            # Import locally to avoid NetworkX conflicts
            from src.data.data_utils import get_backbone_coords
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
    sc_plddt_list = []
    rmsd_within_thresh_list = []  # RMSD <= 8Å
    rmsd_within_2A_list = []      # RMSD <= 2Å
    tm_within_thresh_list = []
    gdt_within_thresh_list = []
    plddt_within_thresh_list = []  # pLDDT >= 0.70
    
    # Vienna metrics storage
    vienna_mfe_list = []
    vienna_ed_list = []
    vienna_ednt_list = []
    vienna_pS0_list = []
    vienna_entropy_list = []
    vienna_diversity_list = []
    vienna_tm_list = []
    
    # Diversity metrics storage
    diversity_3mer_list = []
    novelty_tpn_list = []
    
    # Extended metrics storage (INF, clash, lDDT, MCQ)
    inf_all_list = []
    inf_wc_list = []
    inf_nwc_list = []
    inf_stack_list = []
    clashscore_pre_list = []
    clashscore_post_list = []
    lddt_list = []
    mcq_abs_list = []
    mcq_R_list = []
    mcq_sd_list = []
    
    # Process each structure (with optional limit for debugging)
    max_structures = getattr(cfg.eval, 'max_structures', None)
    items_to_process = ds if max_structures is None else ds[:max_structures]
    
    for idx, item in enumerate(tqdm(items_to_process, desc=f"Evaluating {ckpt_name}")):
        # Skip if graph sequence is empty
        if len(item.seq) == 0:
            continue
            
        # Sample n_samples sequences
        graph = item.graph.clone().to(device)
        graph.seq = item.seq.to(device)
        
        # Sample sequences from the model
        samples, logits = model.sample(graph, n_samples, temperature, return_logits=True)
        
        # Compute sequence recovery (teacher-forced)
        recovery = samples.eq(item.seq.to(device)).float().cpu().numpy()
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
        
        # Vienna thermodynamic metrics
        vienna_requested = any(m.startswith('vienna') for m in metrics) or 'sc_vienna' in metrics
        if vienna_requested:
            try:
                from src.evaluator import vienna_ensemble_metrics, vienna_mfe, _sanitize_db_for_vienna
                
                # Choose target structure and track source
                vienna_target_source = "native"  # Track whether we use native or MFE
                if len(item.raw_data.get('sec_struct_list', [])) > 0:
                    target_db_full = _sanitize_db_for_vienna(item.raw_data['sec_struct_list'][0])
                    vienna_target_source = "native_structure"
                else:
                    # Fold native sequence to get target
                    _mfe_native, target_db_full = vienna_mfe(item.raw_data['sequence'], 37.0)
                    vienna_target_source = f"MFE_fold (E={_mfe_native:.1f})"
                
                # Apply mask if needed (with bounds checking)
                if mask_coords is not None and mask_coords.sum() < len(mask_coords):
                    keep_idx = np.where(mask_coords)[0]
                    # Bounds check to prevent string index out of range
                    keep_idx = keep_idx[keep_idx < len(target_db_full)]
                    if len(keep_idx) > 0:
                        target_db = "".join(target_db_full[i] for i in keep_idx)
                    else:
                        target_db = target_db_full
                else:
                    target_db = target_db_full
                
                # Compute Vienna metrics for each sample
                v_mfe_scores, v_ed_scores, v_ednt_scores = [], [], []
                v_pS0_scores, v_entropy_scores, v_diversity_scores = [], [], []
                v_tm_scores = []  # Add melting temperature storage
                
                for seq_nums in samples.cpu().numpy():
                    from src.constants import NUM_TO_LETTER
                    seq = "".join([NUM_TO_LETTER[n] for n in seq_nums])
                    if mask_coords is not None and mask_coords.sum() < len(mask_coords):
                        # Bounds check to prevent string index out of range
                        keep_idx_seq = keep_idx[keep_idx < len(seq)]
                        if len(keep_idx_seq) > 0:
                            seq = "".join(seq[i] for i in keep_idx_seq)
                        # If no valid indices, keep original sequence
                    
                    v = vienna_ensemble_metrics(seq, target_db=target_db, T=37.0)
                    v_mfe_scores.append(v["mfe"])
                    v_ed_scores.append(v["ED"])
                    v_ednt_scores.append(v["ED_per_nt"])
                    v_pS0_scores.append(v["pS0"])
                    v_entropy_scores.append(v["entropy_mean"])
                    v_diversity_scores.append(v["diversity"])
                    
                    # Calculate melting temperature (optional, can be expensive)
                    # Use step=2.0 for faster calculation vs default step=1.0
                    from src.evaluator import vienna_Tm_by_pS0
                    try:
                        tm = vienna_Tm_by_pS0(seq, target_db, Tmin=10, Tmax=95, step=2.0, threshold=0.5)
                        v_tm_scores.append(tm)
                    except Exception as e:
                        print(f"Vienna Tm calculation failed for sample: {e}")
                        v_tm_scores.append(float('nan'))
                
                # Store metrics
                vienna_mfe_list.extend(v_mfe_scores)
                vienna_ed_list.extend(v_ed_scores)
                vienna_ednt_list.extend(v_ednt_scores)
                vienna_pS0_list.extend([x for x in v_pS0_scores if not np.isnan(x)])
                vienna_entropy_list.extend(v_entropy_scores)
                vienna_diversity_list.extend(v_diversity_scores)
                vienna_tm_list.extend([x for x in v_tm_scores if not np.isnan(x)])  # Store Tm values
                
            except Exception as e:
                print(f"Vienna metrics failed for {item.gid}: {e}")
                # Add dummy values for failed computation
                vienna_mfe_list.extend([0.0] * n_samples)
                vienna_ed_list.extend([0.0] * n_samples)
                vienna_ednt_list.extend([0.0] * n_samples)
                vienna_pS0_list.extend([0.0] * n_samples)
                vienna_entropy_list.extend([0.0] * n_samples)
                vienna_diversity_list.extend([0.0] * n_samples)
                vienna_tm_list.extend([0.0] * n_samples)  # Add Tm fallback
        
        # Diversity metrics (3-mer correlation)
        if 'diversity_3mer' in metrics:
            try:
                from src.evaluator import get_three_mer_corr
                
                # Convert samples to sequences
                sample_seqs = []
                for seq_nums in samples.cpu().numpy():
                    from src.constants import NUM_TO_LETTER
                    seq = "".join([NUM_TO_LETTER[n] for n in seq_nums])
                    sample_seqs.append(seq)
                
                # Calculate 3-mer diversity (1 - correlation with native)
                native_seq = item.raw_data['sequence']
                corr_scores = get_three_mer_corr(sample_seqs, native_seq, mask_coords)
                diversity_scores = 1.0 - corr_scores  # Convert correlation to diversity
                
                diversity_3mer_list.extend(diversity_scores.tolist())
                
            except Exception as e:
                print(f"3-mer diversity failed for {item.gid}: {e}")
                diversity_3mer_list.extend([0.0] * n_samples)
        
        # Trimer Profile Novelty (TPN) metric
        if 'novelty_tpn' in metrics:
            try:
                # Calculate TPN for designed sequences
                tpn_scores = get_trimer_profile_novelty(samples.cpu().numpy(), mask_coords)
                novelty_tpn_list.extend(tpn_scores.tolist())
                
            except Exception as e:
                print(f"TPN novelty failed for {item.gid}: {e}")
                novelty_tpn_list.extend([0.0] * n_samples)
        
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
                
                # Use extended evaluator that returns 8 values: (rmsd, tm, gdt, plddt, inf_dict, clash, lddt, mcq)
                from src.constants import PROJECT_PATH
                result = self_consistency_score_rhofold_extended(
                    samples.cpu().numpy(),
                    item.raw_data,
                    mask_coords,
                    rhofold,
                    sample_output_dir,
                    save_designs=save_designs,
                    save_pdbs=False,
                    use_relax=getattr(cfg.eval, 'use_relax', True),  # Config-controlled Amber relaxation
                    use_inf=True,
                    use_clash=True,
                    use_lddt=getattr(cfg.eval, 'use_lddt', True),     # Config-controlled lDDT calculation
                    use_mcq=True,
                    phenix_wrapper_path=os.path.join(PROJECT_PATH, "tools", "run_phenix.sh"),
                )
                
                # Unpack 8 values from extended function
                sc_rmsd, sc_tm, sc_gdt, sc_plddt, inf_dict, clash_dict, lddt_scores, mcq_dict = result
                
                sc_rmsd_list.extend(sc_rmsd.tolist())
                sc_tm_list.extend(sc_tm.tolist())
                sc_gdt_list.extend(sc_gdt.tolist())
                sc_plddt_list.extend(sc_plddt.tolist())
                
                # Store extended metrics (INF and clash)
                if inf_dict and 'all' in inf_dict:
                    inf_all_list.extend(inf_dict['all'].tolist() if hasattr(inf_dict['all'], 'tolist') else [inf_dict['all']] * n_samples)
                    inf_wc_list.extend(inf_dict['wc'].tolist() if hasattr(inf_dict['wc'], 'tolist') else [inf_dict['wc']] * n_samples)
                    inf_nwc_list.extend(inf_dict['nwc'].tolist() if hasattr(inf_dict['nwc'], 'tolist') else [inf_dict['nwc']] * n_samples)
                    inf_stack_list.extend(inf_dict['stack'].tolist() if hasattr(inf_dict['stack'], 'tolist') else [inf_dict['stack']] * n_samples)
                
                if clash_dict is not None and 'pre_relax' in clash_dict:
                    # Handle pre-relax clash scores
                    clash_pre_scores = clash_dict['pre_relax']
                    if clash_pre_scores is not None and len(clash_pre_scores) > 0:
                        clash_pre_vals = clash_pre_scores.tolist() if hasattr(clash_pre_scores, 'tolist') else clash_pre_scores
                        for c_val in clash_pre_vals:
                            if c_val > 100:
                                print(f"  ⚠️ Warning: High pre-relax clash score {c_val:.1f} for {item.gid} (expected <50)")
                        clashscore_pre_list.extend(clash_pre_vals)
                    
                    # Handle post-relax clash scores
                    clash_post_scores = clash_dict['post_relax']
                    if clash_post_scores is not None and len(clash_post_scores) > 0:
                        clash_post_vals = clash_post_scores.tolist() if hasattr(clash_post_scores, 'tolist') else clash_post_scores
                        for c_val in clash_post_vals:
                            if not np.isnan(c_val) and c_val > 100:
                                print(f"  ⚠️ Warning: High post-relax clash score {c_val:.1f} for {item.gid} (expected <50)")
                        clashscore_post_list.extend(clash_post_vals)
                
                # Store lDDT scores
                if lddt_scores is not None and len(lddt_scores) > 0:
                    lddt_vals = lddt_scores.tolist() if hasattr(lddt_scores, 'tolist') else lddt_scores
                    lddt_list.extend(lddt_vals)
                
                # Store MCQ scores
                if mcq_dict and 'mcq_abs_deg' in mcq_dict:
                    mcq_abs_list.extend(mcq_dict['mcq_abs_deg'].tolist() if hasattr(mcq_dict['mcq_abs_deg'], 'tolist') else [mcq_dict['mcq_abs_deg']] * n_samples)
                    mcq_R_list.extend(mcq_dict['R'].tolist() if hasattr(mcq_dict['R'], 'tolist') else [mcq_dict['R']] * n_samples)
                    mcq_sd_list.extend(mcq_dict['circ_sd_deg'].tolist() if hasattr(mcq_dict['circ_sd_deg'], 'tolist') else [mcq_dict['circ_sd_deg']] * n_samples)
                
                # Compute threshold metrics
                rmsd_within_thresh_list.append((sc_rmsd <= RMSD_THRESHOLD).sum() / n_samples)  # 8Å
                rmsd_within_2A_list.append((sc_rmsd <= RMSD_THRESHOLD_2).sum() / n_samples)    # 2Å
                tm_within_thresh_list.append((sc_tm >= TM_THRESHOLD).sum() / n_samples)
                gdt_within_thresh_list.append((sc_gdt >= GDT_THRESHOLD).sum() / n_samples)
                plddt_within_thresh_list.append((sc_plddt >= PLDDT_THRESHOLD).sum() / n_samples)  # 0.70
                
            except Exception as e:
                print(f"RhoFold failed for {item.gid}: {e}")
                sc_rmsd_list.extend([100.0] * n_samples)  # Large RMSD for failed
                sc_tm_list.extend([0.0] * n_samples)
                sc_gdt_list.extend([0.0] * n_samples)
                sc_plddt_list.extend([0.0] * n_samples)  # Low pLDDT for failed
                rmsd_within_thresh_list.append(0.0)
                rmsd_within_2A_list.append(0.0)
                tm_within_thresh_list.append(0.0)
                gdt_within_thresh_list.append(0.0)
                plddt_within_thresh_list.append(0.0)
                
                # CRITICAL: Add NaN values for extended metrics to ensure consistent columns
                # (This is scientifically valid - NaN means "measurement failed", not fake data)
                inf_all_list.extend([np.nan] * n_samples)
                inf_wc_list.extend([np.nan] * n_samples)
                inf_nwc_list.extend([np.nan] * n_samples)
                inf_stack_list.extend([np.nan] * n_samples)
                clashscore_pre_list.extend([np.nan] * n_samples)
                clashscore_post_list.extend([np.nan] * n_samples)
                lddt_list.extend([np.nan] * n_samples)
                mcq_abs_list.extend([np.nan] * n_samples)
                mcq_R_list.extend([np.nan] * n_samples)
                mcq_sd_list.extend([np.nan] * n_samples)
    
    # Aggregate metrics
    n_processed = len(items_to_process)
    results = {
        "n_structures": n_processed,
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
            "sc_plddt": np.mean(sc_plddt_list) if sc_plddt_list else 0.0,
            "rmsd_within_8A": np.mean(rmsd_within_thresh_list) if rmsd_within_thresh_list else 0.0,
            "rmsd_within_2A": np.mean(rmsd_within_2A_list) if rmsd_within_2A_list else 0.0,
            "tm_above_045": np.mean(tm_within_thresh_list) if tm_within_thresh_list else 0.0,
            "gdt_above_050": np.mean(gdt_within_thresh_list) if gdt_within_thresh_list else 0.0,
            "plddt_above_070": np.mean(plddt_within_thresh_list) if plddt_within_thresh_list else 0.0,
        })
        
        # Add extended metrics if available
        if inf_all_list:
            results.update({
                "inf_all": np.nanmean(inf_all_list),
                "inf_wc": np.nanmean(inf_wc_list),
                "inf_nwc": np.nanmean(inf_nwc_list),
                "inf_stack": np.nanmean(inf_stack_list),
            })
        if clashscore_pre_list:
            results["clashscore_pre_relax"] = np.nanmean(clashscore_pre_list)
        if clashscore_post_list:
            results["clashscore_post_relax"] = np.nanmean(clashscore_post_list)
        if lddt_list:
            # Calculate lDDT statistics with proper NaN handling
            lddt_array = np.array(lddt_list)
            valid_lddt = lddt_array[~np.isnan(lddt_array)]
            results["lddt"] = np.mean(valid_lddt) if len(valid_lddt) > 0 else float('nan')
            results["lddt_success_rate"] = len(valid_lddt) / len(lddt_array) if len(lddt_array) > 0 else 0.0
        if mcq_abs_list:
            results.update({
                "mcq_abs_deg": np.nanmean(mcq_abs_list),
                "mcq_R": np.nanmean(mcq_R_list),
                "mcq_sd_deg": np.nanmean(mcq_sd_list),
            })
    
    # Add Vienna metrics if computed
    vienna_requested = any(m.startswith('vienna') for m in metrics) or 'sc_vienna' in metrics
    if vienna_requested:
        results.update({
            "vienna_mfe": np.mean(vienna_mfe_list) if vienna_mfe_list else 0.0,
            "vienna_ED": np.mean(vienna_ed_list) if vienna_ed_list else 0.0,
            "vienna_ED_per_nt": np.mean(vienna_ednt_list) if vienna_ednt_list else 0.0,
            "vienna_pS0": np.mean(vienna_pS0_list) if vienna_pS0_list else 0.0,
            "vienna_entropy": np.mean(vienna_entropy_list) if vienna_entropy_list else 0.0,
            "vienna_diversity": np.mean(vienna_diversity_list) if vienna_diversity_list else 0.0,
            "vienna_Tm": np.mean(vienna_tm_list) if vienna_tm_list else 0.0,
        })
    
    # Add diversity metrics if computed
    if 'diversity_3mer' in metrics:
        results["diversity_3mer"] = np.mean(diversity_3mer_list) if diversity_3mer_list else 0.0
    
    # Add novelty metrics if computed
    if 'novelty_tpn' in metrics:
        results["novelty_tpn"] = np.mean(novelty_tpn_list) if novelty_tpn_list else 0.0
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--n_samples", default=None, type=int, help="Number of samples per structure (overrides config)")
    parser.add_argument("--temperature", default=None, type=float, help="Sampling temperature (overrides config)")
    parser.add_argument("--metrics", nargs="+", 
                       default=None,  # Use config file metrics if not specified
                       help="Metrics to compute (overrides config if specified)")
    parser.add_argument("--save_designs", action="store_true", help="Save designed sequences")
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    set_seed(getattr(cfg, 'seed', 42))
    device = torch.device(getattr(cfg, 'device', 'cuda') if torch.cuda.is_available() else "cpu")
    
    # Use config metrics if not specified on command line
    if args.metrics is None:
        args.metrics = getattr(cfg.eval, 'metrics', ['recovery', 'perplexity', 'sc_eternafold', 'sc_rhofold', 'sc_vienna', 'diversity_3mer'])
    print(f"Metrics to compute: {args.metrics}")
    
    # Load dataset
    split_name = getattr(cfg.paths, 'split_name', 'test')
    print(f"Loading {split_name} split...")
    ds = FullEvalDataset(
        cfg.paths.processed_pt, 
        cfg.paths.split_pt, 
        split_name, 
        cfg.featurizer, 
        device="cpu"
    )
    print(f"Loaded {len(ds)} structures")
    
    # Initialize wandb if enabled
    wb = getattr(cfg.eval, 'wandb', None)
    use_wandb = getattr(wb, "enable", False) if wb else False
    if use_wandb:
        run_name = getattr(wb, 'run_name', None) or f"full_eval_{split_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=wb.project, entity=wb.entity, name=run_name, tags=getattr(wb, 'tags', None))
    
    # Evaluate each checkpoint
    out_dir = getattr(cfg.eval, 'out_dir', 'runs/eval_full')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    
    for ck in cfg.paths.checkpoints:
        name, path = ck.name, ck.path
        print(f"\n[Evaluating] {name} <- {path}")
        
        # Use command line args if provided, otherwise use config values
        n_samples = args.n_samples if args.n_samples is not None else getattr(cfg.eval, 'n_samples', 8)
        temperature = args.temperature if args.temperature is not None else getattr(cfg.eval, 'temperature', 0.5)
        
        stats = eval_full_metrics(
            cfg, ds, name, path, device,
            n_samples=n_samples,
            temperature=temperature,
            metrics=args.metrics,
            save_designs=args.save_designs,
            output_dir=out_dir
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
            print(f"    pLDDT: {row.get('sc_plddt', 0):.4f}")
            print(f"    % RMSD ≤ 8Å: {row.get('rmsd_within_8A', 0):.2%}")
            print(f"    % RMSD ≤ 2Å: {row.get('rmsd_within_2A', 0):.2%}")
            print(f"    % TM ≥ 0.45: {row.get('tm_above_045', 0):.2%}")
            print(f"    % GDT ≥ 0.50: {row.get('gdt_above_050', 0):.2%}")
            print(f"    % pLDDT ≥ 0.70: {row.get('plddt_above_070', 0):.2%}")
            # Show INF scores if available with interpretation
            if 'inf_all' in row and not pd.isna(row.get('inf_all', np.nan)):
                print(f"  Interaction Network Fidelity (0-1, higher=better):")
                print(f"    INF (all): {row.get('inf_all', 0):.4f} - overall contact correctness")
                print(f"    INF (WC): {row.get('inf_wc', 0):.4f} - Watson-Crick pairing fidelity")
                inf_nwc = row.get('inf_nwc', 0)
                nwc_note = " (normal for complex cases)" if inf_nwc < 0 else ""
                print(f"    INF (non-WC): {inf_nwc:.4f} - non-WC interactions{nwc_note}")
                print(f"    INF (stack): {row.get('inf_stack', 0):.4f} - stacking interactions")
            # Show clash scores if available with warnings
            if 'clashscore_pre_relax' in row and not pd.isna(row.get('clashscore_pre_relax', np.nan)):
                clash_pre_val = row.get('clashscore_pre_relax', 0)
                clash_pre_note = " (high)" if clash_pre_val > 100 else " ✓" if clash_pre_val < 20 else ""
                print(f"    Clash score (pre-relax): {clash_pre_val:.2f}{clash_pre_note}")
                
            if 'clashscore_post_relax' in row and not pd.isna(row.get('clashscore_post_relax', np.nan)):
                clash_post_val = row.get('clashscore_post_relax', 0)
                clash_post_note = " (high)" if clash_post_val > 100 else " ✓" if clash_post_val < 20 else ""
                print(f"    Clash score (post-relax): {clash_post_val:.2f}{clash_post_note}")
            elif 'clashscore_pre_relax' in row:
                print(f"    Clash score (post-relax): N/A (no relaxation performed)")
            # Show lDDT score if available
            if 'lddt' in row and not pd.isna(row.get('lddt', np.nan)):
                lddt_val = row.get('lddt', 0)
                lddt_success_rate = row.get('lddt_success_rate', 0.0)
                lddt_note = " ✓" if lddt_val > 0.7 else "" if lddt_val < 0.3 else ""
                print(f"    lDDT: {lddt_val:.4f} - local distance accuracy{lddt_note} (success: {lddt_success_rate:.1%})")
            # Show MCQ scores if available
            if 'mcq_abs_deg' in row and not pd.isna(row.get('mcq_abs_deg', np.nan)):
                mcq_abs = row.get('mcq_abs_deg', 0)
                mcq_R = row.get('mcq_R', 0)
                mcq_sd = row.get('mcq_sd_deg', 0)
                mcq_note = " ✓" if mcq_abs < 30 else "" if mcq_abs > 60 else ""
                print(f"  Mean of Circular Quantities (degrees, lower=better):")
                print(f"    MCQ abs: {mcq_abs:.2f}° - torsional accuracy{mcq_note}")
                print(f"    MCQ R: {mcq_R:.3f} - concentration (higher=better)")
                print(f"    MCQ σ: {mcq_sd:.2f}° - circular std dev")
        
        # Show Vienna thermodynamics metrics if available
        if any(k.startswith('vienna_') for k in row.keys()):
            # Note: Target structure is from native or MFE fold (tracked during computation)
            print(f"  Thermodynamics (Vienna):")
            if 'vienna_mfe' in row:
                print(f"    MFE: {row.get('vienna_mfe', 0):.2f} kcal/mol")
            if 'vienna_ED_per_nt' in row:
                print(f"    Ensemble Defect/nt: {row.get('vienna_ED_per_nt', 0):.4f}")
            if 'vienna_pS0' in row:
                print(f"    P(target): {row.get('vienna_pS0', 0):.4f}")
            if 'vienna_entropy' in row:
                print(f"    Shannon Entropy: {row.get('vienna_entropy', 0):.3f}")
            if 'vienna_Tm' in row:
                print(f"    Tm: {row.get('vienna_Tm', 0):.1f} °C")
        
        # Show diversity and novelty metrics if available  
        if 'diversity_3mer' in row or 'novelty_tpn' in row:
            print(f"  Diversity & Novelty:")
            if 'diversity_3mer' in row:
                print(f"    3-mer diversity: {row.get('diversity_3mer', 0):.4f}")
            if 'novelty_tpn' in row:
                print(f"    Trimer Profile Novelty (TPN): {row.get('novelty_tpn', 0):.4f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({f"{name}/{k}": v for k, v in row.items() if k not in ["ckpt_name", "ckpt_path", "split"]})
    
    # Create output directory in dpo/eval_results/
    eval_results_dir = "dpo/eval_results"
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write CSV with all results (keep for backward compatibility)
    out_csv = os.path.join(cfg.eval.out_dir, f"full_eval_{cfg.paths.split_name}.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[Saved CSV results to] {out_csv}")
    
    # Get actual values used (from args or config)
    actual_n_samples = args.n_samples if args.n_samples is not None else getattr(cfg.eval, 'n_samples', 8)
    actual_temperature = args.temperature if args.temperature is not None else getattr(cfg.eval, 'temperature', 0.5)
    
    # Prepare comprehensive JSON output
    json_output = {
        "metadata": {
            "timestamp": timestamp,
            "split": cfg.paths.split_name,
            "n_structures": len(ds),
            "n_samples_per_structure": actual_n_samples,
            "temperature": actual_temperature,
            "metrics_computed": args.metrics,
            "save_designs": args.save_designs,
            "config_file": args.config,
            "command_args": vars(args)
        },
        "checkpoints_evaluated": [],
        "summary": {},
        "per_checkpoint_results": rows
    }
    
    # Calculate summary statistics across all checkpoints
    if rows:
        # Get all metric keys (excluding metadata fields)
        metric_keys = [k for k in rows[0].keys() if k not in ["ckpt_name", "ckpt_path", "split"]]
        
        for metric in metric_keys:
            values = [r[metric] for r in rows if metric in r]
            if values:
                json_output["summary"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Add checkpoint names for reference
        json_output["checkpoints_evaluated"] = [r["ckpt_name"] for r in rows]
    
    # Save JSON to dpo/eval_results/ with checkpoint names for identification
    ckpt_names_str = "_".join([r["ckpt_name"] for r in rows[:3]])  # First 3 to avoid too long names
    if len(rows) > 3:
        ckpt_names_str += f"_and_{len(rows)-3}more"
    json_filename = f"eval_{cfg.paths.split_name}_{ckpt_names_str}_{timestamp}.json"
    json_path = os.path.join(eval_results_dir, json_filename)
    
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"[Saved JSON results to] {json_path}")
    
    # Also save a "latest" symlink for easy access
    latest_json_path = os.path.join(eval_results_dir, f"eval_{cfg.paths.split_name}_latest.json")
    try:
        # Remove existing symlink/file if it exists
        if os.path.lexists(latest_json_path):  # lexists detects broken symlinks too
            os.remove(latest_json_path)
        os.symlink(os.path.abspath(json_path), os.path.abspath(latest_json_path))
        print(f"[Created latest symlink] {latest_json_path}")
    except (FileExistsError, OSError) as e:
        # Handle race conditions and permission issues more robustly
        print(f"[Warning] Could not create symlink {latest_json_path}: {e}")
        print(f"[Info] Main results still saved to: {json_path}")
    
    if use_wandb:
        # Create comparison table with consistent columns
        if rows:
            # Get all unique columns across all rows
            all_columns = set()
            for r in rows:
                all_columns.update(r.keys())
            all_columns = sorted(list(all_columns))
            
            # Create table with all columns
            table = wandb.Table(columns=all_columns)
            for r in rows:
                # Fill missing columns with NaN for consistency
                row_data = [r.get(col, np.nan) for col in all_columns]
                table.add_data(*row_data)
            wandb.log({"checkpoint_comparison": table})
        wandb.finish()


if __name__ == "__main__":
    main()