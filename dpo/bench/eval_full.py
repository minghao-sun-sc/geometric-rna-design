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

# Pass@k analysis imports (optional, only loaded when pass@k is enabled)
try:
    from multiround.passk import calculate_passk_metrics, plot_metric_distributions
    PASSK_AVAILABLE = True
except ImportError:
    PASSK_AVAILABLE = False


def _to_sn(o):
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(x) for x in o]
    return o

def _convert_for_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    else:
        return obj


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
    
    def __init__(self, processed_pt: str, split_pt: str, split_name: str, feat_cfg: SN, device="cpu", small_dataset: bool = False):
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
        
        # Apply small dataset option for fast testing
        if small_dataset:
            # Use indices 3-20 instead of 0-3 to avoid length mismatch issues in first few structures
            small_end = min(20, len(idxs))
            small_start = min(3, len(idxs) - 1)
            small_count = small_end - small_start
            print(f"🔬 Small dataset mode: Using structures {small_start}-{small_end-1} ({small_count} structures) from {len(idxs)} available")
            idxs = idxs[small_start:small_end]
        else:
            print(f"📊 Full dataset mode: Using all {len(idxs)} structures")

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
    output_dir: Optional[str] = None,
    passk_cfg: Optional[SN] = None  # Pass@k configuration
) -> Dict[str, Any]:
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
    
    # Pass@k configuration handling
    passk_enabled = passk_cfg is not None and getattr(passk_cfg, 'enable', False) and PASSK_AVAILABLE
    passk_n_samples = getattr(passk_cfg, 'n_samples_passk', 64) if passk_cfg else n_samples
    collect_individual = getattr(passk_cfg, 'collect_individual_metrics', False) if passk_cfg else False
    
    if passk_enabled and not PASSK_AVAILABLE:
        print("⚠️ Warning: Pass@k analysis requested but multiround.passk module not available. Skipping pass@k.")
        passk_enabled = False
    
    if passk_enabled:
        print(f"🎯 Pass@k analysis enabled: k_values={getattr(passk_cfg, 'k_values', [])}, n_samples={passk_n_samples}")
        # Override n_samples for pass@k if specified
        if passk_n_samples > n_samples:
            n_samples = passk_n_samples
            print(f"📈 Increased n_samples to {n_samples} for pass@k analysis")
    
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
    
    # Individual metrics collection for pass@k analysis (when enabled)
    individual_metrics = [] if (passk_enabled and collect_individual) else None
    
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
        vienna_success = False  # Track if Vienna calculation succeeded for this structure
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
                
                # Store metrics — NaN-filter at append (consistent with pS0/Tm) so
                # downstream np.mean / np.nanmean produces a real scalar instead of NaN.
                vienna_mfe_list.extend([x for x in v_mfe_scores if not np.isnan(x)])
                vienna_ed_list.extend([x for x in v_ed_scores if not np.isnan(x)])
                vienna_ednt_list.extend([x for x in v_ednt_scores if not np.isnan(x)])
                vienna_pS0_list.extend([x for x in v_pS0_scores if not np.isnan(x)])
                vienna_entropy_list.extend([x for x in v_entropy_scores if not np.isnan(x)])
                vienna_diversity_list.extend([x for x in v_diversity_scores if not np.isnan(x)])
                vienna_tm_list.extend([x for x in v_tm_scores if not np.isnan(x)])  # Store Tm values
                vienna_success = True  # Mark Vienna calculation as successful
                
            except Exception as e:
                print(f"Vienna metrics failed for {item.gid}: {e}")
                # Do NOT add dummy values - this allows proper exclusion from pass@k analysis
                # The individual_metrics collection will automatically skip missing Vienna metrics
        
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
        
        # Collect individual metrics for pass@k analysis (if enabled)
        if individual_metrics is not None:
            for sample_idx in range(n_samples):
                sample_metrics = {
                    "structure_id": item.gid,
                    "sample_idx": sample_idx,
                    "recovery": recovery[sample_idx].mean() if len(recovery) > sample_idx else 0.0,
                    "perplexity": perplexity_list[-(n_samples-sample_idx)] if len(perplexity_list) >= n_samples else 0.0,
                }
                
                # Add 2D metrics if available
                if 'sc_eternafold' in metrics and len(sc_eternafold_list) >= sample_idx + 1:
                    sample_metrics["sc_eternafold"] = sc_eternafold_list[-(n_samples-sample_idx)]
                
                # Add 3D metrics if available
                if 'sc_rhofold' in metrics and len(sc_rmsd_list) >= sample_idx + 1:
                    sample_metrics.update({
                        "sc_rmsd": sc_rmsd_list[-(n_samples-sample_idx)],
                        "sc_tm": sc_tm_list[-(n_samples-sample_idx)],
                        "sc_gdt": sc_gdt_list[-(n_samples-sample_idx)],
                        "sc_plddt": sc_plddt_list[-(n_samples-sample_idx)],
                    })
                    
                    # Add extended metrics if available
                    if len(inf_all_list) >= sample_idx + 1:
                        sample_metrics["inf_all"] = inf_all_list[-(n_samples-sample_idx)]
                    if len(clashscore_pre_list) >= sample_idx + 1:
                        sample_metrics["clashscore_pre"] = clashscore_pre_list[-(n_samples-sample_idx)]
                    if len(lddt_list) >= sample_idx + 1:
                        sample_metrics["lddt"] = lddt_list[-(n_samples-sample_idx)]
                    if len(mcq_abs_list) >= sample_idx + 1:
                        sample_metrics["mcq_abs_deg"] = mcq_abs_list[-(n_samples-sample_idx)]
                
                # Add Vienna metrics if calculation succeeded for this structure
                if vienna_success and len(vienna_mfe_list) >= sample_idx + 1:
                    sample_metrics["vienna_mfe"] = vienna_mfe_list[-(n_samples-sample_idx)]
                
                # Add diversity metrics if available
                if len(diversity_3mer_list) >= sample_idx + 1:
                    sample_metrics["diversity_3mer"] = diversity_3mer_list[-(n_samples-sample_idx)]
                
                individual_metrics.append(sample_metrics)
    
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
            "vienna_mfe": float(np.nanmean(vienna_mfe_list)) if vienna_mfe_list else float('nan'),
            "vienna_ED": float(np.nanmean(vienna_ed_list)) if vienna_ed_list else float('nan'),
            "vienna_ED_per_nt": float(np.nanmean(vienna_ednt_list)) if vienna_ednt_list else float('nan'),
            "vienna_pS0": float(np.nanmean(vienna_pS0_list)) if vienna_pS0_list else float('nan'),
            "vienna_entropy": float(np.nanmean(vienna_entropy_list)) if vienna_entropy_list else float('nan'),
            "vienna_diversity": float(np.nanmean(vienna_diversity_list)) if vienna_diversity_list else float('nan'),
            "vienna_Tm": float(np.nanmean(vienna_tm_list)) if vienna_tm_list else float('nan'),
        })
    
    # Add diversity metrics if computed
    if 'diversity_3mer' in metrics:
        results["diversity_3mer"] = np.mean(diversity_3mer_list) if diversity_3mer_list else 0.0
    
    # Add novelty metrics if computed
    if 'novelty_tpn' in metrics:
        results["novelty_tpn"] = np.mean(novelty_tpn_list) if novelty_tpn_list else 0.0
    
    # Pass@k analysis (if enabled)
    passk_results = None
    if passk_enabled and individual_metrics:
        try:
            print(f"\n🎯 Computing pass@k analysis...")
            
            # Organize individual metrics by structure
            metrics_by_structure = {}
            for sample_data in individual_metrics:
                struct_id = sample_data["structure_id"]
                if struct_id not in metrics_by_structure:
                    metrics_by_structure[struct_id] = []
                metrics_by_structure[struct_id].append(sample_data)
            
            # Get thresholds from config and convert SimpleNamespace to dict if needed
            thresholds_sn = getattr(passk_cfg, 'thresholds', {})
            if hasattr(thresholds_sn, '__dict__'):
                # Convert SimpleNamespace to dict
                thresholds = thresholds_sn.__dict__
            else:
                thresholds = thresholds_sn
            k_values = getattr(passk_cfg, 'k_values', [1, 2, 4, 8, 16, 32, 64])
            
            # Calculate pass@k metrics
            passk_results = calculate_passk_metrics(
                metrics_by_structure,
                k_values=k_values,
                thresholds=thresholds
            )
            
            # Add pass@k results to main results
            if passk_results:
                results["passk_analysis"] = passk_results
                print(f"✅ Pass@k analysis completed: {len(k_values)} k-values, {len(metrics_by_structure)} structures")
            
            # Generate distribution plots if enabled
            plot_distributions = getattr(passk_cfg, 'plot_distributions', False)
            if plot_distributions and output_dir:
                try:
                    plot_metrics = getattr(passk_cfg, 'plot_metrics', ["plddt", "rmsd", "tm_score", "mfe", "inf_all"])
                    passk_output_dir = os.path.join(output_dir, getattr(passk_cfg, 'passk_output_dir', 'passk_analysis'))
                    os.makedirs(passk_output_dir, exist_ok=True)
                    
                    plot_metric_distributions(
                        individual_metrics,
                        output_dir=passk_output_dir,
                        metrics_to_plot=plot_metrics,
                        checkpoint_name=ckpt_name
                    )
                    print(f"📊 Distribution plots saved to {passk_output_dir}")
                except Exception as e:
                    print(f"⚠️ Warning: Distribution plotting failed: {e}")
            
            # Save detailed pass@k results if enabled
            save_passk = getattr(passk_cfg, 'save_passk_results', True)
            if save_passk and output_dir and passk_results:
                try:
                    passk_output_dir = os.path.join(output_dir, getattr(passk_cfg, 'passk_output_dir', 'passk_analysis'))
                    os.makedirs(passk_output_dir, exist_ok=True)
                    
                    # Save detailed results
                    passk_detailed_path = os.path.join(passk_output_dir, f"{ckpt_name}_passk_detailed.json")
                    detailed_data = {
                        "checkpoint_name": ckpt_name,
                        "k_values": k_values,
                        "thresholds": thresholds,
                        "passk_results": passk_results,
                        "individual_metrics": individual_metrics
                    }
                    # Convert numpy types to JSON-serializable types
                    detailed_data_clean = _convert_for_json(detailed_data)
                    
                    with open(passk_detailed_path, 'w') as f:
                        json.dump(detailed_data_clean, f, indent=2)
                    print(f"💾 Detailed pass@k results saved to {passk_detailed_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save detailed pass@k results: {e}")
            
        except Exception as e:
            print(f"❌ Pass@k analysis failed: {e}")
            results["passk_analysis"] = {"error": str(e)}
    
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
    small_dataset = getattr(cfg.paths, 'small_dataset', False)
    print(f"Loading {split_name} split...")
    ds = FullEvalDataset(
        cfg.paths.processed_pt, 
        cfg.paths.split_pt, 
        split_name, 
        cfg.featurizer, 
        device="cpu",
        small_dataset=small_dataset
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
        # save_designs: CLI flag wins; otherwise honor cfg.eval.save_designs (was previously ignored)
        save_designs = args.save_designs if args.save_designs else bool(getattr(cfg.eval, 'save_designs', False))

        # Get pass@k configuration from config file
        passk_cfg = getattr(cfg.eval, 'passk', None) if hasattr(cfg, 'eval') else None

        stats = eval_full_metrics(
            cfg, ds, name, path, device,
            n_samples=n_samples,
            temperature=temperature,
            metrics=args.metrics,
            save_designs=save_designs,
            output_dir=out_dir,
            passk_cfg=passk_cfg
        )
        
        # Filter out non-numeric values (like pass@k analysis dict) before converting to float
        numeric_stats = {}
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_stats[k] = float(v)
            elif k == "passk_analysis":
                # Store pass@k analysis separately, don't convert to float
                continue
            else:
                # For other non-numeric values, store as-is
                numeric_stats[k] = v
        
        row = {
            "ckpt_name": name,
            "ckpt_path": path,
            "split": cfg.paths.split_name,
            **numeric_stats
        }
        
        # Add pass@k analysis to row if available (for JSON output)
        if "passk_analysis" in stats:
            row["passk_analysis"] = stats["passk_analysis"]
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
                # Only compute statistics for numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric_values:
                    json_output["summary"][metric] = {
                        "mean": float(np.mean(numeric_values)),
                        "std": float(np.std(numeric_values)),
                        "min": float(np.min(numeric_values)),
                        "max": float(np.max(numeric_values))
                    }
                elif metric == "passk_analysis":
                    # For pass@k analysis, just note it's available
                    json_output["summary"][metric] = {"note": "Pass@k analysis results available in per_checkpoint_results"}
                else:
                    # For other non-numeric metrics, note their type
                    json_output["summary"][metric] = {"note": f"Non-numeric data ({type(values[0]).__name__})"}
        
        # Add checkpoint names for reference
        json_output["checkpoints_evaluated"] = [r["ckpt_name"] for r in rows]
    
    # Save JSON to dpo/eval_results/ with checkpoint names for identification
    ckpt_names_str = "_".join([r["ckpt_name"] for r in rows[:3]])  # First 3 to avoid too long names
    if len(rows) > 3:
        ckpt_names_str += f"_and_{len(rows)-3}more"
    json_filename = f"eval_{cfg.paths.split_name}_{ckpt_names_str}_{timestamp}.json"
    json_path = os.path.join(eval_results_dir, json_filename)
    
    # Convert numpy types to JSON-serializable types
    json_output_clean = _convert_for_json(json_output)
    
    with open(json_path, "w") as f:
        json.dump(json_output_clean, f, indent=2)
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