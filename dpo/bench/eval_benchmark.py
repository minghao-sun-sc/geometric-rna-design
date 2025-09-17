from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os, math, csv, time, yaml, json, shutil
from types import SimpleNamespace as SN
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

from src.data.featurizer import RNAGraphFeaturizer
from src.data.data_utils import get_backbone_coords
from src.constants import NUM_TO_LETTER, LETTER_TO_NUM
from dpo.ref_manager import build_model_from_cfg
from dpo.utils import set_seed, load_processed_pt

# Import all evaluation metrics from evaluator
from src.evaluator import (
    self_consistency_score_eternafold,
    self_consistency_score_rhofold,
    vienna_mfe,
    vienna_ensemble_metrics,
    vienna_Tm_by_pS0,
    get_clash_score_phenix,
    get_inf,
    get_lddt_robust,
    usalign_tm_vs_natives,
    mcq_avg_vs_natives,
    get_three_mer_corr
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
    raw_data: Dict[str, Any]  # Original raw data for comprehensive evaluation
    native_seq_str: str  # Native sequence as string for comparison

class ComprehensiveDataset(Dataset):
    """Enhanced dataset for comprehensive evaluation including all metrics."""
    
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
                "id_list": entry.get("id_list", [f"idx_{gi}"]),
                "rfam_list": entry.get("rfam_list", ["unknown"]),
                "eq_class_list": entry.get("eq_class_list", ["unknown"]),
                "cluster_structsim0.45": entry.get("cluster_structsim0.45", "unknown")
            }
            
            # Use the graph's sequence to ensure length consistency
            seq = graph.seq.to(device)
            gid = entry["id_list"][0] if entry.get("id_list") else f"idx_{gi}"
            native_seq_str = entry["sequence"]
            
            self.items.append(GraphItem(
                gid=gid, 
                graph=graph, 
                seq=seq, 
                raw_data=raw_data, 
                native_seq_str=native_seq_str
            ))

    def __len__(self): 
        return len(self.items)
        
    def __getitem__(self, i): 
        return self.items[i]

def _human_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def _sample_sequence(model, graph, temperature=1.0, device="cpu"):
    """Sample a sequence from the model given a graph structure."""
    model.eval()
    with torch.no_grad():
        # Start with random initialization for sampling
        seq_len = graph.seq.size(0)
        sampled_seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        
        for i in range(seq_len):
            # Create input with sampled tokens so far
            current_graph = graph.clone()
            current_graph.seq = sampled_seq
            
            # Get logits for next position  
            logits = model(current_graph)  # [L, 4]
            
            # Sample from distribution at position i
            if temperature > 0:
                probs = F.softmax(logits[i] / temperature, dim=-1)
                sampled_seq[i] = torch.multinomial(probs, 1).item()
            else:
                sampled_seq[i] = logits[i].argmax().item()
    
    return sampled_seq


def _seq_to_string(seq_tensor):
    """Convert sequence tensor to string."""
    return "".join([NUM_TO_LETTER[int(x)] for x in seq_tensor])


def _save_sequence_fasta(sequences: List[str], seq_ids: List[str], filepath: str):
    """Save sequences to FASTA file."""
    records = []
    for seq_id, seq in zip(seq_ids, sequences):
        record = SeqRecord(Seq(seq), id=seq_id, description="")
        records.append(record)
    
    with open(filepath, 'w') as f:
        SeqIO.write(records, f, "fasta")


def _compute_basic_metrics(model, dataset, device) -> Dict[str, float]:
    """Compute basic teacher-forced metrics (recovery, perplexity)."""
    model.eval()
    n_graphs = 0
    total_tokens = 0
    correct_tokens = 0
    exact_ok = 0
    sum_ce = 0.0

    with torch.no_grad():
        for item in tqdm(dataset, desc="Computing basic metrics"):
            # Teacher-forced logits on the native sequence
            g = item.graph.clone()
            g.seq = item.seq
            logits = model(g)   # [L, 4]
            
            # CE and accuracy
            ce = F.cross_entropy(logits, item.seq, reduction="sum")   # NLL (sum over L)
            sum_ce += float(ce.item())
            preds = logits.argmax(dim=-1)
            correct = (preds == item.seq).sum().item()
            correct_tokens += correct
            total_tokens += item.seq.numel()
            exact_ok += 1 if (correct == item.seq.numel()) else 0
            n_graphs += 1

    nll_token = sum_ce / max(1, total_tokens)
    ppl = math.exp(nll_token)
    acc = correct_tokens / max(1, total_tokens)
    exact = exact_ok / max(1, n_graphs)

    return {
        "n_graphs": n_graphs,
        "recovery": acc,
        "seq_exact": exact,
        "nll_per_token": nll_token,
        "perplexity": ppl
    }


def _compute_comprehensive_metrics(
    model, dataset, n_samples: int, temperature: float, 
    metrics_to_compute: List[str], out_dir: str, 
    save_designs: bool, device
) -> Dict[str, float]:
    """
    Compute comprehensive metrics including self-consistency through sampling.
    """
    model.eval()
    results = {}
    all_sampled_sequences = []
    all_seq_ids = []
    
    # Store per-target results for detailed analysis
    per_target_results = []
    
    # Define which external tools are available
    # TODO: Add checks for tool availability
    tools_available = {
        'eternafold': True,  # Assume available
        'rhofold': True,     # Assume available  
        'vienna': True,      # ViennaRNA tools
        'phenix': True,      # Clash score
        'usalign': True,     # Structure alignment
        'rna_assessment': True,  # INF/lDDT tools
    }
    
    print(f"\nSampling {n_samples} sequences per structure (T={temperature})")
    
    with torch.no_grad():
        for item_idx, item in enumerate(tqdm(dataset, desc="Evaluating structures")):
            target_results = {
                'gid': item.gid,
                'native_seq': item.native_seq_str,
                'sampled_sequences': [],
                'metrics': {}
            }
            
            # Sample multiple sequences for this structure
            sampled_seqs = []
            for sample_idx in range(n_samples):
                sampled_tensor = _sample_sequence(model, item.graph, temperature, device)
                sampled_seq = _seq_to_string(sampled_tensor)
                sampled_seqs.append(sampled_seq)
                
                # Store for FASTA output
                seq_id = f"{item.gid}_sample_{sample_idx}"
                all_sampled_sequences.append(sampled_seq)
                all_seq_ids.append(seq_id)
            
            target_results['sampled_sequences'] = sampled_seqs
            
            # Compute metrics for this target
            target_metrics = {}
            
            # 2D Self-consistency (EternaFold)
            if 'sc_eternafold' in metrics_to_compute and tools_available['eternafold']:
                try:
                    # Get secondary structure list and mask_coords from featurized data
                    true_sec_struct_list = item.raw_data.get('sec_struct_list', ["." * len(item.native_seq_str)])
                    
                    # Get mask_coords from the featurized graph data (like in original implementation)
                    # mask_coords = data.mask_coords.cpu().numpy() in original
                    if hasattr(item.graph, 'mask_coords'):
                        mask_coords = item.graph.mask_coords.cpu().numpy()
                    else:
                        # Fallback: create mask_coords (no masking - all positions valid)
                        seq_len = len(item.native_seq_str)
                        mask_coords = np.ones(seq_len, dtype=bool)
                    
                    # Convert string sequences to numpy arrays of indices for EternaFold
                    # EternaFold expects (n_samples, seq_len) array of nucleotide indices
                    samples_as_indices = []
                    for seq_str in sampled_seqs:
                        seq_indices = [LETTER_TO_NUM.get(c, 0) for c in seq_str]  # fallback to 'A' for unknown
                        samples_as_indices.append(seq_indices)
                    samples_array = np.array(samples_as_indices)
                    
                    # Call EternaFold function (returns np.array, not dict)
                    sc_2d_scores = self_consistency_score_eternafold(
                        samples_array, true_sec_struct_list, mask_coords, return_sec_structs=False
                    )
                    
                    # EternaFold returns array of MCC scores per sample, take mean
                    target_metrics['sc_eternafold_mcc'] = float(sc_2d_scores.mean()) if sc_2d_scores.size > 0 else np.nan
                    target_metrics['sc_eternafold_f1'] = np.nan  # F1 not directly available from this function
                except Exception as e:
                    print(f"Warning: EternaFold failed for {item.gid}: {e}")
                    target_metrics['sc_eternafold_mcc'] = np.nan
                    target_metrics['sc_eternafold_f1'] = np.nan
            
            # 3D Self-consistency (RhoFold)  
            if 'sc_rhofold' in metrics_to_compute and tools_available['rhofold']:
                try:
                    # Create output directory for RhoFold
                    rhofold_out_dir = os.path.join(out_dir, f"rhofold_{item.gid}")
                    os.makedirs(rhofold_out_dir, exist_ok=True)
                    
                    # Note: RhoFold implementation would require model loading
                    # For now, we'll use a simple fallback that indicates these metrics are not computed
                    # 
                    # When RhoFold is properly integrated, use this pattern:
                    # result = self_consistency_score_rhofold(
                    #     samples_array, item.raw_data, mask_coords, rhofold_model, rhofold_out_dir
                    # )
                    # if len(result) == 3:
                    #     rmsd_array, tm_array, gdt_array = result
                    # else:
                    #     rmsd_array, tm_array, gdt_array, plddt_array, inf_dict, clash_scores = result
                    # target_metrics['sc_rhofold_rmsd'] = rmsd_array.mean()
                    # target_metrics['sc_rhofold_tm'] = tm_array.mean() 
                    # target_metrics['sc_rhofold_gdt'] = gdt_array.mean()
                    # target_metrics['sc_rhofold_rmsd_good'] = (rmsd_array <= 2.0).mean()
                    
                    print(f"Note: RhoFold requires model loading - skipping for {item.gid}")
                    target_metrics['sc_rhofold_rmsd'] = np.nan
                    target_metrics['sc_rhofold_tm'] = np.nan
                    target_metrics['sc_rhofold_gdt'] = np.nan
                    target_metrics['sc_rhofold_rmsd_good'] = 0.0
                except Exception as e:
                    print(f"Warning: RhoFold failed for {item.gid}: {e}")
                    target_metrics['sc_rhofold_rmsd'] = np.nan
                    target_metrics['sc_rhofold_tm'] = np.nan
                    target_metrics['sc_rhofold_gdt'] = np.nan
                    target_metrics['sc_rhofold_rmsd_good'] = 0.0
            
            # Thermodynamic metrics (ViennaRNA)
            if any(m in metrics_to_compute for m in ['vienna_mfe', 'vienna_ensemble', 'vienna_tm']):
                vienna_mfe_vals = []
                vienna_ed_vals = []
                vienna_entropy_vals = []
                vienna_tm_vals = []
                
                for seq in sampled_seqs:
                    try:
                        if 'vienna_mfe' in metrics_to_compute:
                            mfe_val, _ = vienna_mfe(seq, T=37.0)
                            vienna_mfe_vals.append(mfe_val)
                            
                        if 'vienna_ensemble' in metrics_to_compute:
                            # Get target secondary structure from native
                            target_db = "." * len(seq)  # Simple fallback
                            if item.raw_data.get('sec_struct_list'):
                                target_db = item.raw_data['sec_struct_list'][0]
                            
                            # Handle length mismatch with hybrid approach
                            seq_len = len(seq)
                            target_len = len(target_db)
                            length_diff = abs(seq_len - target_len)
                            
                            if length_diff <= 3:
                                # Small mismatch: truncate to shorter length
                                min_len = min(seq_len, target_len)
                                seq_trimmed = seq[:min_len]
                                target_db_trimmed = target_db[:min_len]
                                
                                ensemble_metrics = vienna_ensemble_metrics(
                                    seq_trimmed, target_db=target_db_trimmed, T=37.0, return_positional_entropy=False
                                )
                                vienna_ed_vals.append(ensemble_metrics.get('ED', np.nan))
                                vienna_entropy_vals.append(ensemble_metrics.get('shannon_entropy', np.nan))
                            else:
                                # Large mismatch: skip this calculation
                                print(f"Warning: Vienna calculation skipped for {item.gid}: target_db length {target_len} != seq length {seq_len} (diff > 3)")
                                vienna_ed_vals.append(np.nan)
                                vienna_entropy_vals.append(np.nan)
                        
                        if 'vienna_tm' in metrics_to_compute:
                            target_db = "." * len(seq)  # Simple fallback  
                            if item.raw_data.get('sec_struct_list'):
                                target_db = item.raw_data['sec_struct_list'][0]
                            
                            # Handle length mismatch with hybrid approach
                            seq_len = len(seq)
                            target_len = len(target_db)
                            length_diff = abs(seq_len - target_len)
                            
                            if length_diff <= 3:
                                # Small mismatch: truncate to shorter length
                                min_len = min(seq_len, target_len)
                                seq_trimmed = seq[:min_len]
                                target_db_trimmed = target_db[:min_len]
                                
                                tm_val = vienna_Tm_by_pS0(seq_trimmed, target_db_trimmed, Tmin=10, Tmax=95)
                                vienna_tm_vals.append(tm_val)
                            else:
                                # Large mismatch: skip this calculation
                                print(f"Warning: Vienna Tm calculation skipped for {item.gid}: target_db length {target_len} != seq length {seq_len} (diff > 3)")
                                vienna_tm_vals.append(np.nan)
                            
                    except Exception as e:
                        print(f"Warning: Vienna calculation failed for {item.gid}: {e}")
                        continue
                
                if vienna_mfe_vals:
                    target_metrics['vienna_mfe_avg'] = np.mean(vienna_mfe_vals)
                if vienna_ed_vals:
                    target_metrics['vienna_ensemble_ed_avg'] = np.mean([x for x in vienna_ed_vals if not np.isnan(x)])
                if vienna_entropy_vals:
                    target_metrics['vienna_ensemble_entropy_avg'] = np.mean([x for x in vienna_entropy_vals if not np.isnan(x)])
                if vienna_tm_vals:
                    target_metrics['vienna_tm_avg'] = np.mean([x for x in vienna_tm_vals if not np.isnan(x)])
            
            # Sequence diversity (3-mer correlation)
            if 'diversity_3mer' in metrics_to_compute:
                try:
                    # Compute 3-mer correlation with native sequence
                    # Ensure we have valid inputs
                    if sampled_seqs and item.native_seq_str:
                        # Use the same mask_coords as EternaFold for consistency
                        if hasattr(item.graph, 'mask_coords'):
                            mask_coords = item.graph.mask_coords.cpu().numpy()
                        else:
                            # Fallback: create mask_coords matching true sequence length (no masking)
                            seq_len = len(item.native_seq_str)
                            mask_coords = np.ones(seq_len, dtype=bool)
                        
                        diversity_scores = get_three_mer_corr(sampled_seqs, item.native_seq_str, mask_coords)
                        # Take mean of correlation scores across all samples
                        target_metrics['diversity_3mer'] = float(np.mean(diversity_scores)) if diversity_scores.size > 0 else np.nan
                    else:
                        print(f"Warning: Invalid input for diversity calculation for {item.gid}")
                        target_metrics['diversity_3mer'] = np.nan
                except Exception as e:
                    print(f"Warning: Diversity calculation failed for {item.gid}: {e}")
                    target_metrics['diversity_3mer'] = np.nan
            
            # Additional structural metrics (requires folding sequences to structures)
            if any(m in metrics_to_compute for m in ['clash_score', 'inf_scores', 'lddt_score', 'usalign_tm', 'mcq_torsion']):
                # Note: These metrics require 3D structure prediction for sampled sequences
                # For now, we'll skip them as they require RhoFold integration
                # TODO: Implement when RhoFold pipeline is ready
                print(f"Note: Structural metrics (clash_score, inf_scores, etc.) require 3D folding - skipping for now")
                for metric in ['clash_score', 'inf_scores', 'lddt_score', 'usalign_tm', 'mcq_torsion']:
                    if metric in metrics_to_compute:
                        target_metrics[f'{metric}_avg'] = np.nan
            
            target_results['metrics'] = target_metrics
            per_target_results.append(target_results)
    
    # Aggregate results across all targets
    print("\nAggregating results across targets...")
    
    for metric_name in ['sc_eternafold_mcc', 'sc_eternafold_f1', 'sc_rhofold_rmsd', 'sc_rhofold_tm', 
                       'sc_rhofold_gdt', 'sc_rhofold_rmsd_good', 'vienna_mfe_avg', 
                       'vienna_ensemble_ed_avg', 'vienna_ensemble_entropy_avg', 'vienna_tm_avg',
                       'diversity_3mer', 'clash_score_avg', 'inf_scores_avg', 'lddt_score_avg',
                       'usalign_tm_avg', 'mcq_torsion_avg']:
        values = [r['metrics'].get(metric_name, np.nan) for r in per_target_results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            results[metric_name] = np.mean(valid_values)
            results[f"{metric_name}_std"] = np.std(valid_values)
            results[f"{metric_name}_n"] = len(valid_values)
        else:
            results[metric_name] = np.nan
            results[f"{metric_name}_std"] = np.nan
            results[f"{metric_name}_n"] = 0
    
    # Save detailed results
    detailed_results_path = os.path.join(out_dir, "detailed_results.json")
    with open(detailed_results_path, 'w') as f:
        json.dump(per_target_results, f, indent=2, default=str)
    
    # Save designed sequences if requested
    if save_designs and all_sampled_sequences:
        designs_dir = os.path.join(out_dir, "designs")
        os.makedirs(designs_dir, exist_ok=True)
        sequences_path = os.path.join(designs_dir, "sampled_sequences.fasta")
        _save_sequence_fasta(all_sampled_sequences, all_seq_ids, sequences_path)
        print(f"Saved {len(all_sampled_sequences)} sequences to {sequences_path}")
    
    return results


def _load_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    # two formats: (A) state_dict directly; (B) {"model": state_dict, ...}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return model

@torch.no_grad()
def eval_comprehensive_checkpoint(
    cfg: SN, 
    dataset: ComprehensiveDataset, 
    ckpt_name: str, 
    ckpt_path: str, 
    device,
    out_dir: str
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a checkpoint with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    # Load model
    model = build_model_from_cfg(cfg.model).to(device)
    _load_weights(model, ckpt_path, device)
    model.eval()
    
    # Create output directory for this checkpoint
    ckpt_out_dir = os.path.join(out_dir, ckpt_name)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    
    # Get evaluation configuration
    eval_cfg = cfg.eval
    n_samples = getattr(eval_cfg, 'n_samples', 8)
    temperature = getattr(eval_cfg, 'temperature', 0.5)
    metrics_to_compute = getattr(eval_cfg, 'metrics', ['recovery', 'perplexity'])
    save_designs = getattr(eval_cfg, 'save_designs', False)
    
    print(f"Configuration:")
    print(f"  - n_samples: {n_samples}")
    print(f"  - temperature: {temperature}")
    print(f"  - metrics: {metrics_to_compute}")
    print(f"  - save_designs: {save_designs}")
    print(f"  - output_dir: {ckpt_out_dir}")
    
    results = {}
    
    # Always compute basic metrics (fast)
    print(f"\n[1/2] Computing basic metrics...")
    basic_metrics = _compute_basic_metrics(model, dataset, device)
    results.update(basic_metrics)
    
    # Compute comprehensive metrics if requested
    needs_sampling = any(m in metrics_to_compute for m in [
        'sc_eternafold', 'sc_rhofold', 'vienna_mfe', 'vienna_ensemble', 
        'vienna_tm', 'diversity_3mer', 'clash_score', 'inf_scores', 
        'lddt_score', 'usalign_tm', 'mcq_torsion'
    ])
    
    if needs_sampling:
        print(f"[2/2] Computing comprehensive metrics...")
        comprehensive_metrics = _compute_comprehensive_metrics(
            model, dataset, n_samples, temperature, 
            metrics_to_compute, ckpt_out_dir, save_designs, device
        )
        results.update(comprehensive_metrics)
    else:
        print(f"[2/2] Skipping comprehensive metrics (not requested)")
    
    # Save checkpoint-specific results
    results_path = os.path.join(ckpt_out_dir, "metrics_summary.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Checkpoint evaluation complete")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive RNA model evaluation")
    parser.add_argument("--config", required=True, type=str, help="Path to configuration file")
    args = parser.parse_args()

    print("="*80)
    print("RNA COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)

    cfg = load_cfg(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration loaded: {args.config}")
    print(f"Device: {device}")
    print(f"Split: {cfg.paths.split_name}")

    # Load dataset with comprehensive capabilities
    print(f"\nLoading dataset...")
    dataset = ComprehensiveDataset(
        cfg.paths.processed_pt, 
        cfg.paths.split_pt, 
        cfg.paths.split_name, 
        cfg.featurizer, 
        device=device
    )
    print(f"Loaded {len(dataset)} structures for evaluation")

    # Setup output directory
    out_dir = getattr(cfg.eval, 'out_dir', 'dpo/eval_results/comprehensive_eval')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # W&B initialization
    wb = getattr(cfg.eval, 'wandb', None)
    use_wandb = getattr(wb, "enable", False) if wb else False
    if use_wandb:
        run_name = getattr(wb, 'run_name', None) or f"comprehensive_eval_{cfg.paths.split_name}_{_human_ts()}"
        wandb.init(
            project=getattr(wb, 'project', 'DPO-RNA'), 
            entity=getattr(wb, 'entity', None), 
            name=run_name,
            tags=getattr(wb, 'tags', ['comprehensive_eval'])
        )
        print(f"W&B initialized: {run_name}")

    # Evaluate all checkpoints
    all_results = []
    for ck in cfg.paths.checkpoints:
        name, path = ck.name, ck.path
        
        try:
            results = eval_comprehensive_checkpoint(cfg, dataset, name, path, device, out_dir)
            
            # Prepare row for summary table
            row = {
                "ckpt_name": name, 
                "ckpt_path": path, 
                "split": cfg.paths.split_name,
                **{k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else v 
                   for k, v in results.items()}
            }
            all_results.append(row)
            
            # Print summary for this checkpoint
            print(f"\n📊 {name} Summary:")
            if 'recovery' in results:
                print(f"  Recovery: {results['recovery']:.4f}")
            if 'perplexity' in results:
                print(f"  Perplexity: {results['perplexity']:.2f}")
            if 'sc_eternafold_mcc' in results:
                print(f"  2D Self-consistency (MCC): {results['sc_eternafold_mcc']:.3f}")
            if 'sc_rhofold_rmsd' in results:
                print(f"  3D Self-consistency (RMSD): {results['sc_rhofold_rmsd']:.2f}Å")
            if 'vienna_mfe_avg' in results:
                print(f"  MFE: {results['vienna_mfe_avg']:.1f} kcal/mol")
            
            # Log to W&B
            if use_wandb:
                wandb_metrics = {f"{name}/{k}": v for k, v in results.items() 
                               if isinstance(v, (int, float)) and not np.isnan(v)}
                wandb.log(wandb_metrics)
                
        except Exception as e:
            print(f"ERROR evaluating {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save comprehensive CSV summary
    if all_results:
        # Create comprehensive CSV with all metrics
        summary_csv = os.path.join(out_dir, "metrics_summary.csv")
        
        # Get all possible columns from all results
        all_columns = set()
        for row in all_results:
            all_columns.update(row.keys())
        all_columns = sorted(list(all_columns))
        
        df = pd.DataFrame(all_results)
        df = df.reindex(columns=all_columns, fill_value=np.nan)
        df.to_csv(summary_csv, index=False)
        print(f"\n💾 Saved comprehensive results to: {summary_csv}")
        
        # Create a simplified summary for quick viewing
        basic_columns = ['ckpt_name', 'recovery', 'perplexity', 'sc_eternafold_mcc', 
                        'sc_rhofold_rmsd', 'sc_rhofold_tm', 'vienna_mfe_avg']
        basic_df = df[[col for col in basic_columns if col in df.columns]]
        basic_summary_csv = os.path.join(out_dir, "basic_summary.csv")
        basic_df.to_csv(basic_summary_csv, index=False)
        print(f"💾 Saved basic summary to: {basic_summary_csv}")

        # W&B table
        if use_wandb:
            table = wandb.Table(dataframe=basic_df)
            wandb.log({"evaluation_summary": table})

    if use_wandb:
        wandb.finish()

    print(f"\n🎉 Comprehensive evaluation complete!")
    print(f"Results available in: {out_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
