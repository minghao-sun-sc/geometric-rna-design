#!/usr/bin/env python3
"""
Debug script to understand RhoFold self-consistency issues
"""

import os
import sys
sys.path.insert(0, "/mnt/rna01/smh/projects/offline-dpo")

import numpy as np
import torch
from types import SimpleNamespace as SN

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from dpo.bench.eval_full import load_cfg, FullEvalDataset
from dpo.ref_manager import build_model_from_cfg
from src.evaluator import self_consistency_score_rhofold
from src.data.data_utils import get_c4p_coords

def debug_rhofold_evaluation():
    """Debug RhoFold evaluation to understand high RMSD issues"""
    
    # Load config and dataset
    cfg = load_cfg("dpo/configs/bench_full_test.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load first item only for debugging
    ds = FullEvalDataset(
        cfg.paths.processed_pt,
        cfg.paths.split_pt,
        cfg.paths.split_name,
        cfg.featurizer,
        device=device
    )
    
    print(f"Dataset size: {len(ds)}")
    if len(ds) == 0:
        print("No data to debug")
        return
    
    # Get first item
    item = ds[0]
    print(f"\nDebugging item: {item.gid}")
    print(f"Sequence length: {len(item.seq)}")
    print(f"Graph sequence length: {len(item.graph.seq)}")
    
    # Check raw data structure
    raw_data = item.raw_data
    print(f"Raw data keys: {raw_data.keys()}")
    print(f"Number of coordinates: {len(raw_data['coords_list'])}")
    
    # Check mask_coords
    if hasattr(item.graph, 'mask_coords'):
        mask_coords = item.graph.mask_coords.cpu().numpy()
        print(f"Mask coords shape: {mask_coords.shape}")
        print(f"Mask coords sum: {mask_coords.sum()} / {len(mask_coords)}")
    else:
        mask_coords = np.ones(len(item.seq), dtype=bool)
        print(f"No mask_coords found, using all True mask of length {len(mask_coords)}")
    
    # Check coordinate shapes
    for i, coords in enumerate(raw_data['coords_list']):
        print(f"Coords {i} shape: {coords.shape}")
        c4p_coords = get_c4p_coords(coords)
        print(f"C4' coords {i} shape: {c4p_coords.shape}")
        masked_c4p = c4p_coords[mask_coords, :]
        print(f"Masked C4' coords {i} shape: {masked_c4p.shape}")
    
    # Load model and sample
    model = build_model_from_cfg(cfg.model).to(device)
    ckpt_path = cfg.paths.checkpoints[0].path
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    # Sample 2 sequences
    n_samples = 2
    with torch.no_grad():
        graph = item.graph.clone()
        graph.seq = item.seq
        samples, _ = model.sample(graph, n_samples, temperature=1.0)
        
    print(f"\nSampled sequences shape: {samples.shape}")
    
    # Convert to numpy for evaluation
    samples_np = samples.cpu().numpy()
    
    # Print first few characters of sequences
    from src.constants import NUM_TO_LETTER
    print("Ground truth sequence:", "".join([NUM_TO_LETTER[x] for x in item.seq.cpu().numpy()[:20]]) + "...")
    if samples_np.ndim == 1:
        print(f"Sample sequence:   ", "".join([NUM_TO_LETTER[x] for x in samples_np[:20]]) + "...")
        samples_np = samples_np.reshape(1, -1)  # Reshape to (1, seq_len)
    else:
        for i, seq in enumerate(samples_np):
            print(f"Sample {i} sequence:   ", "".join([NUM_TO_LETTER[x] for x in seq[:20]]) + "...")
    
    # Try RhoFold evaluation on this single item
    print(f"\nTesting RhoFold evaluation...")
    try:
        # Initialize RhoFold
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        from src.constants import PROJECT_PATH
        
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
        rhofold = rhofold.to(device)
        rhofold.eval()
        
        # Run evaluation 
        output_dir = "dpo/debug/rhofold_debug"
        sc_rmsd, sc_tm, sc_gdt = self_consistency_score_rhofold(
            samples_np,
            raw_data,
            mask_coords,
            rhofold,
            output_dir,
            save_designs=True,
            save_pdbs=True,
            use_relax=False
        )
        
        print(f"Self-consistency results:")
        print(f"  RMSD: {sc_rmsd} (avg: {sc_rmsd.mean():.2f})")
        print(f"  TM-score: {sc_tm} (avg: {sc_tm.mean():.4f})")
        print(f"  GDT: {sc_gdt} (avg: {sc_gdt.mean():.4f})")
        
        print(f"Debug files saved to: {output_dir}")
        
    except Exception as e:
        print(f"RhoFold evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rhofold_evaluation()