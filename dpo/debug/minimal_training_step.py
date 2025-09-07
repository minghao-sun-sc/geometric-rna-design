#!/usr/bin/env python
"""
Minimal replication of the exact training step that's failing.
"""

import sys
import torch
import yaml
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from dpo.patches import patch_featurizer_three_bead
patch_featurizer_three_bead()

sys.path.append(str(Path(__file__).parent.parent.parent))

from dpo.data import PreferencePairDataset, collate_pairs  
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from dpo.losses import dpo_sft_step
from dpo.lightning_module import _unwrap_batch

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    dc = cfg["data"]
    
    # Exact same dataset setup
    dataset = PreferencePairDataset(
        processed_pt=dc["processed_pt"],
        split_file=dc["split_file"],
        pairs_path=dc["pairs_path_train"],  # Use train set like real training
        split="train",
        max_num_conformers=dc["max_num_conformers"],
        radius=dc["radius"],
        top_k=dc["top_k"],
        num_rbf=dc["num_rbf"],
        num_posenc=dc["num_posenc"],
        noise_scale=dc["noise_scale"],
        device="cpu",
        use_seq_mask=dc.get("use_seq_mask", True),
        strict_length_check=bool(dc.get("strict_length_check", False)),
        window_align=bool(dc.get("window_align", True)),
        min_window_identity=float(dc.get("min_window_identity", 0.7)),
    )
    
    # Exact same dataloader setup  
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Same as config
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=0  # Disable multiprocessing to avoid worker issues
    )
    
    device = torch.device("cuda")
    
    # Load policy and reference models
    print("Loading models...")
    policy = instantiate(cfg["model"]).to(device)
    ref_model = instantiate(cfg["model"]).to(device)
    
    # Copy policy weights to reference (like Lightning module does)
    ref_model.load_state_dict(policy.state_dict())
    ref_model.eval()
    policy.train()
    
    print("Testing exact training step...")
    
    for batch in dataloader:
        print(f"Got batch with {batch.num_graphs} graphs")
        
        # Exact unwrapping like Lightning module
        batch = _unwrap_batch(batch)
        batch = batch.to(device)  # Move to GPU like Lightning does
        
        try:
            # Exact same DPO step call
            out = dpo_sft_step(
                policy,
                batch, 
                beta=0.2,  # From config
                lambda_sft=0.153,  # From config  
                length_norm=False,  # From config
                ref_model=ref_model,
                train=True,
            )
            
            print(f"SUCCESS! DPO step completed")
            print(f"Loss: {out['loss'].item():.4f}")
            print(f"DPO loss: {out.get('dpo_loss', 'N/A')}")
            print(f"SFT loss: {out.get('sft_loss', 'N/A')}")
            
        except Exception as e:
            print(f"ERROR in dpo_sft_step: {e}")
            
            # Debug the batch structure
            print(f"\nBATCH DEBUG:")
            print(f"Type: {type(batch)}")
            if hasattr(batch, 'y_w'):
                print(f"y_w shape: {batch.y_w.shape}, range: {batch.y_w.min()}-{batch.y_w.max()}")
            if hasattr(batch, 'y_l'):  
                print(f"y_l shape: {batch.y_l.shape}, range: {batch.y_l.min()}-{batch.y_l.max()}")
            if hasattr(batch, 'node_mask'):
                print(f"node_mask shape: {batch.node_mask.shape}")
            if hasattr(batch, 'seq'):
                print(f"seq shape: {batch.seq.shape}, range: {batch.seq.min()}-{batch.seq.max()}")
            
            import traceback
            traceback.print_exc()
        
        break

if __name__ == "__main__":
    main()