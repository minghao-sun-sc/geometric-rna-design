#!/usr/bin/env python
"""
Trace the SFT loss computation to identify the exact issue.
"""

import sys
import torch
import torch.nn.functional as F
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

def debug_sft_loss(logits, targets, mask=None):
    """Debug version of SFT loss computation"""
    print(f"\n=== DEBUG SFT LOSS ===")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Logits dtype: {logits.dtype}, device: {logits.device}")
    print(f"Targets dtype: {targets.dtype}, device: {targets.device}")
    
    # Check target values
    unique_targets = torch.unique(targets)
    print(f"Unique target values: {unique_targets.tolist()}")
    
    vocab_size = logits.shape[-1]
    print(f"Vocabulary size (from logits): {vocab_size}")
    
    # Check for out of bounds
    oob_mask = targets >= vocab_size
    if oob_mask.any():
        n_oob = oob_mask.sum().item()
        oob_values = targets[oob_mask].unique()
        print(f"\n❌ Found {n_oob} out-of-bounds target tokens!")
        print(f"   Out-of-bounds values: {oob_values.tolist()}")
        print(f"   These are >= vocab_size ({vocab_size})")
        
        # Show positions
        oob_positions = oob_mask.nonzero(as_tuple=True)[0]
        print(f"   Positions with OOB tokens: {oob_positions[:10].tolist()}...")
    else:
        print(f"✅ All target tokens are within bounds [0, {vocab_size-1}]")
    
    # Try to compute loss step by step
    try:
        # Step 1: Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        print(f"\n✅ Log softmax successful")
        print(f"   Log probs shape: {log_probs.shape}")
        
        # Step 2: Clamp targets to be safe
        targets_safe = torch.clamp(targets, 0, vocab_size - 1)
        if not torch.equal(targets, targets_safe):
            n_clamped = (targets != targets_safe).sum().item()
            print(f"\n⚠️ Clamped {n_clamped} target tokens")
            
        # Step 3: Gather
        print(f"\nAttempting gather operation...")
        print(f"  targets_safe min: {targets_safe.min()}, max: {targets_safe.max()}")
        
        # Debug: manually check a few indices
        for i in range(min(5, targets_safe.shape[0])):
            target_idx = targets_safe[i].item()
            print(f"  Position {i}: target_idx={target_idx}, log_prob={log_probs[i, target_idx].item():.4f}")
        
        # Actual gather
        gathered = torch.gather(log_probs, dim=-1, index=targets_safe.unsqueeze(-1))
        print(f"✅ Gather successful!")
        print(f"   Gathered shape: {gathered.shape}")
        
        nll = -gathered.squeeze(-1)
        
        if mask is not None:
            nll = nll * mask
            n_valid = mask.sum().item()
            print(f"\nApplied mask: {n_valid}/{mask.shape[0]} valid positions")
        
        loss = nll.mean()
        print(f"\n✅ Loss computed successfully: {loss.item():.4f}")
        
        return loss
        
    except Exception as e:
        print(f"\n❌ Error during loss computation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    dc = cfg["data"]
    
    # Load dataset
    dataset = PreferencePairDataset(
        processed_pt=dc["processed_pt"],
        split_file=dc["split_file"],
        pairs_path=dc["pairs_path_train"],
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Same as training
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=0
    )
    
    # Load model
    model = instantiate(cfg["model"])
    model = model.to("cpu")
    model.eval()
    
    print("=== TESTING SFT LOSS COMPUTATION ===")
    
    # Get a batch
    batch = next(iter(dataloader))
    batch = batch.to("cpu")
    
    print(f"\nBatch info:")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Sequence shape: {batch.seq.shape}")
    print(f"  y_w shape: {batch.y_w.shape}")
    print(f"  y_l shape: {batch.y_l.shape}")
    print(f"  node_mask shape: {batch.node_mask.shape}")
    
    # Forward pass
    print(f"\n=== MODEL FORWARD ===")
    with torch.no_grad():
        logits = model(batch)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Model output dim: {logits.shape[-1]}")
    
    # Test SFT loss on winner sequences
    print(f"\n=== TESTING SFT LOSS ON WINNERS ===")
    loss_w = debug_sft_loss(logits, batch.y_w, batch.node_mask)
    
    # Test SFT loss on loser sequences
    print(f"\n=== TESTING SFT LOSS ON LOSERS ===")
    loss_l = debug_sft_loss(logits, batch.y_l, batch.node_mask)
    
    # Additional checks
    print(f"\n=== ADDITIONAL CHECKS ===")
    
    # Check if target sequences have been properly encoded
    print(f"\nChecking sequence encoding...")
    
    # Get first item's raw sequences
    first_pair = dataset.pairs[0]
    winner_seq = first_pair["winner_seq"]
    loser_seq = first_pair["loser_seq"]
    
    print(f"First pair sequences:")
    print(f"  Winner: {winner_seq[:20]}... (len={len(winner_seq)})")
    print(f"  Loser: {loser_seq[:20]}... (len={len(loser_seq)})")
    
    # Check encoding
    from dpo.data import _normalize_seq
    norm_winner = _normalize_seq(winner_seq)
    norm_loser = _normalize_seq(loser_seq)
    
    print(f"\nNormalized sequences:")
    print(f"  Winner: {norm_winner[:20]}...")
    print(f"  Loser: {norm_loser[:20]}...")
    
    # Check for unknown characters
    vocab = set(dataset.letter_to_num.keys())
    winner_chars = set(norm_winner)
    loser_chars = set(norm_loser)
    
    unknown_w = winner_chars - vocab
    unknown_l = loser_chars - vocab
    
    if unknown_w:
        print(f"\n⚠️ Unknown characters in winner: {unknown_w}")
    if unknown_l:
        print(f"\n⚠️ Unknown characters in loser: {unknown_l}")

if __name__ == "__main__":
    main()