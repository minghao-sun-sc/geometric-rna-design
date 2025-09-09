#!/usr/bin/env python3
"""
Analyze sequence quality and diversity to understand evaluation results
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
from src.constants import NUM_TO_LETTER

def sequence_distance(seq1, seq2):
    """Compute Hamming distance between two sequences"""
    return (seq1 != seq2).sum() / len(seq1)

def analyze_sequence_quality():
    """Analyze sequence quality and diversity"""
    
    # Load config and dataset
    cfg = load_cfg("dpo/configs/bench_full_test.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets for comparison
    print("Loading test dataset...")
    ds_test = FullEvalDataset(
        cfg.paths.processed_pt,
        cfg.paths.split_pt,
        "test",
        cfg.featurizer,
        device=device
    )
    
    print("Loading val dataset...")
    cfg_val = cfg
    cfg_val.paths.split_name = "val"
    ds_val = FullEvalDataset(
        cfg.paths.processed_pt,
        cfg.paths.split_pt,
        "val", 
        cfg.featurizer,
        device=device
    )
    
    print(f"Test dataset size: {len(ds_test)}")
    print(f"Val dataset size: {len(ds_val)}")
    
    # Load model
    model = build_model_from_cfg(cfg.model).to(device)
    ckpt_path = cfg.paths.checkpoints[0].path
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    def analyze_dataset(ds, name, n_items=None):
        print(f"\n=== {name.upper()} DATASET ANALYSIS ===")
        
        if n_items is None:
            n_items = len(ds)
        n_items = min(n_items, len(ds))
        
        sequence_lengths = []
        recoveries = []
        sequence_distances = []
        
        with torch.no_grad():
            for i in range(n_items):
                item = ds[i]
                seq_len = len(item.seq)
                sequence_lengths.append(seq_len)
                
                # Sample 2 sequences
                graph = item.graph.clone()
                graph.seq = item.seq
                samples, _ = model.sample(graph, 2, temperature=1.0)
                
                if samples.ndim == 1:
                    samples = samples.unsqueeze(0)
                
                # Compute recovery for each sample
                sample_recoveries = []
                sample_distances = []
                for j in range(samples.shape[0]):
                    sample = samples[j]
                    recovery = (sample == item.seq).float().mean().item()
                    recoveries.append(recovery)
                    sample_recoveries.append(recovery)
                    
                    # Compute sequence distance
                    distance = sequence_distance(sample.cpu().numpy(), item.seq.cpu().numpy())
                    sequence_distances.append(distance)
                    sample_distances.append(distance)
                
                if i < 3:  # Show first 3 examples
                    print(f"\nExample {i+1} ({item.gid}):")
                    print(f"  Length: {seq_len}")
                    gt_seq = "".join([NUM_TO_LETTER[x] for x in item.seq.cpu().numpy()])
                    print(f"  Ground truth: {gt_seq[:50]}{'...' if len(gt_seq) > 50 else ''}")
                    
                    for j in range(samples.shape[0]):
                        sample_seq = "".join([NUM_TO_LETTER[x] for x in samples[j].cpu().numpy()])
                        recovery = sample_recoveries[j]
                        distance = sample_distances[j]
                        print(f"  Sample {j+1}:     {sample_seq[:50]}{'...' if len(sample_seq) > 50 else ''}")
                        print(f"    Recovery: {recovery:.3f}, Distance: {distance:.3f}")
        
        # Summary statistics
        print(f"\n{name} Summary:")
        print(f"  Sequence lengths: {np.mean(sequence_lengths):.1f} ± {np.std(sequence_lengths):.1f} ({np.min(sequence_lengths)}-{np.max(sequence_lengths)})")
        print(f"  Recovery: {np.mean(recoveries):.3f} ± {np.std(recoveries):.3f}")
        print(f"  Sequence distance: {np.mean(sequence_distances):.3f} ± {np.std(sequence_distances):.3f}")
        print(f"  High recovery (>0.5): {(np.array(recoveries) > 0.5).mean():.1%}")
        print(f"  Low distance (<0.3): {(np.array(sequence_distances) < 0.3).mean():.1%}")
        
        return {
            "lengths": sequence_lengths,
            "recoveries": recoveries,
            "distances": sequence_distances
        }
    
    # Analyze both datasets
    test_stats = analyze_dataset(ds_test, "TEST", n_items=5)
    val_stats = analyze_dataset(ds_val, "VAL", n_items=20)
    
    # Overall comparison
    print(f"\n=== COMPARISON ===")
    print(f"Test vs Val recovery: {np.mean(test_stats['recoveries']):.3f} vs {np.mean(val_stats['recoveries']):.3f}")
    print(f"Test vs Val distance: {np.mean(test_stats['distances']):.3f} vs {np.mean(val_stats['distances']):.3f}")
    
    # Expected vs actual performance
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print(f"Expected gRNAde recovery: ~45-60% (from literature)")
    print(f"Actual recovery: {np.mean(val_stats['recoveries']):.1%}")
    print(f"Expected 2D self-consistency: ~65-85%")
    print(f"Actual 2D self-consistency: ~54% (val)")
    print(f"Expected 3D self-consistency: Variable, often poor for diverse sequences")
    print(f"Actual 3D self-consistency: Very poor (~21Å RMSD)")
    
    print(f"\n=== CONCLUSIONS ===")
    if np.mean(val_stats['recoveries']) < 0.40:
        print("❌ Recovery is lower than expected - model may need more training")
    else:
        print("✅ Recovery is reasonable for a base model")
        
    if np.mean(val_stats['distances']) > 0.6:
        print("❌ Sequence distances are high - model generates very different sequences")
    else:
        print("✅ Sequence distances are reasonable")
        
    print("📋 3D self-consistency is poor but this is expected when:")
    print("   - Generated sequences are very different from ground truth")
    print("   - RhoFold fails to predict correct 3D structure for novel sequences")
    print("   - This is a common limitation in inverse folding evaluation")

if __name__ == "__main__":
    analyze_sequence_quality()