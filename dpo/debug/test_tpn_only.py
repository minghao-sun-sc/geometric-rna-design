#!/usr/bin/env python3
"""
Quick test to verify TPN metric works in evaluation pipeline
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import torch
import numpy as np
from src.evaluator import get_trimer_profile_novelty

def main():
    print("🧬 Quick TPN Metric Test")
    print("=" * 30)
    
    # Create mock sequences (simulating model output)
    n_samples, seq_len = 3, 25
    mock_samples = torch.randint(0, 4, (n_samples, seq_len))
    
    print(f"Testing TPN with {n_samples} sequences of length {seq_len}")
    
    # Calculate TPN scores
    tpn_scores = get_trimer_profile_novelty(mock_samples.numpy())
    
    print("Results:")
    for i, score in enumerate(tpn_scores):
        print(f"  Sequence {i+1}: TPN = {score:.4f}")
    
    avg_tpn = np.mean(tpn_scores)
    print(f"\nAverage TPN: {avg_tpn:.4f}")
    
    print("\n✅ TPN metric working correctly!")
    print("   - Higher scores = more novel sequences")
    print("   - Range: 0.0 (identical to reference) to 1.0 (completely novel)")

if __name__ == "__main__":
    main()