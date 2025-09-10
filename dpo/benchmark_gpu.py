#!/usr/bin/env python3
"""
Benchmark GPU utilization and memory usage with different batch sizes.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import argparse
import time
import torch
import numpy as np
from types import SimpleNamespace as SN
import yaml

from dpo.data import build_dataloaders
from dpo.ref_manager import build_policy_and_reference
from dpo.losses import dpo_step_losses
from torch.cuda.amp import autocast


def load_cfg(path: str) -> SN:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_sn(raw)


def _to_sn(o):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o


def benchmark_batch_size(config_path: str, num_batches: int = 100):
    """
    Benchmark training with a specific configuration.
    
    Returns:
        dict: Benchmark results including throughput and memory usage
    """
    cfg = load_cfg(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build data and models
    print(f"\nTesting batch_size={cfg.training.batch_size}, num_workers={cfg.training.num_workers}")
    train_loader, _, _, meta = build_dataloaders(cfg, device=device)
    policy, reference = build_policy_and_reference(cfg, device)
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], 
                     dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
            out = dpo_step_losses(
                model=policy,
                ref_model=reference,
                batch=batch,
                beta=cfg.dpo.beta,
                label_smoothing=cfg.dpo.label_smoothing,
                max_len=cfg.dpo.max_len
            )
            loss = out["loss_dpo"]
            loss.backward()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    batch_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
            
        torch.cuda.synchronize()
        start_time = time.time()
        
        with autocast(enabled=cfg.training.precision in ["fp16", "bf16"],
                     dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
            out = dpo_step_losses(
                model=policy,
                ref_model=reference,
                batch=batch,
                beta=cfg.dpo.beta,
                label_smoothing=cfg.dpo.label_smoothing,
                max_len=cfg.dpo.max_len
            )
            loss = out["loss_dpo"]
            loss.backward()
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        if (i + 1) % 10 == 0:
            avg_time = np.mean(batch_times[-10:])
            throughput = cfg.training.batch_size / avg_time
            print(f"  Batch {i+1}/{num_batches}: {avg_time:.3f}s/batch, {throughput:.1f} samples/s")
    
    # Calculate metrics
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    throughput = cfg.training.batch_size / avg_batch_time
    
    # Memory usage
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    results = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "avg_batch_time": avg_batch_time,
        "std_batch_time": std_batch_time,
        "throughput_samples_per_sec": throughput,
        "peak_memory_gb": peak_memory,
        "current_memory_gb": current_memory,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=[
        "dpo/configs/defaults.yaml",
        "dpo/configs/batch_8.yaml",
        "dpo/configs/batch_16.yaml"
    ])
    parser.add_argument("--num_batches", type=int, default=50)
    args = parser.parse_args()
    
    print("GPU Utilization Benchmark")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("WARNING: No GPU available, running on CPU")
    
    all_results = []
    
    for config_path in args.configs:
        print(f"\nTesting config: {config_path}")
        print("-" * 40)
        
        try:
            results = benchmark_batch_size(config_path, args.num_batches)
            all_results.append(results)
            
            print(f"\nResults for batch_size={results['batch_size']}:")
            print(f"  Avg batch time: {results['avg_batch_time']:.3f} ± {results['std_batch_time']:.3f} s")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/s")
            print(f"  Peak memory: {results['peak_memory_gb']:.2f} GB")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            if "out of memory" in str(e).lower():
                print(f"  Batch size {config_path} is too large for this GPU")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Batch Size':<12} {'Throughput':<15} {'Memory (GB)':<12} {'Time/Batch':<12}")
    print("-" * 51)
    
    for r in all_results:
        print(f"{r['batch_size']:<12} {r['throughput_samples_per_sec']:<15.1f} "
              f"{r['peak_memory_gb']:<12.2f} {r['avg_batch_time']:<12.3f}")
    
    # Find optimal
    if all_results:
        best = max(all_results, key=lambda x: x['throughput_samples_per_sec'])
        print(f"\nOptimal batch size: {best['batch_size']} "
              f"({best['throughput_samples_per_sec']:.1f} samples/s)")


if __name__ == "__main__":
    main()