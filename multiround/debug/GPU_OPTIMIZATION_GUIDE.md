# 🚀 GPU Optimization Guide for Fast Debug Training

## 📊 Current Issues & Solutions

### Problem: Low GPU Utilization (~30% on A100)
**Root Causes:**
- Small batch size (16) → GPU underutilized
- High gradient accumulation (4) → Many small forward passes
- Frequent I/O operations (saves/validation every 50 steps)
- CPU-bound data loading bottleneck
- Expensive evaluations blocking training

## ✅ Applied Optimizations in `00_debug_fast_gpu.yaml`

### 1. **Batch Size & Memory Optimization** 
```yaml
# OLD (30% utilization)
batch_size: 16
grad_accum_steps: 4  # Effective batch = 64

# NEW (should get 70-90% utilization)
batch_size: 64       # 4x larger - A100 has 40GB/80GB memory
grad_accum_steps: 1  # No accumulation needed
```
**Impact**: Direct 4x increase in GPU compute per step

### 2. **Data Loading Optimization**
```yaml
# OLD
num_workers: 8
prefetch_factor: 2 (default)
persistent_workers: false (default)

# NEW  
num_workers: 16           # 2x more workers
prefetch_factor: 4        # Prefetch more batches
persistent_workers: true  # Keep workers alive
```
**Impact**: Eliminates data loading bottleneck

### 3. **Reduce I/O Overhead**
```yaml
# OLD
save_every: 50
val_every: 50
log_every: 1

# NEW
save_every: 200     # 4x less frequent
val_every: 200      # 4x less frequent  
log_every: 10       # 10x less logging
```
**Impact**: Less time spent on disk I/O

### 4. **Speed Up Evaluation**
```yaml
# Reduced samples
n_samples_eval: 4         # Was 8
n_samples_final_eval: 8   # Was 64

# Skip expensive metrics
metrics: [recovery, perplexity, sc_eternafold]  # Skip sc_rhofold, vienna
use_lddt: false          # Skip expensive lDDT calculation
save_designs: false      # Skip saving PDB files
```
**Impact**: Evaluation 8-10x faster

### 5. **Training Data Reduction**
```yaml
epochs_per_round: 1      # Was 2
data_subset_ratio: 0.25  # Use 25% of data for debug
```
**Impact**: 8x less data to process

## 🔧 Additional Optimization Strategies

### A. **Advanced GPU Optimizations**
```python
# 1. Enable torch.compile (if stable)
training:
  compile: true  # Can give 10-30% speedup

# 2. Use larger batch if memory allows
batch_size: 128  # Try if A100 has 80GB

# 3. Enable automatic mixed precision (AMP) optimizer
optimizer:
  use_8bit: true  # If using bitsandbytes
```

### B. **Data Pipeline Optimizations**
```python
# 1. Cache dataset in memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 2. Use faster data format (if possible)
# Convert JSONL to binary format like .pt or .npz

# 3. Enable CUDA graphs for small models
training:
  use_cuda_graphs: true  # For consistent input shapes
```

### C. **Multi-GPU Strategies** (if available)
```bash
# Use DataParallel or DistributedDataParallel
torchrun --nproc_per_node=2 -m multiround.train --config ...

# Or with SLURM
srun -N1 -n2 python -m multiround.train --config ...
```

### D. **Profile & Monitor**
```python
# Add profiling to find bottlenecks
import torch.profiler

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as prof:
    # training loop
    
# Monitor with nvidia-smi
watch -n 0.5 nvidia-smi
```

## 📈 Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | ~30% | 70-90% | **2.3-3x** |
| Steps/second | ~2-3 | ~8-12 | **3-4x** |
| Time per epoch | ~45 min | ~5-8 min | **6-9x faster** |
| Total debug time | ~3 hours | ~20-30 min | **6-9x faster** |
| Memory usage | ~8GB | ~20-30GB | Better utilization |

## 🎯 Quick Commands

### Fastest Debug (1 round, minimal):
```bash
python -m multiround.train \
  --config multiround/config/experiments/00_debug_fast_gpu.yaml \
  --override multiround.num_rounds=1
```

### Monitor GPU:
```bash
# Terminal 1: Run training
python -m multiround.train --config multiround/config/experiments/00_debug_fast_gpu.yaml

# Terminal 2: Monitor GPU
watch -n 0.5 'nvidia-smi | grep python'
```

### Profile Performance:
```bash
# Add --profile flag if implemented
python -m multiround.train \
  --config multiround/config/experiments/00_debug_fast_gpu.yaml \
  --profile
```

## 🔍 Debugging Tips

1. **If OOM (Out of Memory)**:
   - Reduce batch_size to 32
   - Increase grad_accum_steps to 2
   - Reduce num_workers to 8

2. **If still slow**:
   - Check data loading: add timing logs
   - Profile with `torch.profiler`
   - Try compile: true (if PyTorch 2.0+)

3. **If unstable**:
   - Reduce learning rate slightly
   - Increase warmup_steps back to 100
   - Use gradient clipping

## 💡 Key Insights for A100

- **A100 40GB**: Can handle batch_size up to 128
- **A100 80GB**: Can handle batch_size up to 256
- **Memory bandwidth**: A100 excels with large batches
- **Tensor Cores**: bf16 precision fully utilizes them
- **MIG**: Can partition GPU if needed for parallel experiments

## 📊 Monitoring Script

```python
# Save as monitor_gpu.py
import subprocess
import time

while True:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                           '--format=csv,noheader,nounits'], capture_output=True, text=True)
    gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
    print(f"GPU: {gpu_util}% | Memory: {mem_used}/{mem_total} MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
    time.sleep(1)
```

Run with: `python monitor_gpu.py`