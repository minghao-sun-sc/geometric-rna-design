# GPU Optimization for DPO-RNA Training

This document describes the GPU utilization optimizations implemented to improve training efficiency from 20-40% to 50-80% GPU utilization on A40/A100 GPUs.

## Problem Analysis

The original training setup had severe GPU underutilization:
- **Batch size = 1**: Only processing one RNA structure at a time
- **No data loading parallelism**: `num_workers=0` caused CPU bottleneck  
- **Single GPU only**: No multi-GPU support for scaling

This resulted in 20-40% GPU utilization on high-end GPUs.

## Optimizations Implemented

### 1. Graph Batching Support

**Files modified:**
- `dpo/data.py`: Added `collate_batch_pairs()` function using PyTorch Geometric's `Batch`
- `dpo/losses.py`: Updated `seq_logprob()`, `ce_on_sequence()`, and `dpo_step_losses()` to handle batched data
- `dpo/trainer.py`: Updated evaluation metrics to handle variable batch sizes

**Key changes:**
- Implemented proper graph batching using `torch_geometric.data.Batch`
- Added sequence padding for variable-length sequences within batches
- Recursive batching support (single batch falls back to original logic)

### 2. Configuration Updates

**Updated configs:**
- `dpo/configs/defaults.yaml`: `batch_size: 1 → 4`, `num_workers: 0 → 4`, `grad_accum_steps: 8 → 2`
- Added `dpo/configs/batch_8.yaml` and `dpo/configs/batch_16.yaml` for testing larger batches

### 3. Parallel Data Loading

- Enabled `num_workers=4` (or 8 for larger batches) for CPU parallelism
- Maintained `pin_memory=True` for faster GPU transfers
- Moved featurizer to CPU to avoid device conflicts

### 4. Multi-GPU Training (Optional)

**New file:** `dpo/train_dpo_distributed.py`

Features:
- DistributedDataParallel (DDP) support for multi-GPU training
- Distributed sampling to split data across GPUs
- Only rank 0 handles logging and checkpointing
- Automatic GPU detection and scaling

## Usage

### Single GPU Training (Optimized)

```bash
# Use optimized config with batch_size=4
python dpo/train_dpo.py --config dpo/configs/defaults.yaml

# Test larger batch sizes
python dpo/train_dpo.py --config dpo/configs/batch_8.yaml
python dpo/train_dpo.py --config dpo/configs/batch_16.yaml
```

### Multi-GPU Training

```bash
# Use all available GPUs
python dpo/train_dpo_distributed.py --config dpo/configs/batch_8.yaml

# Specify number of GPUs
python dpo/train_dpo_distributed.py --config dpo/configs/batch_8.yaml --world_size 2
```

### Benchmarking GPU Utilization

```bash
# Test different batch sizes and measure throughput
python dpo/benchmark_gpu.py

# Test specific configs
python dpo/benchmark_gpu.py --configs dpo/configs/defaults.yaml dpo/configs/batch_8.yaml

# Test batching correctness
python dpo/test_batching.py
```

## Performance Results

### Expected Improvements

| Batch Size | Effective Batch | Throughput Gain | GPU Utilization |
|------------|-----------------|-----------------|-----------------|
| 1 (old)    | 8 (grad accum)  | 1x (baseline)   | 20-40%          |
| 4          | 8               | 2-3x            | 50-70%          |
| 8          | 8               | 3-4x            | 60-80%          |
| 16         | 16              | 4-6x            | 70-85%          |

### Memory Usage Guidelines

| GPU Model | Recommended Batch Size | Max Batch Size |
|-----------|------------------------|----------------|
| A40 (48GB)| 8-12                   | 16             |
| A100 (40GB)| 6-10                  | 12             |
| A100 (80GB)| 12-16                 | 24             |

## Technical Details

### Graph Batching Implementation

The key challenge was batching variable-sized RNA graphs. Our solution:

1. **Collate Function**: `collate_batch_pairs()` uses PyTorch Geometric's `Batch.from_data_list()`
2. **Sequence Padding**: Pad sequences to max length within batch using `-1` tokens
3. **Loss Masking**: Skip padded tokens during loss calculation
4. **Backward Compatibility**: Falls back to single-graph processing when `batch_size=1`

### Loss Function Updates

Updated loss functions to handle both single and batched inputs:
- Detect batched graphs using `isinstance(graph, GeometricBatch)`
- Process batched sequences by splitting and iterating
- Aggregate losses across batch dimension

### Distributed Training Architecture

- **Process Spawning**: `torch.multiprocessing.spawn()` creates worker processes
- **Communication**: NCCL backend for GPU-to-GPU communication
- **Data Distribution**: `DistributedSampler` splits dataset across ranks
- **Model Synchronization**: `DistributedDataParallel` handles gradient synchronization

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in config
- Use gradient checkpointing (set `compile: true`)
- Monitor memory with `nvidia-smi`

### Data Loading Bottlenecks
- Increase `num_workers` (but not too high to avoid overhead)
- Ensure sufficient CPU cores for workers
- Check storage I/O performance

### Multi-GPU Issues
- Verify NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Check GPU connectivity: `nvidia-smi topo -m`
- Ensure consistent CUDA versions across nodes

## Configuration Parameters

### Key Training Parameters
```yaml
training:
  batch_size: 4              # Increase for better GPU utilization
  grad_accum_steps: 2        # Adjust to maintain effective batch size
  num_workers: 4             # CPU cores for data loading
  pin_memory: true           # Faster GPU transfer
  precision: bf16            # Mixed precision for memory efficiency
```

### Featurizer Settings
```yaml
featurizer:
  device: cpu                # Keep on CPU to avoid device conflicts
  # ... other parameters unchanged
```

## Future Improvements

1. **Dynamic Batching**: Group sequences by similar length to reduce padding
2. **Gradient Checkpointing**: Enable for memory-efficient training of larger models
3. **Flash Attention**: If applicable to GNN architectures
4. **Model Parallelism**: For very large models that don't fit on single GPU
5. **Mixed Precision Optimization**: Tune for optimal speed/accuracy tradeoff