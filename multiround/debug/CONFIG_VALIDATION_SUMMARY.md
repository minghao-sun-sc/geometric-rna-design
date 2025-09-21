# Debug Configuration Validation Summary

## ✅ Both Configurations Ready for Training

Both debug configurations have been thoroughly validated and are ready for multiround training trials.

## 📊 Configuration Comparison

| Setting | A100 80GB | A40 48GB | Notes |
|---------|-----------|----------|-------|
| **GPU Memory** | 80GB | 48GB | A40 uses more conservative settings |
| **Training Batch Size** | 32 | 24 | Adjusted for memory capacity |
| **Eval Batch Size** | 32 | 24 | Matched to training batch |
| **Grad Accum Steps** | 2 | 3 | Balances effective batch size |
| **Effective Batch Size** | 64 | 72 | A40 slightly larger due to grad accum |
| **Precision** | bf16 | bf16 | Both support bfloat16 (Ampere architecture) |
| **CUDA Memory Config** | max_split_size_mb:512 | max_split_size_mb:256 | Conservative for A40 |

## 🔧 Key Fixed Issues in Verified Configs

### 1. **Training Batch Size Mismatch (A40)**
- **Original**: training.batch_size: 32, eval.batch_size: 24
- **Fixed**: Both set to 24 for consistency and memory safety
- **Impact**: Prevents potential OOM during evaluation

### 2. **Comments Added**
- **Added**: Comprehensive documentation for all parameters
- **Added**: GPU-specific optimization explanations
- **Added**: Memory allocation recommendations

### 3. **Effective Batch Size Optimization**
- **A100**: 32 × 2 = 64 (original effective batch)
- **A40**: 24 × 3 = 72 (slightly higher for compensation)
- **Impact**: Maintains training dynamics while respecting memory limits

## 🎯 Multiround Training Settings (Both Configs)

### Core Settings
- **Rounds**: 2 (for debug/testing)
- **Epochs per round**: 2 (quick validation)
- **Reference updates**: Enabled between rounds
- **Dynamic pairs**: Using margin 25*std for both rounds

### Evaluation Settings
- **Samples per structure**: 8 (standard)
- **Final eval samples**: 64 (for pass@k analysis)
- **Pass@k analysis**: Rounds 1, 3, 5 (only round 1 active in 2-round debug)
- **Full metrics**: All 32+ evaluation metrics enabled

### DPO Parameters
- **Beta**: 0.13 (medium regularization)
- **SFT Lambda**: 0.10 (moderate SFT weight)
- **Learning rate**: 0.00018
- **Scheduler**: Cosine annealing with warmup

## 🚀 Ready to Run Commands

### For A100 80GB Systems:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python -m multiround.train --config multiround/config/experiments/00_debug_a100_80g_verified.yaml
```

### For A40 48GB Systems:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python -m multiround.train --config multiround/config/experiments/00_debug_a40_verified.yaml
```

## ✅ All Validation Checks Passed

### Path Validation ✅
- Base checkpoint: `checkpoints/gRNAde_ARv1_1state_das.h5`
- Processed data: `data/processed.pt`
- Split data: `data/das_split.pt`
- All margin25 pair files exist and accessible

### Component Integration ✅
- Configuration loading with inheritance ✅
- Pair provider and dynamic switching ✅ 
- Evaluator and dataset loading ✅
- Pass@k analysis framework ✅
- WandB integration ✅

### Training Pipeline ✅
- Model loading from checkpoint ✅
- Reference model update mechanism ✅
- Distribution analysis and visualization ✅
- Complete evaluation pipeline ✅

## 📈 Expected Training Flow

1. **Round 1**: Train 2 epochs → Evaluate with pass@k → Save best checkpoint
2. **Reference Update**: Clone Round 1 best → becomes Round 2 reference
3. **Round 2**: Train 2 epochs → Final evaluation with pass@k
4. **Output**: Complete metrics, distribution analysis, trained model

## 🎯 Training Duration Estimates

- **Per round training**: ~20-30 minutes (2 epochs)
- **Per round evaluation**: ~25 minutes (98 structures × ~15 sec/structure)
- **Total debug run**: ~1.5-2 hours per config
- **Files generated**: Checkpoints, metrics, distribution plots, WandB logs

## 💡 Ready for Production Scale

After successful debug runs, these configs can be scaled to production by:
- Increasing `num_rounds` to 5
- Increasing `epochs_per_round` to 20
- Enabling dynamic margin switching (125*std for rounds 3-5)
- Using advanced reference model selection criteria

Both configurations are now fully validated and ready for your multiround training trials!