# RNA Preference Optimization: DPO & SimPO for RNA Inverse Folding

## 📋 Project Overview

This project implements **preference optimization methods** (DPO & SimPO) to fine-tune the gRNAde RNA inverse folding model, improving its ability to design RNA sequences that fold into target 3D structures. We support both **Direct Preference Optimization (DPO)** and **Simplified Preference Optimization (SimPO)** for RNA design.

### Key Features
- 🚀 **Zero-modification approach**: No changes to original gRNAde codebase (`src/`, `gRNAde.py`)
- 📊 **Offline preference learning**: Pre-computed structural quality scores (RhoFold RMSD/pLDDT, ViennaRNA MFE)
- 🎯 **Dual optimization methods**: Both DPO (reference-based) and SimPO (reference-free)
- 📈 **Comprehensive evaluation**: Recovery, perplexity, 2D/3D self-consistency metrics
- ⚡ **Optimized training**: Graph batching for improved GPU utilization

## 🧬 Background

**Challenge**: RNA inverse folding (designing sequences for target 3D structures) is critical for RNA therapeutics and synthetic biology, but current models often generate sequences that don't fold correctly.

**Solution**: We use preference optimization methods to learn from preference pairs where "winner" sequences have better structural properties than "loser" sequences, teaching the model to generate higher-quality designs.

**Base Model**: [gRNAde](https://github.com/chaitjo/geometric-rna-design) - a geometric deep learning model for RNA design that conditions on 3D backbone structures.

## 🔬 Optimization Methods

### Direct Preference Optimization (DPO)
- **Reference-based**: Uses a frozen reference model for stability
- **Loss**: `-log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))`
- **Memory**: Higher (stores both policy and reference models)
- **Training**: More stable but slower

### Simplified Preference Optimization (SimPO) ⭐
- **Reference-free**: No reference model needed - more memory efficient
- **Loss**: `-log σ(β * (avg_logp_w - avg_logp_l) - γ)` with length normalization
- **Memory**: Lower (only policy model)
- **Training**: Faster convergence, easier hyperparameter tuning
- **Hyperparameters**: `β ∈ [1.5, 2.5]`, `γ ∈ [0.3, 1.2]`

**Recommendation**: Start with **SimPO** as it's more efficient and easier to tune.

## 📊 Dataset

### Preference Pair Construction
1. **Generation**: 10 candidate sequences per RNA backbone using gRNAde
2. **Scoring**: 
   - **3D Structure**: RhoFold predictions (RMSD, pLDDT)
   - **Energy**: ViennaRNA MFE calculations
3. **Filtering**:
   - Remove poor predictions (RMSD ≥ 128Å or pLDDT < 0.30)
   - Winners: RMSD < 8Å AND pLDDT > 0.7
   - Losers: Opposite criteria
   - Confidence margin: 0.125σ difference required

### Dataset Statistics
```
Location: data/pairs_margin125/by_das/clean/
├── train.clean.jsonl  (23,850 pairs from 2,385 structures)
├── val.clean.jsonl    (100 pairs)
└── test.clean.jsonl   (98 pairs)
```

### Data Format (JSONL)
```json
{
  "pdb_file": "1A34_1_B-C.pdb",
  "winner_seq": "AUGCGCUAGCUA...",
  "loser_seq": "AUGCGCUAGCUG...",
  "weight": 0.85
}
```

## 🚀 Training

### Quick Start
```bash
# SimPO/DPO training (default, recommended)
python -m dpo.train --config dpo/configs/defaults.yaml

# Evaluation Pipeline
# Full Evaluation (with Sampling)

python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml 

# Alternative parameters: --n_samples 8 --temperature 0.5

# SBATCH; training & eval

sbatch dpo/scripts/train.slurm dpo/configs/defaults.yaml

# or

sbatch dpo/scripts/train.slurm dpo/configs/experiments/exp003_simpo_b2.0_g0.8_l0.05_b16.yaml

# From the project root 'ribopo/, SoC Cluster'
sbatch dpo/scripts/eval.slurm dpo/configs/bench_full.yaml

# From the DNA Cluster
sbatch dpo/scripts/eval_dna.slurm dpo/configs/bench_full.yaml


# --n_samples 8 --temperature 0.5 optional, can override



```

The above commands are frequently used. 











```bash
# Basic Evaluation (Teacher-forced)

python -m dpo.bench.eval_benchmark --config dpo/configs/bench.yaml

# Metrics: Recovery, Perplexity



# DPO training (override loss_type)
python -m dpo.train --config dpo/configs/defaults.yaml --loss_type dpo

# Override run name
python -m dpo.train --config dpo/configs/defaults.yaml --run_name custom_experiment_name

# Disable wandb logging
python -m dpo.train --config dpo/configs/defaults.yaml --wandb_mode disabled

# Offline mode (for limited internet)
python -m dpo.train --config dpo/configs/defaults.yaml --wandb_mode offline
```

### SLURM Job Submission
```bash
# Submit SimPO job (recommended)
sbatch --gres=gpu:A100:1 --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --wrap="python -m dpo.train --config dpo/configs/defaults.yaml"

# Submit DPO job
sbatch --gres=gpu:A100:1 --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --wrap="python -m dpo.train --config dpo/configs/defaults.yaml --loss_type dpo"

# Using job script
sbatch examples/submit_job.sh dpo/configs/defaults.yaml
```

**Example job scripts** are provided in `examples/` - just update the paths and module loading for your cluster.

### Available Configurations

#### `dpo/configs/defaults.yaml` - SimPO Configuration (Recommended)
```yaml
loss_type: simpo        # "dpo" or "simpo"

wandb:
  enable: true
  project: DPO-RNA
  entity: minghao-sun-soc
  run_name: simpo_rna_train

training:
  batch_size: 4         # Optimized GPU utilization
  grad_accum_steps: 4   # Effective batch size = 16
  num_workers: 8        # Adjust based on CPU allocation
  
# SimPO parameters (used when loss_type=simpo)
simpo:
  beta: 2.0             # β ∈ [1.5, 2.5] - reward scale
  gamma: 0.5            # γ ∈ [0.3, 1.2] - target margin
  sft_lambda: 0.0       # Optional SFT regularization

# DPO parameters (used when loss_type=dpo)  
dpo:
  beta: 0.10            # DPO temperature
  sft_lambda: 0.10      # SFT regularization
  
optimizer:
  lr: 1.0e-4
```

### Hyperparameter Guidelines

#### SimPO Hyperparameters
- **β (beta)**: `1.5-2.5` - Controls reward scaling. Higher β = stronger preference signal
- **γ (gamma)**: `0.3-1.2` - Target reward margin. Start with `0.5`
- **Learning Rate**: `1e-4` to `5e-6` - Use lower LR for stable training
- **SFT λ**: `0.0-0.1` - Optional regularization on winners

#### DPO Hyperparameters  
- **β (beta)**: `0.1-0.5` - DPO temperature parameter
- **Learning Rate**: `1e-4` to `2e-4`
- **SFT λ**: `0.1-0.2` - Regularization weight

### GPU Utilization Optimization
The training pipeline has been optimized for high GPU utilization:
- **Graph Batching**: Using PyTorch Geometric's `Batch` for proper RNA graph batching
- **Parallel Data Loading**: Multi-worker data preprocessing (requires adequate CPU allocation)
- **Mixed Precision**: BFloat16 for memory efficiency

#### Resource Requirements
| Batch Size | GPU Memory | CPU Cores | Expected GPU Util | Throughput |
|------------|------------|-----------|-------------------|------------|
| 4          | ~15GB      | 4         | 50-70%           | 3-4x       |
| 8          | ~25GB      | 8         | 60-80%           | 5-6x       |
| 16         | ~40GB      | 8         | 70-85%           | 7-8x       |


### Training Algorithm

#### SimPO (Reference-free)
- **Objective**: Length-normalized preference optimization with target margin
- **Loss**: `-log σ(β * (avg_logp_w - avg_logp_l) - γ)`
- **Memory Efficient**: Only policy model needed
- **Checkpoint selection**: Best validation reward accuracy

#### DPO (Reference-based)
- **Objective**: Reference-tethered DPO loss + SFT anchor on winners
- **Reference reset**: At the start of each training round
- **Memory**: Policy + frozen reference model
- **Checkpoint selection**: Best validation preference accuracy

### Monitoring
Training logs to [Weights & Biases](https://wandb.ai):

#### SimPO Metrics
- `train/simpo/reward_acc`: % with β*(avg_w-avg_l) > γ (key metric)
- `train/simpo/avg_logp_w`: Average log-prob for winners
- `train/simpo/avg_logp_l`: Average log-prob for losers
- `train/simpo/z_margin`: Decision margin (β*(avg_w-avg_l) - γ)
- `train/simpo/sft_loss`: SFT regularization loss (if enabled)

#### DPO Metrics
- `train/loss_dpo`: DPO loss
- `train/pref_acc`: Preference accuracy (% correct rankings)
- `train/margin`: Average logprob margin (winner - loser)

#### Common Metrics
- `train/loss`: Total training loss
- `train/lr`: Learning rate
- **Config logging**: All hyperparameters automatically logged for easy filtering

## 📈 Evaluation & Benchmarking

### Evaluation Pipeline

#### 1. Basic Evaluation (Teacher-forced)
```bash
python -m dpo.bench.eval_benchmark --config dpo/configs/bench.yaml
```
Metrics: Recovery, Perplexity

#### 2. Full Evaluation (with Sampling)
```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml --n_samples 8 --temperature 0.5
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recovery** | % of native sequence recovered | 55-60% |
| **Perplexity** | Model confidence (lower=better) | 1.2-1.4 |
| **2D Self-consistency** | EternaFold MCC score | 65-70% |
| **3D Self-consistency** | RhoFold RMSD/TM-score/GDT | Variable |

### Benchmarking Results

#### Baseline (gRNAde)
```
Test Set (98 structures, 8 samples, temp=0.5):
- Recovery: 51.98%
- Perplexity: 1.39
- 2D Self-consistency: 60.32%
- 3D RMSD: 11.13Å
```

#### DPO Fine-tuned
```
[To be updated after training completion]
- Recovery: Target 55-60%
- Perplexity: Target 1.2-1.4
- 2D Self-consistency: Target 65-70%
```

### Running Complete Benchmark
```bash
# Evaluate both base and DPO models
python -m dpo.bench.eval_full \
    --config dpo/configs/bench_full.yaml \
    --checkpoints \
        checkpoints/gRNAde_ARv1_1state_das.h5 \
        runs/offline_dpo/dpo_rna_v3/best.pt \
    --n_samples 8 \
    --temperature 0.5
```

## 🗂️ Project Structure

```
dpo/
├── train_dpo.py           # Main training script
├── trainer.py             # DPO trainer class
├── data.py                # Dataset & dataloader (with RBF fix)
├── losses.py              # DPO loss implementation
├── ref_manager.py         # Reference model management
├── utils.py               # Utilities
│
├── bench/                 # Evaluation suite
│   ├── eval_benchmark.py  # Basic metrics
│   ├── eval_full.py       # Full evaluation with self-consistency
│   └── eval.md            # Evaluation documentation
│
├── configs/               # Configuration files
│   ├── defaults.yaml      # Training config
│   ├── bench_full.yaml    # Full evaluation config
│   └── bench_full_test.yaml # Test evaluation config
│
├── scripts/               # Data processing scripts
│   ├── split_and_filter_pairs.py
│   └── build_master_report.py
│
└── debug/                 # Debug & analysis tools
    ├── performance_analysis_summary.md
    └── analyze_sequence_quality.py
```

## 🔧 Technical Implementation

### Key Design Choices

1. **No gRNAde Modifications**: All DPO logic contained in `dpo/` folder
2. **Error Resilience**: Graceful handling of featurization failures (RBF expansion errors)
3. **Memory Efficiency**: Single-graph batching with gradient accumulation
4. **Device Flexibility**: CPU featurization to avoid CUDA issues

### DPO Loss Formulation
```python
L_DPO = -log(σ(β * (log π(w|x)/π_ref(w|x) - log π(l|x)/π_ref(l|x))))
L_SFT = -log π(w|x)  # Winners only
L_total = L_DPO + λ * L_SFT
```

Where:
- `w`: Winner sequence
- `l`: Loser sequence  
- `x`: RNA backbone structure
- `β`: Temperature parameter (0.163)
- `λ`: SFT weight (0.152)

### Handling Edge Cases

The implementation includes robust error handling for:
- Length mismatches between sequences and graphs
- Empty edge tensors in featurization
- Device mismatches (CPU/CUDA)
- Missing or corrupted structures

## 📝 Citation

@article{grnade2024,
  title = {gRNAde: Geometric RNA Design},
  author = {Joshi, Chaitanya K. and others},
  journal = {ICLR},
  year = {2024}
}
```

## 🤝 Acknowledgments

- **gRNAde**: Base model and infrastructure
- **RhoFold**: 3D structure prediction for evaluation
- **ViennaRNA**: Secondary structure and energy calculations
- **Protein DPO**: Methodology inspiration

## 📧 Contact

---

*Last updated: September 2024*