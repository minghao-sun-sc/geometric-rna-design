# DPO-RNA: Direct Preference Optimization for RNA Inverse Folding

## 📋 Project Overview

DPO-RNA implements **Direct Preference Optimization** (DPO) to fine-tune the gRNAde RNA inverse folding model, improving its ability to design RNA sequences that fold into target 3D structures. This project adapts the protein DPO framework to RNA, using structural quality metrics to create preference pairs for training.

### Key Features
- 🚀 **Zero-modification approach**: No changes to original gRNAde codebase (`src/`, `gRNAde.py`)
- 📊 **Offline preference learning**: Pre-computed structural quality scores (RhoFold RMSD/pLDDT, ViennaRNA MFE)
- 🎯 **Multi-metric optimization**: Balances fold accuracy, stability, and sequence quality
- 📈 **Comprehensive evaluation**: Recovery, perplexity, 2D/3D self-consistency metrics

## 🧬 Background

**Challenge**: RNA inverse folding (designing sequences for target 3D structures) is critical for RNA therapeutics and synthetic biology, but current models often generate sequences that don't fold correctly.

**Solution**: We use DPO to learn from preference pairs where "winner" sequences have better structural properties than "loser" sequences, teaching the model to generate higher-quality designs.

**Base Model**: [gRNAde](https://github.com/chaitjo/geometric-rna-design) - a geometric deep learning model for RNA design that conditions on 3D backbone structures.

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
# Basic training
python -m dpo.train_dpo --config dpo/configs/defaults.yaml --run_name dpo_rna_v5

# Resume from checkpoint
python -m dpo.train_dpo --config dpo/configs/defaults.yaml --run_name dpo_rna_v5_resume --resume runs/offline_dpo/dpo_rna_v5/best.pt

```

### Configuration (`dpo/configs/defaults.yaml`)
```yaml
# Key hyperparameters
dpo:
  beta: 0.163           # DPO temperature (controls preference strength)
  sft_lambda: 0.152     # SFT regularization on winners
  
training:
  epochs: 20
  batch_size: 1         # Per-GPU batch size
  grad_accum_steps: 8   # Effective batch size = 8
  
optimizer:
  lr: 2.0e-4
  warmup_steps: 1000
```

### Training Algorithm
- **Objective**: Reference-tethered DPO loss + SFT anchor on winners
- **Reference reset**: At the start of each training round
- **Checkpoint selection**: Best validation preference accuracy
- **Device handling**: Automatic CUDA/CPU fallback with RBF error protection

### Monitoring
Training logs to [Weights & Biases](https://wandb.ai):
- `train/loss_dpo`: DPO loss
- `train/pref_acc`: Preference accuracy (% correct rankings)
- `train/margin`: Average logprob margin (winner - loser)
- `val/pref_acc`: Validation preference accuracy (for best.pt selection)

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

If you use this work, please cite:

```bibtex
@software{dpo_rna_2024,
  title = {DPO-RNA: Direct Preference Optimization for RNA Inverse Folding},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/[your-repo]/offline-dpo}
}

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

For questions or issues, please open an issue on GitHub or contact [your email].

---

*Last updated: September 2024*